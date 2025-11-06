import sys
import os
import json
import time
import asyncio
import signal
import threading
import queue as qmod
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from insightface.utils.face_align import norm_crop as _norm_crop_112
import numpy as np
import cv2
from fractions import Fraction
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Query,
    Body,
    UploadFile,
    File,
    Form,
    Depends,
    Request,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator
from starlette.websockets import WebSocketState
import logging
import pytz
import jwt
import datetime as dt
import secrets
import uuid
import re

# InsightFace
from insightface.app import FaceAnalysis

# Project modules
from service.database import Database
from service.car_camera_worker import CarCameraWorker
from service.face_camera_worker import FaceCameraWorker
from service.record import VideoRecorder
from service import utils
from insightface.utils.face_align import norm_crop as _norm_crop_112

logger = logging.getLogger(__name__)
face_app: Optional[FaceAnalysis] = None

class Formatter(logging.Formatter):
    def converter(self, timestamp):
        dt_ = datetime.fromtimestamp(timestamp)
        tzinfo = pytz.timezone("Asia/BangKOK")
        return tzinfo.localize(dt_)

    def formatTime(self, record, datefmt=None):
        dt_ = self.converter(record.created)
        if datefmt:
            s = dt_.strftime(datefmt)
        else:
            s = dt_.isoformat(timespec="milliseconds")
        return s

handler = logging.StreamHandler()
handler.setFormatter(Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
logging.getLogger('asyncio').setLevel(logging.ERROR)

tz = pytz.timezone("Asia/BangKOK")
now = datetime.now(tz)
logger.info(f"Current real time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# ===================== Environment Variables =====================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"[WARN] dotenv import failed: {e}")

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "server_settings.json" # ⭐️ [เพิ่มใหม่]
BIN_DIR = APP_DIR / "bin"

# Add bundled ffmpeg to PATH if present
if "FFMPEG_BIN" not in os.environ:
    ff = BIN_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if ff.exists():
        os.environ["FFMPEG_BIN"] = str(ff)
os.environ["PATH"] = str(BIN_DIR) + os.pathsep + os.environ.get("PATH", "")

RECORD_ROOT = os.getenv("RECORD_ROOT", os.path.abspath(os.path.join(os.getcwd(), "recordings")))
os.makedirs(RECORD_ROOT, exist_ok=True)

# (ลบ SEGMENT_MINUTES เก่าออก)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))
RECORD_FPS_HINT = int(os.getenv("RECORD_FPS_HINT", "25"))
security = HTTPBearer(auto_error=False)

# ---- RTSP via FFMPEG with TCP + lower latency ----
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|max_delay;500000|stimeout;5000000|buffer_size;2097152"
)

# ======= Face det sizes (ตามที่ขอ) =======
os.environ.setdefault("ENROLL_DET_SIZE", "512,512")
os.environ.setdefault("ENROLL_DET_THRESH", "0.30")
os.environ.setdefault("RUNTIME_DET_SIZE", "1280,1280")
os.environ.setdefault("RUNTIME_DET_THRESH", "0.50")
os.environ.setdefault("FACE_MODEL_NAME", "buffalo_l")

JPEG_QUALITY  = int(os.getenv("JPEG_QUALITY", "80"))
JPEG_OPTIMIZE = int(os.getenv("JPEG_OPTIMIZE", "1"))
MAX_JPEG_SIDE = int(os.getenv("MAX_JPEG_SIDE", "1280"))
STALE_SEC     = float(os.getenv("STALE_SEC", "2.5"))
BLACK_H       = int(os.getenv("BLACK_H", "480"))
BLACK_W       = int(os.getenv("BLACK_W", "854"))
BLACK_480P    = np.zeros((BLACK_H, BLACK_W, 3), dtype=np.uint8)

# ===================== JWT Helpers =====================
class TokenClaims(BaseModel):
    sub: str
    department: str = ""
    access: List[str] = []
    is_admin: bool = False
    iat: int
    exp: int
    must_change: bool = False
    sid: Optional[str] = None
    jti: Optional[str] = None

def create_access_token(username: str, department: str, access: List[str], is_admin: bool,
                        ttl_minutes: Optional[int] = None) -> str:
    now_ = dt.datetime.now(tz)
    ttl = int(ttl_minutes if ttl_minutes is not None else JWT_EXPIRE_MINUTES)
    sid = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    payload = {
        "sub": username,
        "department": department or "",
        "access": access or [],
        "is_admin": bool(is_admin),
        "must_change": False,
        "sid": sid,
        "jti": jti,
        "iat": int(now_.timestamp()),
        "exp": int((now_ + dt.timedelta(minutes=ttl)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def create_temp_login_token(username: str) -> str:
    now_ = dt.datetime.now(tz)
    temp_minutes = 30
    sid = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    payload = {
        "sub": username,
        "department": "",
        "access": [],
        "is_admin": False,
        "must_change": True,
        "sid": sid,
        "jti": jti,
        "iat": int(now_.timestamp()),
        "exp": int((now_ + dt.timedelta(minutes=temp_minutes)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

async def get_token_from_query(
    token: str = Query(None, description="Auth token from query parameter")
) -> Optional[str]:
    """
    (สำหรับ WS) ดึง Token จาก Query parameter 'token' เท่านั้น
    """
    if token:
        logger.debug("Auth (WS): Found token in query parameter.")
        return token
    logger.debug("Auth (WS): No token found in query.")
    return None
# --- ⭐️ [สิ้นสุด] ⭐️ ---
async def get_token_from_header_or_query(
    token: str = Query(None, description="Auth token from query parameter"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    if token:
        logger.debug("Auth: Found token in query parameter.")
        return token
    if credentials:
        logger.debug("Auth: Found token in Authorization header.")
        return credentials.credentials
    logger.debug("Auth: No token found in query or header.")
    return None

async def require_user_flexible(
    token_str: Optional[str] = Depends(get_token_from_header_or_query)
) -> TokenClaims:
    if token_str is None:
        logger.warning("Auth Flexible: Authentication failed - Missing Token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated (Missing Token)"
        )
    try:
        unverified_payload = jwt.decode(token_str, options={"verify_signature": False}, algorithms=[JWT_ALG])
        unverified_claims = TokenClaims(**unverified_payload)
        user = unverified_claims.sub
        sid = unverified_claims.sid
        jti = unverified_claims.jti

        if not (sid and jti and user):
             logger.warning(f"Auth Flexible: Invalid session data in token for user {user or '?'}")
             raise HTTPException(status_code=401, detail="invalid_session_data_in_token")

        if not session_is_active(user, sid, jti):
             logger.warning(f"Auth Flexible: Session revoked for user {user}, sid {sid}")
             raise HTTPException(status_code=401, detail="session_revoked")

        payload = jwt.decode(token_str, JWT_SECRET, algorithms=[JWT_ALG])
        claims = TokenClaims(**payload)

        logger.debug(f"Auth Flexible: User '{claims.sub}' authenticated successfully.")
        return claims

    except jwt.ExpiredSignatureError:
        logger.warning(f"Auth Flexible: Token expired for user {user if 'user' in locals() else '?'}")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Auth Flexible: Invalid token - {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
         logger.error(f"Auth Flexible: Unexpected error during token validation: {e}", exc_info=True)
         raise HTTPException(status_code=401, detail=f"Token validation error: {e}")
    
async def require_user_ws(
    token_str: Optional[str] = Depends(get_token_from_query) # 1. (ใช้ฟังก์ชันใหม่)
) -> TokenClaims:
    """
    (สำหรับ WS) ตรวจสอบ Token ที่ได้จาก Query parameter เท่านั้น
    """
    if token_str is None:
        logger.warning("Auth (WS): Authentication failed - Missing Token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated (Missing Token)"
        )
    try:
        # (ตรรกะที่เหลือเหมือน require_user_flexible)
        unverified_payload = jwt.decode(token_str, options={"verify_signature": False}, algorithms=[JWT_ALG])
        unverified_claims = TokenClaims(**unverified_payload)
        user = unverified_claims.sub
        sid = unverified_claims.sid
        jti = unverified_claims.jti

        if not (sid and jti and user):
             logger.warning(f"Auth (WS): Invalid session data in token for user {user or '?'}")
             raise HTTPException(status_code=401, detail="invalid_session_data_in_token")

        if not session_is_active(user, sid, jti):
             logger.warning(f"Auth (WS): Session revoked for user {user}, sid {sid}")
             raise HTTPException(status_code=401, detail="session_revoked")

        payload = jwt.decode(token_str, JWT_SECRET, algorithms=[JWT_ALG])
        claims = TokenClaims(**payload)

        logger.debug(f"Auth (WS): User '{claims.sub}' authenticated successfully.")
        return claims
    
    except jwt.ExpiredSignatureError:
        logger.warning(f"Auth (WS): Token expired for user {user if 'user' in locals() else '?'}")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Auth (WS): Invalid token - {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
         logger.error(f"Auth (WS): Unexpected error during token validation: {e}", exc_info=True)
         raise HTTPException(status_code=401, detail=f"Token validation error: {e}")
# --- ⭐️ [สิ้นสุด] ⭐️ ---
def require_admin_flexible(claims: TokenClaims = Depends(require_user_flexible)) -> TokenClaims:
    if not claims.is_admin:
        logger.warning(f"Authorization Failed: User '{claims.sub}' attempted admin action without admin rights.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator rights required")
    logger.debug(f"Authorization Granted: User '{claims.sub}' has admin rights.")
    return claims

active_sessions: dict[str, dict] = {}

def _db_has_session_table() -> bool:
    return db is not None and all(
        hasattr(db, m) for m in (
            "session_has_active", "session_register", "session_revoke_user",
            "session_revoke_one", "session_is_active"
        )
    )

def session_has_active(user: str) -> bool:
    if db and _db_has_session_table():
        try:
            return bool(db.session_has_active(user))
        except Exception:
            pass
    ent = active_sessions.get(user)
    if not ent:
        return False
    if ent.get("exp", 0) <= int(dt.datetime.now(tz).timestamp()):
        active_sessions.pop(user, None)
        return False
    return ent.get("active", False)

def session_register(user: str, sid: str, jti: str, exp: int, user_agent: str = "", ip: Optional[str] = None):
    if db and _db_has_session_table():
        try:
            db.session_register(user_name=user, sid=sid, jti=jti, exp=exp, user_agent=user_agent, ip=ip)
            return
        except Exception:
            pass
    active_sessions[user] = {"sid": sid, "jti": jti, "exp": exp, "active": True}

def session_revoke_user(user: str):
    if db and _db_has_session_table():
        try:
            db.session_revoke_user(user)
            return
        except Exception:
            pass
    if user in active_sessions:
        active_sessions[user]["active"] = False

def session_revoke_one(user: str, sid: str):
    if db and _db_has_session_table():
        try:
            db.session_revoke_one(user, sid)
            return
        except Exception:
            pass
    ent = active_sessions.get(user)
    if ent and ent.get("sid") == sid:
        ent["active"] = False

def session_is_active(user: str, sid: str, jti: str) -> bool:
    if db and _db_has_session_table():
        try:
            return bool(db.session_is_active(user, sid, jti))
        except Exception:
            pass
    ent = active_sessions.get(user)
    if not ent:
        return False
    if ent.get("sid") != sid or ent.get("jti") != jti:
        return False
    if ent.get("exp", 0) <= int(dt.datetime.now(tz).timestamp()):
        ent["active"] = False
        return False
    return ent.get("active", True)

def get_claims(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[TokenClaims]:
    if credentials is None:
        return None
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return TokenClaims(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_user(claims: Optional[TokenClaims] = Depends(get_claims)) -> TokenClaims:
    if claims is None:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    sid, jti, user = claims.sid, claims.jti, claims.sub
    if not (sid and jti and user):
        raise HTTPException(status_code=401, detail="invalid_session")
    if not session_is_active(user, sid, jti):
        raise HTTPException(status_code=401, detail="session_revoked")
    return claims

def require_admin(claims: TokenClaims = Depends(require_user)) -> TokenClaims:
    if not claims.is_admin:
        raise HTTPException(status_code=403, detail="IT admin required")
    return claims

# ===================== InsightFace Setup (dual apps) =====================
def _get_ort_providers():
    req = [p.strip() for p in os.getenv("INSIGHTFACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",") if p.strip()]
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        logger.info(f"[INSIGHT] Available providers: {list(avail)}")
        use = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else [p for p in req if p in avail] or ["CPUExecutionProvider"]
        logger.info(f"[INSIGHT] Requested={req} -> Using={use}")
        return use
    except Exception as e:
        logger.warning(f"[INSIGHT] onnxruntime not available or error occurred: {e}, fallback to CPU.")
        return ["CPUExecutionProvider"]

_runtime_face_app = None
_enroll_face_app = None

def _build_face_app(det_size, det_thresh):
    providers = _get_ort_providers()
    app = FaceAnalysis(name=os.getenv("FACE_MODEL_NAME", "buffalo_l"), providers=providers)
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    rec_override = os.getenv("FACE_REC_OVERRIDE", "").strip()
    if rec_override:
        logger.warning("[INSIGHT] FACE_REC_OVERRIDE ถูกตั้งค่า แต่ API ปัจจุบันไม่รองรับ .model.load; ข้ามไปก่อน")
    return app

def get_runtime_face_app():
    global _runtime_face_app
    if _runtime_face_app is None:
        w, h = os.getenv("RUNTIME_DET_SIZE", "1280,1280").split(",")
        thr = float(os.getenv("RUNTIME_DET_THRESH", "0.5"))
        _runtime_face_app = _build_face_app((int(w), int(h)), thr)
        logger.info(f"[Face] Runtime FaceAnalysis ready ({w}x{h}, thr={thr})")
    return _runtime_face_app

def get_enroll_face_app():
    global _enroll_face_app
    if _enroll_face_app is None:
        w, h = os.getenv("ENROLL_DET_SIZE", "512,512").split(",")
        thr = float(os.getenv("ENROLL_DET_THRESH", "0.3"))
        _enroll_face_app = _build_face_app((int(w), int(h)), thr)
        logger.info(f"[Face] Enroll FaceAnalysis ready ({w}x{h}, thr={thr})")
    return _enroll_face_app

def compute_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    app = get_runtime_face_app()
    try:
        faces = app.get(image)
        if not faces:
            logger.warning("[compute_embedding] No faces detected in image")
            return None
        return faces[0].embedding
    except Exception as e:
        logger.error(f"[compute_embedding] Error: {e}")
        return None

# ===================== FastAPI App =====================
app = FastAPI(title="CCTV Headless Server (Stable Mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Global Variables =====================
db: Optional[Database] = None
workers: List[tuple] = []
latest_frame: Dict[str, np.ndarray] = {}
latest_frame_sub: Dict[str, np.ndarray] = {}
preview_mode: Dict[str, str] = {}
sub_preview_threads: Dict[str, tuple] = {}
latest_lock = threading.Lock()
event_subscribers: List[WebSocket] = []
camera_meta_by_name: Dict[str, Dict[str, Any]] = {}
latest_frame_ts: Dict[str, float] = {}
camera_last_known_status: Dict[str, str] = {}
camera_down_timestamp: Dict[str, float] = {}
health_monitor_lock = threading.Lock()
HEALTH_CHECK_INTERVAL_SEC = 10

# ⭐️ [แก้ไข] โหลดค่า Default จาก .env
runtime_settings = {
    "SEGMENT_MINUTES": int(os.getenv("SEGMENT_MINUTES", "15"))
}
runtime_settings_lock = threading.Lock()
# ===================== Hot-reload known faces =====================
def _l2norm_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return (x / n).astype(np.float32)

def refresh_all_face_workers() -> int:
    refreshed = 0
    for (w, *_rest) in workers:
        if not isinstance(w, FaceCameraWorker):
            continue
        try:
            if hasattr(w, "reload_known_faces") and callable(getattr(w, "reload_known_faces")):
                w.reload_known_faces()
                refreshed += 1
                continue

            if w.db is None:
                continue
            known = w.db.load_known_faces()
            with w._kn_lock:
                w.known_ids  = [r[0] for r in known]
                w.known_names = [r[2] for r in known]
                w.known_depts = [(r[3] or "Unknown") for r in known]
                if known:
                    embs = np.stack([r[1] for r in known]).astype(np.float32)
                    w.known_embs = _l2norm_np(embs)
                else:
                    w.known_embs = np.empty((0, 512), dtype=np.float32)
            refreshed += 1
            logger.info(f"[ENROLL] Hot-reloaded {len(known)} known faces into worker {w.name}")
        except Exception as e:
            logger.error(f"[ENROLL] refresh worker {getattr(w,'name','?')} failed: {e}")
    return refreshed

# ===================== Helpers =====================
def _norm_zone(z: Optional[str]) -> Optional[str]:
    if not z:
        return None
    s = str(z).strip().lower()
    if s in ("building", "face", "people", "person"):
        return "face"
    if s in ("vehicle", "vehicles", "car"):
        return "car"
    return s

def _infer_sub_url_from_main(main_url: str) -> Optional[str]:
    if not main_url:
        return None
    try:
        u = urlparse(main_url)
        qs = dict(parse_qsl(u.query))
        if "subtype" in qs:
            qs["subtype"] = "1"
            return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(qs), u.fragment))
        if "/Streaming/Channels/" in u.path:
            tail = u.path.split("/Streaming/Channels/")[-1]
            if tail and tail.isdigit() and len(tail) == 3 and tail.endswith("1"):
                new_tail = tail[:-1] + "2"
                new_path = u.path[:-3] + new_tail
                return urlunparse((u.scheme, u.netloc, new_path, u.params, u.query, u.fragment))
    except Exception:
        pass
    return None

def _even_pad(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_bottom = h & 1
    pad_right = w & 1
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    return frame

def _open_rtsp(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    return cap

def encode_frame_to_jpeg(frame: np.ndarray) -> bytes:
    try:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        side = max(h, w)
        if side > MAX_JPEG_SIDE:
            scale = MAX_JPEG_SIDE / float(side)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        if JPEG_OPTIMIZE:
            params += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            logger.error(f"[MJPG] Failed to encode frame to JPEG")
            return b""
        return buf.tobytes()
    except Exception as e:
        logger.error(f"[MJPG] Error encoding frame: {e}")
        return b""

# ===================== Models =====================
class CameraIn(BaseModel):
    camera_name: str
    url: str
    url2: Optional[str] = None
    zone: str
    comp: Optional[str] = None
    @field_validator("url")
    @classmethod
    def must_be_rtsp(cls, v: str):
        v = (v or "").replace("\\", "/").strip()
        if not (v.startswith("rtsp://") or v.startswith("rtsps://")):
            raise ValueError("url ต้องเป็น RTSP (rtsp:// หรือ rtsps://)")
        return v

    @field_validator("url2")
    @classmethod
    def must_be_rtsp_optional(cls, v: Optional[str]):
        if v is None:
            return v
        v = (v or "").replace("\\", "/").strip()
        if not (v.startswith("rtsp://") or v.startswith("rtsps://")):
            raise ValueError("url2 ต้องเป็น RTSP (rtsp:// หรือ rtsps://)")
        return v

# ===================== Worker and Recorder Spawner =====================
def spawn_camera(cam: dict) -> bool:
    camera_name = str(cam["camera_name"])
    zone = _norm_zone(cam.get("zone")) or "face"
    url = cam["url"]
    comp = cam.get("comp")

    for (w, *_rest) in workers:
        if getattr(w, "name", None) == camera_name:
            logger.info(f"[SPAWN] Camera {camera_name} already running")
            return False

    frame_q = qmod.Queue(maxsize=3000)
    log_q = qmod.Queue()
    status_q = qmod.Queue()

    worker = None
    recorder = None
    
    try:
        if zone == "car":
            logger.info(f"🚗 [SPAWN CAR {camera_name}] URL: {url}")
            worker = CarCameraWorker(
                id=cam.get("id"),
                name=camera_name,
                url=url,
                zone=zone,
                department=comp,
                user_access=[comp] if comp else [],
            )
        else:
            logger.info(f"👤 [SPAWN FACE {camera_name}] URL: {url}")
            worker = FaceCameraWorker(
                id=cam.get("id"),
                name=camera_name,
                camera_url=url,
                zone=zone,
                department=comp,
                user_access=[comp] if comp else [],
            )
        
        # --- ⭐️ [แก้ไข] อ่านค่าจาก runtime_settings ⭐️ ---
        with runtime_settings_lock:
            current_segment_minutes = runtime_settings.get("SEGMENT_MINUTES", 15)

        recorder = VideoRecorder(
            output_dir=RECORD_ROOT,
            zone=zone,
            segment_minutes=current_segment_minutes, # ⭐️ (ใช้ค่านี)
            department=comp or "Unknown",
            camera_id=camera_name,
        )
        # --- ⭐️ [สิ้นสุด] ⭐️ ---
        
        test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not test_cap.isOpened():
            logger.error(f"❌ [SPAWN {camera_name}] URL INVALID: {url}")
            test_cap.release()
            return False
        test_cap.release()
        logger.info(f"✅ [SPAWN {camera_name}] URL OK")
        
        recorder.start_recording(camera_id=camera_name)
        logger.info(f"📹 [REC {camera_name}] START {comp}/{zone}")
        
    except Exception as e:
        logger.error(f"❌ [SPAWN {camera_name}] Setup FAILED: {e}")
        if recorder:
            try:
                recorder.stop_recording()
            except:
                pass
        return False
    
    loop = asyncio.get_running_loop()

    def worker_runner():
        try:
            logger.info(f"🔄 [WORKER {camera_name}] Starting...")
            worker.run(frame_q, log_q, status_q)
            logger.info(f"🛑 [WORKER {camera_name}] Finished")
        except Exception as e:
            logger.error(f"❌ [WORKER {camera_name}] ERROR: {e}")
            loop.create_task(ws_broadcast({"type": "error", "name": camera_name, "message": str(e)}))

    def pump():
        while True:
            try:
                msg = log_q.get_nowait()
                loop.create_task(ws_broadcast({"type": "log", "name": camera_name, "message": msg}))
            except qmod.Empty:
                pass
            try:
                st = status_q.get_nowait()
                loop.create_task(ws_broadcast({"type": "status", "name": camera_name, "status": st}))
            except qmod.Empty:
                pass
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def frame_pump():
        logger.info(f"🔄 [FRAME {camera_name}] Pump starting...")
        start_frame_pump(camera_name, frame_q, recorder)

    t_worker = threading.Thread(target=worker_runner, daemon=True)
    t_worker.start()
    
    t_pump = threading.Thread(target=pump, daemon=True)
    t_pump.start()
    
    t_frame = threading.Thread(target=frame_pump, daemon=True)
    t_frame.start()
    
    workers.append((worker, t_worker, frame_q, log_q, status_q, recorder))
    
    camera_meta_by_name[camera_name] = {
        "zone": zone,
        "url": url,
        "url2": cam.get("url2"),
        "comp": comp,
    }
    preview_mode[camera_name] = "main"
    
    logger.info(f"✅ [SPAWN {camera_name}] OK | Type: {zone} | Comp: {comp}")
    return True

def start_frame_pump(camera_name: str, frame_q: qmod.Queue, recorder: VideoRecorder):
    last_ok_ts = time.time()
    while True:
        try:
            data = frame_q.get(timeout=1.0)
            while True:
                try:
                    data = frame_q.get_nowait()
                except qmod.Empty:
                    break

            if not isinstance(data, tuple) or len(data) < 2:
                continue

            cam_name, frame = data[0], data[1]
            if not hasattr(frame, "shape"):
                continue

            frame2 = _even_pad(frame)

            with latest_lock:
                latest_frame[cam_name] = frame2
                latest_frame_ts[cam_name] = time.time()

            try:
                recorder.write_frame(frame2)
            except Exception as e:
                logger.warning(f"[REC {cam_name}] write_frame failed: {e}")

            if os.getenv("PREVIEW_ENHANCE", "1") == "1":
                try:
                    preview_frame = utils.enhance_preview(frame2)
                    with latest_lock:
                        latest_frame[cam_name] = preview_frame
                        latest_frame_ts[cam_name] = time.time()
                except Exception as e:
                    logger.debug(f"[PREVIEW {cam_name}] enhance failed: {e}")

            last_ok_ts = time.time()

        except qmod.Empty:
            if time.time() - last_ok_ts > STALE_SEC:
                with latest_lock:
                    latest_frame.pop(camera_name, None)
                    latest_frame_ts.pop(camera_name, None)
            continue
        except Exception as e:
            logger.error(f"[ERROR] Frame pump {camera_name} error: {e}")
            time.sleep(0.02)

def _is_stale(cam: str) -> bool:
    ts = latest_frame_ts.get(cam, 0.0)
    return (time.time() - ts) > STALE_SEC if ts else True

def _generate_health_status() -> dict:
    all_camera_names = list(camera_meta_by_name.keys())
    down_list = []
    ok_list = []
    
    logger.info(f"[HEALTH CHECK] Checking {len(all_camera_names)} configured cameras...")
    
    for cam_name in all_camera_names:
        if _is_stale(cam_name):
            down_list.append(cam_name)
        else:
            ok_list.append(cam_name)
            
    return {
        "total": len(all_camera_names),
        "ok_count": len(ok_list),
        "down_count": len(down_list),
        "down_list": down_list
    }


def camera_health_monitor_thread(loop: asyncio.AbstractEventLoop):
    global camera_last_known_status, camera_down_timestamp
    
    time.sleep(15) 
    logger.info(f"[Health Monitor] Thread started. Check interval: {HEALTH_CHECK_INTERVAL_SEC} sec.")
    
    try:
        if db:
            camera_last_known_status = db.get_all_last_camera_statuses()
            logger.info(f"[Health Monitor] Loaded {len(camera_last_known_status)} last known statuses from DB.")
    except Exception as e:
        logger.error(f"[Health Monitor] Failed to load last statuses: {e}")

    while True:
        try:
            current_camera_list = list(camera_meta_by_name.keys())
            now_ts = time.time()
            
            for cam_name in current_camera_list:
                new_status = "DOWN" if _is_stale(cam_name) else "OK"
                
                with health_monitor_lock:
                    old_status = camera_last_known_status.get(cam_name)
                
                if new_status == "OK":
                    first_down_time = camera_down_timestamp.pop(cam_name, None)
                    
                    if old_status == "DOWN":
                        logger.warning(f"[Health Monitor] STATUS CHANGE: {cam_name} | DOWN -> OK")
                        if db:
                            db.log_camera_event(cam_name, "OK", dt.datetime.now(tz))
                        
                        if first_down_time is not None:
                             logger.info(f"[Health Monitor] {cam_name} recovered before 30-min threshold.")
                    
                    with health_monitor_lock:
                        camera_last_known_status[cam_name] = "OK"

                else: # new_status == "DOWN"
                    if old_status != "DOWN":
                        logger.warning(f"[Health Monitor] STATUS CHANGE: {cam_name} | OK -> DOWN. Starting 30-min timer...")
                        camera_down_timestamp[cam_name] = now_ts
                        with health_monitor_lock:
                            camera_last_known_status[cam_name] = "DOWN"
                    
                    else: 
                        first_down_time = camera_down_timestamp.get(cam_name)
                        
                        if first_down_time is not None:
                            down_duration = now_ts - first_down_time
                            
                            if down_duration >= 1800: # (30 นาที)
                                logger.error(f"[Health Monitor] {cam_name} has been DOWN for 30 minutes. LOGGING EVENT.")
                                if db:
                                    db.log_camera_event(cam_name, "DOWN", dt.datetime.now(tz))
                                camera_down_timestamp.pop(cam_name, None)
            
            # --- ⭐️ [เพิ่มใหม่] Broadcast สถานะ ---
            with health_monitor_lock:
                current_statuses = dict(camera_last_known_status)
            loop.create_task(
                ws_broadcast({
                    "type": "health_status", 
                    "data": current_statuses
                })
            )
            # --- ⭐️ [สิ้นสุด] ---
                                
        except Exception as e:
            logger.error(f"[Health Monitor] Error in loop: {e}", exc_info=True)
        
        time.sleep(HEALTH_CHECK_INTERVAL_SEC)


# ===================== Lifespan =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, workers, camera_meta_by_name

    # --- ⭐️ 1. [แก้ไข] โหลด Settings และ Loop "ก่อน" เสมอ ⭐️ ---
    load_persistent_settings() 
    loop = asyncio.get_running_loop() 
    # --- ⭐️ ---

    utils.ensure_opencv_rtsp_env()
    logger.info(f"Application startup at real time: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        get_runtime_face_app()
    except Exception as e:
        logger.warning(f"[WARN] FaceAnalysis runtime init failed: {e}")

    try:
        local_db = Database()
        if not local_db or not local_db.conn:
            raise RuntimeError("Cannot connect to database")
        db = local_db
    except Exception as e:
        logger.error(f"[ERROR] Database init failed: {e}")
        db = None
        raise
        
    # --- ⭐️ 2. [แก้ไข] ลบ Thread ที่ซ้ำซ้อน ⭐️ ---
    # (เรียก Thread ที่รับ 'loop' เพียง 2 ตัว)
    threading.Thread(target=camera_health_monitor_thread, args=(loop,), daemon=True).start()
    
    cameras = db.get_cameras() if db else []
    camera_meta_by_name = {}
    preview_mode = {}
    for c in cameras:  
        nm = str(c.get("camera_name"))
        camera_meta_by_name[nm] = {
            "camera_name": nm,
            "zone": _norm_zone(c.get("zone")) or "face",
            "url": c.get("url"),
            "url2": c.get("url2"),
            "comp": c.get("comp"),
            "enter": c.get("enter"),
        }
        preview_mode[nm] = "main"

    ##app.mount("/recordings", StaticFiles(directory=RECORD_ROOT), name="recordings")

    # (ลบ loop = asyncio.get_running_loop() บรรทัดที่ 2 ออก)
    for cam in cameras:
        camera_name = str(cam["camera_name"])
        zone = _norm_zone(cam.get("zone")) or "face"
        url = cam["url"]
        comp = cam.get("comp")

        frame_q = qmod.Queue(maxsize=3000)
        log_q = qmod.Queue()
        status_q = qmod.Queue()

        if zone == "car":
            worker = CarCameraWorker(
                id=cam.get("id"),
                name=camera_name,
                url=url,
                zone=zone,
                department=comp,
                user_access=[comp] if comp else [],
            )
        else:
            worker = FaceCameraWorker(
                id=cam.get("id"),
                name=camera_name,
                camera_url=url,
                zone=zone,
                department=comp,
                user_access=[comp] if comp else [],
            )

        # --- ⭐️ 3. [แก้ไข] อ่านค่า Segment จาก runtime_settings ⭐️ ---
        with runtime_settings_lock:
            current_segment_minutes = runtime_settings.get("SEGMENT_MINUTES", 15)
            
        recorder = VideoRecorder(
            output_dir=RECORD_ROOT,
            zone=zone,
            segment_minutes=current_segment_minutes, # ⭐️ (ใช้ค่านี)
            department=comp or "Unknown",
            camera_id=camera_name,
        )
        # --- ⭐️ [สิ้นสุด] ⭐️ ---
        
        try:
            recorder.start_recording(camera_id=camera_name)
        except Exception as e:
            logger.error(f"[ERROR] Failed to start recording for camera {camera_name}: {e}")
            continue

        def worker_runner(_worker=worker, _fq=frame_q, _lq=log_q, _sq=status_q, _name=camera_name):
            try:
                _worker.run(_fq, _lq, _sq)
            except Exception as e:
                loop.create_task(ws_broadcast({"type": "error", "name": _name, "message": str(e)}))

        def pump(_lq=log_q, _sq=status_q, _name=camera_name):
            while True:
                try:
                    try:
                        msg = _lq.get_nowait()
                        loop.create_task(ws_broadcast({"type": "log", "name": _name, "message": msg}))
                    except qmod.Empty:
                        pass
                    try:
                        _cc, st = _sq.get_nowait()
                        loop.create_task(ws_broadcast({"type": "status", "name": _name, "status": st}))
                    except qmod.Empty:
                        pass
                    time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)

        t = threading.Thread(target=worker_runner, daemon=True)
        t.start()
        threading.Thread(target=pump, daemon=True).start()
        threading.Thread(target=start_frame_pump, args=(camera_name, frame_q, recorder), daemon=True).start()

        workers.append((worker, t, frame_q, log_q, status_q, recorder))

    stop_once = threading.Event()
    
    def graceful_exit(*_args):
        if stop_once.is_set():
            return
        stop_once.set()
        for item in workers:
            rec = item[5] if len(item) >= 6 else None
            try:
                if rec and rec.is_recording():
                    rec.stop_recording()
            except Exception:
                pass
        try:
            for name in list(sub_preview_threads.keys()):
                stop_sub_preview(name)
        except Exception:
            pass
        time.sleep(0.3)
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, lambda s, f: graceful_exit())
        signal.signal(signal.SIGTERM, lambda s, f: graceful_exit())
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, lambda s, f: graceful_exit())
    except Exception:
        pass

    yield

    for item in workers:
        rec = item[5] if len(item) >= 6 else None
        try:
            if rec and rec.is_recording():
                rec.stop_recording()
        except Exception:
            pass
    try:
        for name in list(sub_preview_threads.keys()):
            stop_sub_preview(name)
    except Exception:
        pass
    await asyncio.sleep(0.2)

app.router.lifespan_context = lifespan
# ===================== WebSocket Broadcast =====================
async def ws_broadcast(event: Dict[str, Any]):
    bad = []
    for ws in list(event_subscribers):
        try:
            await ws.send_text(json.dumps(event, ensure_ascii=False))
        except Exception:
            bad.append(ws)
    for ws in bad:
        try:
            event_subscribers.remove(ws)
        except Exception:
            pass

# ===================== Sub Preview =====================
def start_sub_preview(camera_name: str, sub_url: str):
    if camera_name in sub_preview_threads:
        return

    stop_event = threading.Event()

    def _runner():
        utils.ensure_opencv_rtsp_env()
        cap = None
        last_open = 0.0
        while not stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    if time.time() - last_open < 1.0:
                        time.sleep(0.2)
                        continue
                    cap = _open_rtsp(sub_url)
                    last_open = time.time()
                    if not cap or not cap.isOpened():
                        logger.warning(f"[SUB_PREVIEW {camera_name}] Failed to open RTSP: {sub_url}")
                        time.sleep(0.5)
                        continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    continue

                frame2 = _even_pad(frame)
                with latest_lock:
                    latest_frame_sub[camera_name] = frame2

                time.sleep(0.005)

            except Exception as e:
                logger.error(f"[ERROR] Sub preview {camera_name}: {e}")
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(0.5)

        try:
            if cap:
                cap.release()
        except Exception:
            pass

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    sub_preview_threads[camera_name] = (t, stop_event)
    logger.info(f"[SUB_PREVIEW {camera_name}] Started sub preview thread for {sub_url}")

def stop_sub_preview(camera_name: str):
    item = sub_preview_threads.pop(camera_name, None)
    if not item:
        return
    t, stop_event = item
    try:
        stop_event.set()
        t.join(timeout=2.0)
    except Exception:
        pass
    with latest_lock:
        latest_frame_sub.pop(camera_name, None)
    logger.info(f"[SUB_PREVIEW {camera_name}] Stopped sub preview thread")

# ===================== Access Control Helper =====================
def _has_access_to_camera(camera_name: str, claims: TokenClaims) -> bool:
    meta = camera_meta_by_name.get(camera_name)
    if not meta:
        return False
    cam_comp = meta.get("comp")
    if claims.is_admin:
        return True
    return (cam_comp and cam_comp in (claims.access or []))

# --- ⭐️ [เพิ่มใหม่ 1/2] ฟังก์ชัน Save ⭐️ ---
def save_persistent_settings():
    """(Thread-safe) บันทึก runtime settings ลงไฟล์ .json"""
    with runtime_settings_lock:
        try:
            # (ต้องแน่ใจว่า SETTINGS_FILE ถูกประกาศไว้ด้านบน)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(runtime_settings, f, indent=2)
            logger.info(f"Saved persistent settings to {SETTINGS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save persistent settings: {e}")

# --- ⭐️ [เพิ่มใหม่ 2/2] ฟังก์ชัน Load (ที่ขาดไป) ⭐️ ---
def load_persistent_settings():
    """(Thread-safe) โหลด settings จาก .json ตอนเปิด Server"""
    global runtime_settings
    
    default_segment = int(os.getenv("SEGMENT_MINUTES", "15"))
    settings = {"SEGMENT_MINUTES": default_segment}

    # (ต้องแน่ใจว่า SETTINGS_FILE ถูกประกาศไว้ด้านบน)
    if not SETTINGS_FILE.exists():
        logger.info(f"No {SETTINGS_FILE} found, using default (Segment: {default_segment} min).")
    else:
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "SEGMENT_MINUTES" in data:
                settings["SEGMENT_MINUTES"] = int(data["SEGMENT_MINUTES"])
            
            logger.info(f"Loaded persistent settings from {SETTINGS_FILE}: {settings}")
            
        except Exception as e:
            logger.error(f"Failed to load {SETTINGS_FILE}, using defaults. Error: {e}")

    with runtime_settings_lock:
        runtime_settings = settings

# ===================== Routes =====================
@app.get("/")
def root():
    return {"status": "ok", "message": "CCTV backend (stable) is online"}

# ---------- Auth ----------
@app.post("/auth/login")
def auth_login(
    payload: dict = Body(...),
    force: bool = Query(False),
    remember: bool = Query(False),
    request: Request = None    
):
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="username/password required")

    res = db.check_user(username, password)
    if isinstance(res, tuple) and len(res) == 4:
        is_valid, department, access, is_admin = res
    elif isinstance(res, tuple) and len(res) == 3:
        is_valid, department, access = res
        is_admin = (department == "IT")
    else:
        raise HTTPException(status_code=500, detail="check_user returned unexpected format")

    if not is_valid:
        raise HTTPException(status_code=401, detail="user or password wrong")

    if session_has_active(username) and not force:
        raise HTTPException(status_code=409, detail="User already logged in on another device")

    if force:
        session_revoke_user(username)

    access_list = [x.strip() for x in access.split(",") if x.strip()] if isinstance(access, str) else list(access or [])

    ttl = (60 * 24 *7) if remember else JWT_EXPIRE_MINUTES
    token = create_access_token(username, department, access_list, bool(is_admin), ttl_minutes=ttl)

    try:
        pl = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        ua = request.headers.get("user-agent", "") if request else ""
        ip = request.client.host if (request and request.client) else None
        session_register(username, pl.get("sid"), pl.get("jti"), pl.get("exp"), ua, ip)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"session register failed: {e}")

    return {
        "access_token": token,
        "token_type": "bearer",
        "department": department,
        "access": access_list,
        "is_admin": bool(is_admin),
        "expires_in": (ttl * 60),
    }

@app.post("/auth/logout")
def auth_logout(claims: TokenClaims = Depends(require_user)):
    try:
        session_revoke_one(claims.sub, claims.sid or "")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"logout failed: {e}")

@app.post("/auth/logout-all")
def auth_logout_all(claims: TokenClaims = Depends(require_user)):
    try:
        session_revoke_user(claims.sub)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"logout all failed: {e}")

# ---------- Cameras ----------
def _validate_rtsp_optional(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    v2 = str(v).replace("\\", "/").strip()
    return v2 if (v2.startswith("rtsp://") or v2.startswith("rtsps://")) else None

@app.get("/cameras")
def list_cameras(
    dept: Optional[str] = None,
    access: Optional[str] = None,
    claims: Optional[TokenClaims] = Depends(get_claims),
): 
    if claims:
        access_list = claims.access
        is_admin = claims.is_admin
    else:
        access_list = [x.strip() for x in (access or "").split(",") if x.strip()] or None
        is_admin = False

    cams = db.get_cameras(allowed_departments=access_list) if not is_admin else db.get_cameras()
    return cams

@app.post("/cameras", status_code=201)
async def add_camera(cam: CameraIn, _: TokenClaims = Depends(require_admin)):
    try:
        new_id = db.add_camera(
            camera_name=cam.camera_name,
            url=cam.url,
            url2=cam.url2,
            zone=_norm_zone(cam.zone) or "face",
            comp=cam.comp,
        )
    except Exception as e:
        raise HTTPException(500, f"db error: {e}")

    if new_id is None:
        raise HTTPException(400, "cannot add camera (name duplicate)")

    started = spawn_camera({
        "id": new_id,
        "camera_name": cam.camera_name,
        "url": cam.url,
        "url2": cam.url2,
        "zone": _norm_zone(cam.zone) or "face",
        "comp": cam.comp,
    })

    camera_meta_by_name[cam.camera_name] = {
        "camera_name": cam.camera_name,
        "zone": _norm_zone(cam.zone) or "face",
        "url": cam.url,
        "url2": cam.url2,
        "comp": cam.comp,
    }

    sub_url = cam.url2 or _infer_sub_url_from_main(cam.url)
    if sub_url:
        preview_mode[cam.camera_name] = "sub"
        start_sub_preview(cam.camera_name, sub_url)
        logger.info(f"[CAMERA {cam.camera_name}] Added (start SUB): {sub_url}")
        return {"ok": True, "id": new_id, "started": bool(started), "mode": "sub", "has_sub": True}
    else:
        preview_mode[cam.camera_name] = "main"
        logger.info(f"[CAMERA {cam.camera_name}] Added (start MAIN): {cam.url}")
        return {"ok": True, "id": new_id, "started": bool(started), "mode": "main", "has_sub": False}

@app.put("/cameras/{camera_name}")
async def update_camera(camera_name: str, patch: dict = Body(...), _: TokenClaims = Depends(require_admin)):
    allowed = {"url", "zone", "comp", "url2", "preview_mode"}
    patch = {k: v for k, v in (patch or {}).items() if k in allowed}

    if "zone" in patch:
        patch["zone"] = _norm_zone(patch["zone"]) or "face"
    if "url2" in patch:
        patch["url2"] = _validate_rtsp_optional(patch["url2"])

    if not patch:
        return {"ok": True}

    if not db.update_camera(camera_name, patch):
        raise HTTPException(404, "camera not found")

    meta = camera_meta_by_name.get(camera_name, {})
    meta.update({k: patch[k] for k in patch})
    camera_meta_by_name[camera_name] = meta

    if "preview_mode" in patch:
        requested_mode = patch["preview_mode"]
        url2 = meta.get("url2")

        stop_sub_preview(camera_name)

        if requested_mode == "sub" and url2:
            start_sub_preview(camera_name, url2)
            preview_mode[camera_name] = "sub"   
            logger.info(f"[CAMERA {camera_name}] Switched to sub stream via PUT: {url2}")
        else:
            preview_mode[camera_name] = "main"
            if requested_mode == "sub" and not url2:
                 logger.warning(f"[CAMERA {camera_name}] PUT requested SUB mode but no url2. Falling back to MAIN.")
            else:
                 logger.info(f"[CAMERA {camera_name}] Switched to main stream via PUT: {meta.get('url')}")
    return {"ok": True}

@app.delete("/cameras/{camera_name}")
def delete_camera(camera_name: str, _: TokenClaims = Depends(require_admin)):
    if not db.delete_camera(camera_name):
        raise HTTPException(status_code=404, detail="camera not found")

    stop_sub_preview(camera_name)

    idx_to_remove = None
    for i, (w, t, _fq, _lq, _sq, rec) in enumerate(workers):
        if getattr(w, "name", None) == camera_name:
            try:
                w.stop()
                if rec and rec.is_recording():
                    rec.stop_recording()
                t.join(timeout=2.0)
            except Exception:
                pass
            idx_to_remove = i
            break
    if idx_to_remove is not None:
        workers.pop(idx_to_remove)

    camera_meta_by_name.pop(camera_name, None)
    preview_mode.pop(camera_name, None)
    with latest_lock:
        latest_frame.pop(camera_name, None)
        latest_frame_sub.pop(camera_name, None)
        latest_frame_ts.pop(camera_name, None)
    return {"ok": True}

@app.post("/employees/{emp_id}/update")
async def update_employee_add_image(
    emp_id: str,
    name: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    _: TokenClaims = Depends(require_admin)
):
    if db is None:
        raise HTTPException(500, "database not ready")
    if not files:
        raise HTTPException(400, "no new files uploaded")
    uf = files[0] 
    if not db.employee_exists(emp_id):
        raise HTTPException(404, f"Employee with emp_id {emp_id} not found.")
    final_name = name if name else None
    final_dept = department if department else None 
    enroll_app = get_enroll_face_app()
    
    try:
        content = await uf.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(422, f"{uf.filename}: Cannot decode image")
        faces = enroll_app.get(img)
        if not faces or getattr(faces[0], "embedding", None) is None:
            raise HTTPException(422, f"{uf.filename}: No face detected")
        
        emb = faces[0].embedding.astype(np.float32)
        view_hint = _infer_hint(uf.filename)
        
        ok = db.add_employee(
            emp_id=emp_id,
            name=final_name,
            department=final_dept,
            image_data=content,
            embedding=emb.tobytes(),
            view_hint=view_hint,
            aligned_image_data=None 
        )
        
        if not ok:
            raise HTTPException(409, "All 5 image slots are full. Please delete an old image first.")

    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"[UPDATE_EMP] Error processing {uf.filename}: {e}", exc_info=True)
        raise HTTPException(500, f"Server error: {str(e)}")
    
    refresh_all_face_workers()
    return {"ok": True, "emp_id": emp_id, "new_image_added": True}

@app.delete("/employees/{emp_id}/slot/{slot_num}")
def delete_employee_slot(
    emp_id: str,
    slot_num: int,
    _: TokenClaims = Depends(require_admin)
):
    if db is None:
        raise HTTPException(500, "database not ready")
    
    if slot_num not in (1, 2, 3, 4, 5):
        raise HTTPException(422, "Slot number must be 1, 2, 3, 4, or 5")

    try:
        ok = db.clear_employee_slot(emp_id, slot_num)
        if not ok:
            raise HTTPException(404, "Employee not found or slot already empty")
        
        refresh_all_face_workers() 
        
        return {"ok": True, "emp_id": emp_id, "slot_cleared": slot_num}
    except Exception as e:
        logger.error(f"[DELETE_SLOT] Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    
@app.get("/employees")
def list_employees(
    claims: TokenClaims = Depends(require_user_flexible)
):
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        employees = db.list_employees() 
        return employees
    except AttributeError:
         logger.error("[ERROR] Server code is missing 'db.list_employees()' method in Database class.")
         raise HTTPException(status_code=500, detail="Server is missing 'db.list_employees()' method.")
    except Exception as e:
        logger.error(f"[ERROR] /employees list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing employees: {e}")
    
@app.get("/employees/{emp_id}")
def get_employee_details(
    emp_id: str,
    _: TokenClaims = Depends(require_admin)
):
    if db is None:
        raise HTTPException(500, "database not ready")
    
    details = db.get_employee_details(emp_id) 
    
    if not details:
        raise HTTPException(404, "Employee not found")
    return details

class EmployeeInfoUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None

@app.put("/employees/{emp_id}/info")
def update_employee_info(
    emp_id: str,
    payload: EmployeeInfoUpdate,
    _: TokenClaims = Depends(require_admin)
):
    if db is None:
        raise HTTPException(500, "database not ready")

    ok = db.update_employee_info(
        emp_id, 
        name=payload.name, 
        department=payload.department
    )
    
    if not ok:
        raise HTTPException(404, "Employee not found or update failed")
    
    refresh_all_face_workers()
    
    return {"ok": True, "emp_id": emp_id}
@app.post("/cameras/{camera_name}/preview-mode")
async def set_preview_mode(
    camera_name: str,
    payload: dict = Body(...),
    claims: TokenClaims = Depends(require_user_flexible)
):
    if not _has_access_to_camera(camera_name, claims):
        raise HTTPException(status_code=403, detail=f"No access to camera {camera_name}")

    mode = payload.get("mode")
    if mode not in ("main", "sub"):
        raise HTTPException(422, "mode must be 'main' or 'sub'")

    if camera_name not in camera_meta_by_name:
        raise HTTPException(404, "camera not found")

    meta = camera_meta_by_name[camera_name]
    url2 = meta.get("url2")
    stop_sub_preview(camera_name)

    if mode == "sub" and url2:
        start_sub_preview(camera_name, url2)
        preview_mode[camera_name] = "sub"
    else:
        preview_mode[camera_name] = "main"

    logger.info(f"[PREVIEW] User {claims.sub} set {camera_name} to {preview_mode[camera_name]}")
    return {"ok": True, "mode": preview_mode[camera_name]}

@app.get("/cameras/{camera_name}/preview-mode")
async def get_preview_mode(camera_name: str, _: TokenClaims = Depends(require_admin)):
    if camera_name not in preview_mode:
        return {"mode": "main"}
    return {"mode": preview_mode[camera_name]}

@app.websocket("/ws/ui-updates")
async def websocket_ui_updates(
    websocket: WebSocket,
    # (เปลี่ยนจาก require_user_flexible เป็น require_user_ws)
    claims: TokenClaims = Depends(require_user_ws) 
):
    await websocket.accept()
    
    event_subscribers.append(websocket)
    logger.info(f"[UI WS] Client {claims.sub} connected for UI updates.")
    
    try:
        with health_monitor_lock:
            current_statuses = dict(camera_last_known_status)
        await websocket.send_text(json.dumps({
            "type": "health_status",
            "data": current_statuses
        }, ensure_ascii=False))

        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info(f"[UI WS] Client {claims.sub} disconnected.")
    except Exception as e:
        logger.error(f"[UI WS] Error: {e}")
    finally:
        if websocket in event_subscribers:
            event_subscribers.remove(websocket)

@app.websocket("/ws/mjpg/{camera_name}")
async def websocket_mjpg(websocket: WebSocket, camera_name: str, token: str = Query(...)):
    await websocket.accept()

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        claims = TokenClaims(**payload)
        if not session_is_active(claims.sub, claims.sid or "", claims.jti or ""):
            await websocket.send_json({"error": "session_revoked"})
            await websocket.close(code=1008)
            return
    except jwt.ExpiredSignatureError:
        await websocket.send_json({"error": "Token expired"})
        await websocket.close(code=1008)
        return
    except jwt.InvalidTokenError:
        await websocket.send_json({"error": "Invalid token"})
        await websocket.close(code=1008)
        return

    if not _has_access_to_camera(camera_name, claims):
        await websocket.send_json({"error": f"No access to camera {camera_name}"})
        await websocket.close(code=1008)
        return

    if camera_name not in camera_meta_by_name:
        await websocket.send_json({"error": f"Camera {camera_name} not found"})
        await websocket.close(code=1008)
        return

    logger.info(f"[MJPG {camera_name}] WebSocket connected for user: {claims.sub}")

    boundary = b"--myboundary\r\n"
    content_type = b"Content-Type: image/jpeg\r\n"
    content_length_prefix = b"Content-Length: "
    newline = b"\r\n"

    target_fps = max(1, int(RECORD_FPS_HINT))
    frame_interval = 1.0 / float(target_fps)
    last_send = time.time()
    log_count = 0 

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            if not session_is_active(claims.sub, claims.sid or "", claims.jti or ""):
                await websocket.send_json({"error": "session_revoked"})
                break

            now = time.time()
            elapsed = now - last_send
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

            with latest_lock:
                current_mode = preview_mode.get(camera_name, "main")
                meta = camera_meta_by_name.get(camera_name, {})
                url2 = meta.get("url2")
                
                if current_mode == "sub" and camera_name in latest_frame_sub and url2:
                    frame = latest_frame_sub[camera_name]
                    source = "SUB"
                else:
                    frame = latest_frame.get(camera_name)
                    source = "MAIN"
                    if frame is None and camera_name in latest_frame_sub:
                        frame = latest_frame_sub[camera_name]
                        source = "SUB(fallback)"
                
                log_count += 1
                if log_count >= 250:
                    logger.info(f"[MJPG {camera_name}] Mode={current_mode} | Source={source} | url2={bool(url2)}")
                    log_count = 0

            if frame is None or _is_stale(camera_name):
                jpeg_data = encode_frame_to_jpeg(BLACK_480P)
            else:
                jpeg_data = encode_frame_to_jpeg(frame)

            if not jpeg_data:
                await asyncio.sleep(0.01)
                continue

            try:
                await websocket.send_bytes(
                    boundary +
                    content_type +
                    content_length_prefix + str(len(jpeg_data)).encode() + newline +
                    newline +
                    jpeg_data +
                    newline
                )
                last_send = time.time()
            except Exception as e:
                logger.error(f"[MJPG {camera_name}] WebSocket send error: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"[MJPG {camera_name}] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[MJPG {camera_name}] WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[MJPG {camera_name}] WebSocket closed")

_HINT_KEYWORDS = {
    "center": ("center", "front", "straight", "middle", "mid"),
    "left":   ("left", "l_", "_l", "-l", "(l)", "yawleft"),
    "right":  ("right", "r_", "_r", "-r", "(r)", "yawright"),
}

def _guess_view_hint_from_name(filename: str) -> str:
    if not filename:
        return "center"
    fn = filename.lower()
    for hint, keys in _HINT_KEYWORDS.items():
        if any(k in fn for k in keys):
            return hint
    return "center"

ENR_MAX_IMG_SIDE      = int(os.getenv("ENR_MAX_IMG_SIDE", "3000"))
ENR_MIN_FACE_PX       = int(os.getenv("ENR_MIN_FACE_PX", "140"))
ENR_MIN_BLUR_VAR      = float(os.getenv("ENR_MIN_BLUR_VAR", "120"))
ENR_MIN_DET_SCORE     = float(os.getenv("ENR_MIN_DET_SCORE", "0.60"))
ENR_DUPE_SIM_THRESH   = float(os.getenv("ENR_DUPE_SIM_THRESH", "0.88"))
ENR_STORE_ALIGNED     = os.getenv("ENR_STORE_ALIGNED", "true").lower() in ("1","true","yes")

def _resize_if_too_big(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def _lap_var(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def _best_face(enroll_app, img: np.ndarray):
    faces = enroll_app.get(img) or []
    if not faces:
        return None
    good = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        w, h = max(0, x2-x1), max(0, y2-y1)
        if min(w, h) < ENR_MIN_FACE_PX:
            continue
        det_score = float(getattr(f, "det_score", 0.0) or 0.0)
        if det_score < ENR_MIN_DET_SCORE:
            continue
        crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            continue
        if _lap_var(crop) < ENR_MIN_BLUR_VAR:
            continue
        good.append(f)
    if not good:
        return None
    good.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return good[0]

def _aligned_112(img, face):
    try:
        al = _norm_crop_112(img, landmark=getattr(face, "kps", None), image_size=112)
        return al if al is not None and al.shape[:2] == (112, 112) else None
    except Exception:
        return None

def _l2norm_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return (x / n).astype(np.float32)

def _embedding_from(face_app, face, aligned: Optional[np.ndarray]) -> Optional[np.ndarray]:
    emb = getattr(face, "embedding", None)
    if emb is not None and emb.size == 512:
        return _l2norm_np(emb.astype(np.float32).reshape(-1))
    if aligned is None:
        return None
    rec_model = face_app.models.get("recognition") if hasattr(face_app, "models") else None
    if rec_model is None:
        return None
    try:
        feat = rec_model.get_feat(aligned).reshape(-1).astype(np.float32)
        return _l2norm_np(feat)
    except Exception:
        return None

def _load_emp_existing_embeddings(emp_id: str) -> List[np.ndarray]:
    try:
        if hasattr(db, "load_employee_embeddings"):
            rows = db.load_employee_embeddings(emp_id)
        elif hasattr(db, "get_employee_embeddings"):
            rows = db.get_employee_embeddings(emp_id)
        else:
            rows = []
        out = []
        for r in rows:
            e = np.frombuffer(r, dtype=np.float32) if isinstance(r, (bytes, bytearray)) else np.array(r, dtype=np.float32)
            if e.size == 512:
                out.append(_l2norm_np(e))
        return out
    except Exception:
        return []

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)

def _jpeg_bytes(img_bgr: np.ndarray, q: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else b""

def _infer_hint(filename: str) -> Optional[str]:
    fn = (filename or "").lower()
    if "left" in fn: return "left"
    if "right" in fn: return "right"
    if any(k in fn for k in ("center","front","straight","mid","middle")): return "center"
    return None

@app.post("/employees/enroll")
async def enroll_employee(
    emp_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    files: List[UploadFile] = File(...),
    _: TokenClaims = Depends(require_admin)
):
    if db is None:
        raise HTTPException(500, "database not ready")
    if not files:
        raise HTTPException(400, "no files uploaded")
    
    enroll_app = get_enroll_face_app() 

    def infer_hint(filename: str) -> str | None:
        fn = (filename or "").lower()
        if "left" in fn:   return "left"
        if "right" in fn:  return "right"
        if any(k in fn for k in ("center","front","straight")): return "center"
        return None

    saved_any = False
    errors = []
    for uf in files:
        try:
            content = await uf.read()
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"[ENROLL] skip (decode fail): {uf.filename}")
                errors.append(f"{uf.filename}: Cannot decode image")
                continue

            faces = enroll_app.get(img)
            if not faces or getattr(faces[0], "embedding", None) is None:
                logger.warning(f"[ENROLL] no face detected in: {uf.filename}")
                errors.append(f"{uf.filename}: No face detected")
                continue
            
            emb = faces[0].embedding.astype(np.float32)

            view_hint = infer_hint(uf.filename)
            ok = db.add_employee(
                emp_id=emp_id,
                name=name,
                department=department,
                image_data=content,
                embedding=emb.tobytes(),
                view_hint=view_hint,
            )
            if ok:
                logger.info(f"[ENROLL] saved {uf.filename} -> {view_hint or 'auto-next'}")
                saved_any = True
        except Exception as e:
            logger.error(f"[ENROLL] Error processing {uf.filename}: {e}", exc_info=True)
            errors.append(f"{uf.filename}: Server error - {str(e)}")

    if not saved_any:
        raise HTTPException(
            status_code=422, 
            detail={"message": "No valid new faces could be processed.", "errors": errors}
        )
    
    refresh_all_face_workers()

    return {"ok": True, "emp_id": emp_id, "name": name, "department": department, "errors": errors}

@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: str, _: TokenClaims = Depends(require_admin)):
    if not db.delete_employee(emp_id):
        raise HTTPException(status_code=404, detail="employee not found")
    return {"ok": True}

@app.post("/users/register")
async def register_user(payload: dict = Body(...), _: TokenClaims = Depends(require_admin)):
    username = payload.get("user_name") or payload.get("username")
    password = payload.get("pass_user") or payload.get("password")
    department = payload.get("department", "")
    access = payload.get("access", [])
    is_admin = bool(payload.get("is_admin", False))

    if not username or not password:
        raise HTTPException(422, "missing field user_name/pass_user")

    ok = db.register_user(
        username=username,
        password=password,
        department=department,
        access=access,
        is_admin=is_admin,
    )
    if not ok:
        raise HTTPException(400, "cannot create user (maybe already exists)")
    return {"ok": True}

class AdminTempResetIn(BaseModel):
    temp_password: Optional[str] = None
    expire_minutes: int = 30

@app.post("/admin/users/{username}/reset-password-temp")
async def admin_set_temp_password(
    username: str,
    payload: AdminTempResetIn,
    claims: TokenClaims = Depends(require_admin),
):
    if payload.temp_password and isinstance(payload.temp_password, str) and payload.temp_password.strip():
        temp_pw = payload.temp_password.strip()
    else:
        gen_fn = getattr(utils, "generate_human_temp_password", None)
        if callable(gen_fn):
            temp_pw = gen_fn()
        else:
            alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
            temp_pw = "".join(secrets.choice(alphabet) for _ in range(10))

    expire_minutes = max(5, min(int(payload.expire_minutes or 30), 1440))

    ok = db.set_temp_password(username, temp_pw, expire_minutes)
    if not ok:
        raise HTTPException(404, "user not found or set temp password failed")

    return {
        "ok": True,
        "username": username,
        "temp_password": temp_pw,
        "expire_minutes": expire_minutes
    }

class LoginTempIn(BaseModel):
    username: str
    temp_password: str

@app.post("/auth/login-temp")
def auth_login_temp(payload: LoginTempIn, request: Request = None):
    username = (payload.username or "").strip()
    temp_password = (payload.temp_password or "").strip()
    if not username or not temp_password:
        raise HTTPException(400, "username/temp_password required")

    valid, reason = db.verify_temp_password(username, temp_password)
    if not valid:
        raise HTTPException(401, f"temp login failed: {reason}")

    token = create_temp_login_token(username)

    try:
        pl = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sid = pl.get("sid")
        jti = pl.get("jti")
        exp = pl.get("exp")
        if sid and jti and exp:
            ua = request.headers.get("user-agent", "") if request else ""
            ip = request.client.host if (request and request.client) else None
            session_register(username, sid, jti, exp, ua, ip)
            logger.info(f"[TEMP LOGIN] Session registered: {username} (sid={sid})")
    except Exception as e:
        logger.error(f"[TEMP LOGIN] Session register failed: {e}")

    return {
        "access_token": token,
        "token_type": "bearer",
        "must_change_password": True,
        "expires_in": 30 * 60
    }
class ChangePasswordIn(BaseModel):
    new_password: str

@app.post("/auth/change-password")
def auth_change_password(payload: ChangePasswordIn, claims: TokenClaims = Depends(require_user)):
    new_pw = (payload.new_password or "").strip()
    if not new_pw or len(new_pw) < 6:
        logger.warning(f"[AUTH CHANGE PW] User {claims.sub} provided password too short.")
        raise HTTPException(422, "new_password is too short (min 6 chars)")

    username = claims.sub

    if not claims.must_change:
        logger.warning(f"[AUTH CHANGE PW] User {username} attempted change password without must_change token.")
        raise HTTPException(403, "Forbidden: Action requires temporary password state.")

    try:
        logger.info(f"[AUTH CHANGE PW] Attempting db.consume_temp_password for user: {username}")
        consume_ok = db.consume_temp_password(username, new_pw)
        if not consume_ok:
            logger.error(f"[AUTH CHANGE PW] db.consume_temp_password FAILED for user: {username}. Function returned False.")
            raise HTTPException(500, "Failed to set new password in database.")
        logger.info(f"[AUTH CHANGE PW] db.consume_temp_password SUCCEEDED for user: {username}")
    except Exception as db_err:
        logger.error(f"[AUTH CHANGE PW] EXCEPTION during db.consume_temp_password for user {username}: {db_err}", exc_info=True)
        raise HTTPException(500, f"Database error during password update: {db_err}")

    try:
        prof = db.get_user_profile(username) or {}
        dept = prof.get("department", "")
        access_raw = prof.get("access")
        if isinstance(access_raw, list):
            access = access_raw
        elif isinstance(access_raw, str):
             access = [x.strip() for x in access_raw.split(',') if x.strip()]
        else:
            access = [] 

        is_admin = bool(prof.get("is_admin", False))
        new_token = create_access_token(username, dept, access, is_admin)
        logger.info(f"[AUTH CHANGE PW] New permanent token created for user: {username}")
    except Exception as token_err:
        logger.error(f"[AUTH CHANGE PW] EXCEPTION during token creation for user {username}: {token_err}", exc_info=True)
        raise HTTPException(500, "Password updated, but failed to create new session token. Please log in again.")

    try:
        pl = jwt.decode(new_token, JWT_SECRET, algorithms=[JWT_ALG])
        new_sid = pl.get("sid")
        new_jti = pl.get("jti")
        new_exp = pl.get("exp")

        if not new_sid or not new_jti or not new_exp:
             logger.error(f"[AUTH CHANGE PW] New token for {username} is missing sid/jti/exp.")
             raise HTTPException(500,"Internal error: New token structure invalid.")

        logger.info(f"[AUTH CHANGE PW] Attempting session_register for new token (sid: {new_sid}) for user: {username}")
        session_register(username, new_sid, new_jti, new_exp)
        logger.info(f"[AUTH CHANGE PW] session_register SUCCEEDED for new token (sid: {new_sid}) for user: {username}")

    except jwt.PyJWTError as jwt_err:
        logger.error(f"[AUTH CHANGE PW] EXCEPTION decoding new token for user {username}: {jwt_err}", exc_info=True)
        raise HTTPException(500, "Password updated, but failed to process new session token. Please log in again.")
    except Exception as reg_err:
        logger.error(f"[AUTH CHANGE PW] EXCEPTION during session_register for user {username} (sid: {new_sid}): {reg_err}", exc_info=True)
        raise HTTPException(500, "Password updated and token created, but failed to register new session. Please log in again.")

    logger.info(f"[AUTH CHANGE PW] Password change and session registration successful for user: {username}")
    return {
        "ok": True,
        "message": "Password changed successfully. Use this new token.",
        "access_token": new_token,
        "token_type": "bearer",
        "department": dept,
        "access": access,
        "is_admin": is_admin
    }

def thai_to_arabic(date_str: str) -> str:
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    arabic_digits = "0123456789"
    trans_table = str.maketrans(thai_digits, arabic_digits)
    return date_str.translate(trans_table)

FILENAME_TIME_RE = re.compile(r'_(\d{2})[-_]?(\d{2})[-_]?(\d{2})(?:\.\w+)?$')

def _get_time_range_from_filename(
    date_str: str, 
    filename: str, 
    segment_minutes: int
) -> tuple[dt.datetime, dt.datetime]:
    bangkok_tz = pytz.timezone('Asia/Bangkok')
    
    match = FILENAME_TIME_RE.search(filename)
    
    start_dt_obj = None
    if match:
        try:
            hh, mm, ss = match.groups()
            start_str = f"{date_str} {hh}:{mm}:{ss}"
            start_dt_obj = dt.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S") 
        except ValueError as e:
            logger.warning(f"Could not parse time from filename '{filename}' (match: {match.groups()}). Error: {e}")
            start_dt_obj = None

    if start_dt_obj is None:
        logger.warning(f"Fallback: Could not parse time from {filename}, using 00:00:00 as start.")
        start_dt_obj = dt.datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S")

    start_dt = bangkok_tz.localize(start_dt_obj)
    
    if not match:
        end_dt = start_dt + dt.timedelta(days=1, seconds=-1)
    else:
        end_dt = start_dt + dt.timedelta(minutes=segment_minutes)
    
    return start_dt, end_dt

@app.get("/reports")
async def report_all(
    start: str = Query(..., description="YYYY-MM-DD[ HH:MM:SS]"),
    end: str = Query(..., description="YYYY-MM-DD[ HH:MM:SS]"),
    department: Optional[str] = Query(None, description="filter by camera's comp"),
    type: Optional[str] = Query(None, description="face | car"),
    q: Optional[str] = Query(None, description="search: plate/name/department/province/camera"),
    limit: int = Query(500, ge=1, le=5000),
    claims: TokenClaims = Depends(require_user),
):
    try:
        def thai_to_arabic_local(s: str) -> str:
            return s.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))

        start = thai_to_arabic_local(start.strip())
        end = thai_to_arabic_local(end.strip())

        from datetime import datetime
        import pytz
        bangkok_tz = pytz.timezone('Asia/Bangkok')

        def normalize_dt(s: str, is_end: bool) -> str:
            if len(s) == 10:
                return f"{s} {'23:59:59' if is_end else '00:00:00'}"
            try:
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise HTTPException(400, f"Invalid datetime format: {s}")
            return s

        start_str = normalize_dt(start, False)
        end_str = normalize_dt(end, True)

        start_dt = bangkok_tz.localize(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S"))
        end_dt = bangkok_tz.localize(datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S"))

        type = _norm_zone(type) if type else None
        if type and type not in ("face", "car"):
            raise HTTPException(422, "type must be 'face' or 'car'")

        dep_value = (department or "").strip() if department is not None else None

        cur = db.cursor
        access_list = list(claims.access or [])
        allowed_cams = [a.split("cam:", 1)[1] if a.lower().startswith("cam:") else a for a in access_list]
        allowed_depts = [a for a in access_list if a and a not in camera_meta_by_name]
        if claims.department and claims.department not in allowed_depts:
            allowed_depts.append(claims.department)

        sql = """
        WITH face AS (
            SELECT
                f.timestamp  AS ts,
                f.camera_name,
                'face'       AS kind,
                f.full_name  AS subject,
                f.department AS meta1,
                f.emp_id     AS meta2,
                f.confidence,
                f.similarity
            FROM face_detection_details f
            WHERE f.timestamp BETWEEN %s AND %s
        ),
        car AS (
            SELECT
                cl.timestamp AT TIME ZONE 'Asia/Bangkok' AS ts,
                cl.camera_name,
                'car'         AS kind,
                cl.plate_number AS subject,
                cl.province  AS meta1,
                cl.status    AS meta2,
                NULL::double precision AS confidence,
                NULL::double precision AS similarity
            FROM car_log cl
            WHERE cl.timestamp AT TIME ZONE 'Asia/Bangkok' BETWEEN %s AND %s
        ),
        raw AS (
            SELECT * FROM face
            UNION ALL
            SELECT * FROM car
        )
        SELECT
            r.ts,
            r.camera_name,
            c.zone,
            r.kind,
            r.subject,
            r.meta1,
            r.meta2,
            r.confidence,
            r.similarity
        FROM raw r
        JOIN cameras c ON c.camera_name = r.camera_name
        WHERE 1=1
        """
        params = [start_dt, end_dt, start_dt, end_dt]

        if not claims.is_admin:
            conds = []
            if allowed_depts:
                conds.append("c.comp = ANY(%s)")
                params.append(allowed_depts)
            if allowed_cams:
                conds.append("r.camera_name = ANY(%s)")
                params.append(allowed_cams)
            if conds:
                sql += " AND (" + " OR ".join(conds) + ")"
            else:
                sql += " AND 1=0"

        if dep_value is not None:
            if dep_value in ("", "ไม่มีแผนก", "None"):
                sql += " AND c.comp IS NULL"
            else:
                sql += " AND c.comp = %s"
                params.append(dep_value)

        if type in ("face", "car"):
            sql += " AND r.kind = %s"
            params.append(type)

        if q:
            tokens = [t for t in q.strip().split() if t]
            for t in tokens:
                like = f"%{t}%"
                sql += """
                AND (
                    r.subject ILIKE %s OR
                    r.meta1   ILIKE %s OR
                    r.meta2   ILIKE %s OR
                    r.camera_name ILIKE %s
                )
                """
                params.extend([like, like, like, like])

        sql += " ORDER BY r.ts DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        items = []
        for ts, cam_name, z, kind, subject, meta1, meta2, confidence, similarity in rows:
            base = {
                "type": kind,
                "timestamp": ts.isoformat(),
                "camera_name": cam_name,
                "zone": z,
            }
            if kind == "face":
                base.update({
                    "full_name": subject,
                    "department": meta1,
                    "emp_id": meta2,
                    "confidence": confidence,
                    "similarity": similarity
                })
            else:
                base.update({
                    "plate": subject,
                    "province": meta1,
                    "status": meta2
                })
            items.append(base)

        return {"count": len(items), "items": items}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] /reports - {e}")
        try:
            db.log_error("report_all", str(e), None)
        except Exception:
            pass
        raise HTTPException(500, f"report error: {e}")

@app.get("/reports/by-video-file")
async def report_by_video_file(
    filename: str = Query(...),
    camera: str = Query(...),
    zone: str = Query(...),
    date: str = Query(..., description="YYYY-MM-DD"),
    claims: TokenClaims = Depends(require_user_flexible)
):
    cam_meta = camera_meta_by_name.get(camera)
    cam_comp = cam_meta.get("comp") if cam_meta else None
    user_access = claims.access or []
    if not claims.is_admin and cam_comp and cam_comp not in user_access:
         raise HTTPException(status_code=403, detail=f"No access to reports for camera {camera}")
    
    try:
        with runtime_settings_lock: # ⭐️ [เพิ่ม] อ่านค่าล่าสุด
            current_segment_minutes = runtime_settings.get("SEGMENT_MINUTES", 15)

        start_dt, end_dt = _get_time_range_from_filename(date, filename, current_segment_minutes)
        logger.info(f"[REPORT BY FILE] Querying detections for {camera} ({filename}) between {start_dt} and {end_dt}")

        cur = db.cursor
        sql = """
        WITH face AS (
            SELECT
                f.timestamp  AS ts,
                f.camera_name,
                'face'       AS kind,
                f.full_name  AS subject,
                f.department AS meta1,
                f.emp_id     AS meta2
            FROM face_detection_details f
            WHERE f.timestamp BETWEEN %s AND %s
            AND f.camera_name = %s
        ),
        car AS (
            SELECT
                cl.timestamp AT TIME ZONE 'Asia/Bangkok' AS ts,
                cl.camera_name,
                'car'        AS kind,
                cl.plate_number AS subject,
                cl.province  AS meta1,
                cl.status    AS meta2
            FROM car_log cl
            WHERE cl.timestamp AT TIME ZONE 'Asia/Bangkok' BETWEEN %s AND %s
            AND cl.camera_name = %s
        ),
        raw AS (
            SELECT * FROM face
            UNION ALL
            SELECT * FROM car
        )
        SELECT
            r.ts, r.camera_name, r.kind, r.subject, r.meta1, r.meta2
        FROM raw r
        ORDER BY r.ts ASC
        """
        
        params = [start_dt, end_dt, camera, start_dt, end_dt, camera]
        
        cur.execute(sql, params)
        rows = cur.fetchall()

        items = []
        for ts, cam_name, kind, subject, meta1, meta2 in rows:
            base = {
                "type": kind,
                "timestamp": ts.isoformat(),
                "camera_name": cam_name,
            }
            if kind == "face":
                base.update({
                    "full_name": subject,
                    "department": meta1,
                    "emp_id": meta2
                })
            else: # 'car'
                base.update({
                    "plate": subject,
                    "province": meta1,
                    "status": meta2
                })
            items.append(base)

        return {"count": len(items), "items": items}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] /reports/by-video-file - {e}", exc_info=True)
        raise HTTPException(500, f"Report by file error: {e}")

@app.get("/recordings")
async def list_recordings(
    claims: TokenClaims = Depends(require_user),
    camera: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    person_name: Optional[str] = Query(None)
):
    files = []
    try:
        for dept in os.listdir(RECORD_ROOT):
            dept_path = os.path.join(RECORD_ROOT, dept)
            if not os.path.isdir(dept_path): continue
            for zone_dir in os.listdir(dept_path):
                zone_path = os.path.join(dept_path, zone_dir)
                if not os.path.isdir(zone_path): continue
                for cam_dir in os.listdir(zone_path):
                    cam_path = os.path.join(zone_path, cam_dir)
                    if not os.path.isdir(cam_path): continue
                    
                    cam_meta = camera_meta_by_name.get(cam_dir)
                    cam_comp = cam_meta.get("comp") if cam_meta else dept
                    if not claims.is_admin and cam_comp and cam_comp not in claims.access:
                        continue

                    for date_dir in os.listdir(cam_path):
                        date_path = os.path.join(cam_path, date_dir)
                        if not os.path.isdir(date_path): continue
                        for fn in os.listdir(date_path):
                            if fn.lower().endswith(".mp4"):
                                full = os.path.join(date_path, fn)
                                try:
                                    stat = os.stat(full)
                                    files.append({
                                        "file": fn,
                                        "size_bytes": stat.st_size,
                                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                        "department": dept,
                                        "zone": zone_dir,
                                        "camera": cam_dir,
                                        "date": date_dir,
                                    })
                                except FileNotFoundError:
                                    continue
    except Exception as e:
        logger.error(f"[ERROR] list recordings error: {e}")
        raise HTTPException(500, f"list recordings error: {e}")

    files.sort(key=lambda x: x["modified"], reverse=True)
    
    filtered_files = files
    if camera:
        filtered_files = [f for f in filtered_files if f.get("camera") == camera]
    if zone:
        filtered_files = [f for f in filtered_files if f.get("zone") == zone]
    if date:
        filtered_files = [f for f in filtered_files if f.get("date") == date]

    if person_name and person_name.strip() and db:
        logger.info(f"Person search: '{person_name}' for cam '{camera}' on {date}")
        bangkok_tz = pytz.timezone('Asia/Bangkok')
        
        try:
            start_dt = bangkok_tz.localize(dt.datetime.strptime(f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S"))
            end_dt = bangkok_tz.localize(dt.datetime.strptime(f"{date} 23:59:59", "%Y-%m-%d %H:%M:%S"))
            
            person_timestamps = db.get_timestamps_for_person(start_dt, end_dt, camera, person_name)
        except Exception as e:
            logger.error(f"Failed to get person timestamps: {e}")
            person_timestamps = []

        if not person_timestamps:
            return [] 

        files_with_person = []
        for file_item in filtered_files:
            try:
                # --- ⭐️ [แก้ไข] อ่านค่า Segment ล่าสุด ⭐️ ---
                with runtime_settings_lock:
                    current_segment_minutes = runtime_settings.get("SEGMENT_MINUTES", 15)
                    
                f_start, f_end = _get_time_range_from_filename(
                    file_item["date"], 
                    file_item["file"], 
                    current_segment_minutes # ⭐️ (ใช้ค่านี)
                )
                
                found = False
                for ts in person_timestamps:
                    if f_start <= ts <= f_end:
                        found = True
                        break
                
                if found:
                    files_with_person.append(file_item)
                    
            except Exception as e:
                logger.warning(f"Could not parse time range for {file_item['file']}: {e}")
        
        return files_with_person
        
    return filtered_files

@app.get("/reports/camera-events")
def get_camera_events_report(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    _: TokenClaims = Depends(require_admin)
):
    if not db:
        raise HTTPException(500, "Database not connected")
    try:
        items = db.get_camera_events(start, end)
        return {"items": items}
    except Exception as e:
        logger.error(f"[API /reports/camera-events] Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    
@app.get("/recordings/{department}/{zone}/{camera}/{date}/{filename}")
async def fetch_recording(
    department: str, zone: str, camera: str, date: str, filename: str,
    claims: TokenClaims = Depends(require_user_flexible)
):
    cam_meta = camera_meta_by_name.get(camera)
    cam_comp = cam_meta.get("comp") if cam_meta else department

    user_access = claims.access or []
    if not claims.is_admin and cam_comp and cam_comp not in user_access:
        logger.warning(f"Access Denied: User '{claims.sub}' (Access: {user_access}) tried to access camera '{camera}' (Comp: {cam_comp})")
        raise HTTPException(status_code=403, detail=f"No access to recordings for camera {camera}")

    safe_base = os.path.normpath(RECORD_ROOT)
    def sanitize_part(part: str) -> str:
        part = part.replace("..", "").replace("/", "").replace("\\", "")
        return part

    safe_dept = sanitize_part(department)
    safe_zone = sanitize_part(zone)
    safe_camera = sanitize_part(camera)
    safe_date = sanitize_part(date)
    safe_filename = sanitize_part(filename)

    path = os.path.join(safe_base, safe_dept, safe_zone, safe_camera, safe_date, safe_filename)

    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(safe_base)
    if not abs_path.startswith(abs_base):
         logger.error(f"Path Traversal Attempt Denied: User '{claims.sub}', Requested Path: '{path}', Resolved Path: '{abs_path}'")
         raise HTTPException(status_code=403, detail="Forbidden path")

    if not os.path.isfile(abs_path):
        logger.error(f"File Not Found: User '{claims.sub}' requested non-existent file: '{abs_path}'")
        raise HTTPException(404, "File not found")

    logger.info(f"Access Granted: User '{claims.sub}' accessing recording: '{abs_path}'")
    return FileResponse(abs_path, media_type="video/mp4", filename=safe_filename)

class SegmentUpdate(BaseModel):
    minutes: int

@app.post("/admin/settings/segment")
def admin_set_segment_minutes(
    payload: SegmentUpdate,
    _: TokenClaims = Depends(require_admin)
):
    new_minutes = max(1, min(payload.minutes, 120)) 

    with runtime_settings_lock:
        runtime_settings["SEGMENT_MINUTES"] = new_minutes
    
    save_persistent_settings()

    
    updated_count = 0
    for item in workers:
        recorder = item[5] if len(item) > 5 else None 
        
        if recorder and hasattr(recorder, "update_segment_minutes"):
            try:
                recorder.update_segment_minutes(new_minutes)
                updated_count += 1
            except Exception as e:
                logger.error(f"Failed to update segment time for {getattr(recorder, '_camera_id', '?')}: {e}")
                
    logger.info(f"Admin set segment duration to {new_minutes} mins. {updated_count} running recorders updated.")
    return {"ok": True, "new_segment_minutes": new_minutes, "workers_updated": updated_count}
# ---------- Admin ----------
@app.post("/admin/shutdown")
async def admin_shutdown(payload: dict = Body(...)):
    if payload.get("token") != ADMIN_TOKEN:
        raise HTTPException(403, "forbidden")
    for item in workers:
        worker, t, _fq, _lq, _sq, rec = item
        try:
            worker.stop()
            if rec and rec.is_recording():
                rec.stop_recording()
            t.join(timeout=2.0)
        except Exception:
            pass
    try:
        for name in list(sub_preview_threads.keys()):
            stop_sub_preview(name)
    except Exception:
        pass
    await asyncio.sleep(0.2)
    os._exit(0)

# ===================== Run =====================
if __name__ == "__main__":
    import uvicorn
    KEEP_ALIVE_SEC = 24 * 60 * 60 * 365
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        lifespan="on",
        timeout_keep_alive=KEEP_ALIVE_SEC,
    )