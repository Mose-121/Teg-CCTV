import os
import cv2
import math
import time
import threading
import queue
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import perf_counter
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from service.database import Database
from service import utils
import logging

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# Suppress NumPy warning
np.seterr(all='warn')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to load dotenv lazily
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ================== Safe ENV helpers ==================
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(key: str, default: int, minv: int = None, maxv: int = None) -> int:
    v = os.getenv(key, "").strip()
    try:
        x = int(v)
    except Exception:
        x = default
    if minv is not None:
        x = max(minv, x)
    if maxv is not None:
        x = min(maxv, x)
    return x


def _env_float(key: str, default: float, minv: float = None, maxv: float = None) -> float:
    v = os.getenv(key, "").strip()
    try:
        x = float(v)
    except Exception:
        x = default
    if minv is not None:
        x = max(minv, x)
    if maxv is not None:
        x = min(maxv, x)
    return x


def _env_size_pair(key: str, default: Tuple[int, int]) -> Tuple[int, int]:
    raw = os.getenv(key, "").strip()
    try:
        parts = [p for p in raw.split(",") if p]
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            return (max(64, w), max(64, h))
    except Exception:
        pass
    return default


# ================== ENV / CONFIG (safe defaults) ==================
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l").strip()
RUNTIME_DET_SIZE = _env_size_pair("RUNTIME_DET_SIZE", (640, 640))
FACE_DET_THRESH = _env_float("FACE_DET_THRESH", 0.6, 0.1, 0.99)

FACE_SIM_THRESH = _env_float("FACE_SIM_THRESH", 0.75, 0.5, 0.99)
FACE_DET_INTERVAL = _env_float("FACE_DET_INTERVAL", 0.20, 0.01, 5.0)
FACE_MIN_LOG_INTERVAL = _env_float("FACE_MIN_LOG_INTERVAL", 1.5, 0.0, 3600.0)
FACE_EMB_SMOOTH_K = _env_int("FACE_EMB_SMOOTH_K", 3, 1, 32)

PREVIEW_WIDTH = _env_int("PREVIEW_W", 1280, 64, 4096)
PREVIEW_HEIGHT = _env_int("PREVIEW_H", 720, 64, 4096)
PREVIEW_FROM_CAMERA = _env_bool("PREVIEW_FROM_CAMERA", True)
PREVIEW_COLOR_RAW = _env_bool("PREVIEW_COLOR_RAW", True)

PREPROC_CLAHE = _env_bool("PREPROC_CLAHE", False)
PREPROC_CLAHE_CLIP = _env_float("PREPROC_CLAHE_CLIP", 2.0, 0.1, 10.0)
PREPROC_CLAHE_TILE = _env_int("PREPROC_CLAHE_TILE", 8, 1, 64)
PREPROC_GAMMA = _env_bool("PREPROC_GAMMA", False)
PREPROC_GAMMA_VALUE = _env_float("PREPROC_GAMMA_VALUE", 1.2, 0.1, 5.0)
PREPROC_DENOISE = _env_bool("PREPROC_DENOISE", False)
PREPROC_DENOISE_H = _env_float("PREPROC_DENOISE_H", 5.0, 0.0, 20.0)
PREPROC_SHARPEN = _env_bool("PREPROC_SHARPEN", False)
PREPROC_SHARPEN_ALPHA = _env_float("PREPROC_SHARPEN_ALPHA", 0.5, 0.0, 2.0)

POSE_AWARE_TTA = _env_bool("POSE_AWARE_TTA", True)
RECTIFY_TOPDOWN = _env_bool("RECTIFY_TOPDOWN", True)
DYNAMIC_SIM_THRESH = _env_bool("DYNAMIC_SIM_THRESH", True)

TTA_FLIP_H = _env_bool("TTA_FLIP_H", True)
TTA_ROTATION = _env_bool("TTA_ROTATION", True)
TTA_ROTATION_ANGLE = _env_float("TTA_ROTATION_ANGLE", 15.0, 0.0, 45.0)
TTA_SCALE = _env_bool("TTA_SCALE", True)
TTA_SCALE_FACTOR_MIN = _env_float("TTA_SCALE_FACTOR_MIN", 0.95, 0.5, 1.5)
TTA_SCALE_FACTOR_MAX = _env_float("TTA_SCALE_FACTOR_MAX", 1.05, 0.5, 2.0)
TTA_SHEAR = _env_bool("TTA_SHEAR", True)
TTA_NUM_SAMPLES = _env_int("TTA_NUM_SAMPLES", 5, 1, 8)

SIM_USE_TTA_MAX = _env_bool("SIM_USE_TTA_MAX", True)
SIM_USE_ROLLING_MAX = _env_bool("SIM_USE_ROLLING_MAX", True)

SIM_ROLL_SECONDS = _env_float("SIM_ROLL_SECONDS", 2.5, 0.1, 10.0)
MIN_FACE_SIZE = _env_int("MIN_FACE_SIZE", 96, 16, 2048)
MIN_BLUR_VAR = _env_float("MIN_BLUR_VAR", 100.0, 0.0, 10000.0)

SAVE_MIN_DET_SCORE = _env_float("SAVE_MIN_DET_SCORE", 0.70, 0.0, 1.0)
SAVE_MIN_BLUR_VAR = _env_float("SAVE_MIN_BLUR_VAR", 100.0, 0.0, 10000.0)

TRACK_GRID_SIZE = _env_int("TRACK_GRID_SIZE", 64, 8, 512)

UI_FONT_PATH = os.getenv("UI_FONT_PATH", r"")
UI_FONT_SIZE = _env_int("UI_FONT_SIZE", 28, 8, 128)

FACE_ID_LOCK_SEC = _env_float("FACE_ID_LOCK_SEC", 10.0, 1.0, 60.0)
FACE_LOG_STABLE_SEC = _env_float("FACE_LOG_STABLE_SEC", 0.02, 0.0, 30.0)

SHOW_FACE_METRICS = _env_bool("SHOW_FACE_METRICS", True)
METRICS_DECIMALS = _env_int("METRICS_DECIMALS", 2, 0, 6)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_EVERY_SEC = _env_float("LOG_EVERY_SEC", 5.0, 0.5, 600.0)
LOG_PER_DET_DEBUG = _env_bool("LOG_PER_DET_DEBUG", False)

OPENCV_NUM_THREADS = _env_int("OPENCV_NUM_THREADS", -1, -1, 64)

# --- Dynamic Face Update Thresholds ---
UPDATE_SIM_THRESH = _env_float("UPDATE_SIM_THRESH", 0.82, 0.75, 0.99)
UPDATE_MIN_BLUR_VAR = _env_float("UPDATE_MIN_BLUR_VAR", 200.0, 100.0, 10000.0)
UPDATE_MAX_YAW_DEG = _env_float("UPDATE_MAX_YAW_DEG", 15.0, 5.0, 45.0)
UPDATE_COOLDOWN_SEC = _env_float("UPDATE_COOLDOWN_SEC", 1800.0, 60.0, 86400.0)

# Logging Setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ================== Similarity Helpers ==================
def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


# ================== ORT Providers ==================
def _get_ort_providers() -> List[str]:
    req = os.getenv("INSIGHTFACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
    requested = [p.strip() for p in req.split(",") if p.strip()]
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        ordered = []
        if "CUDAExecutionProvider" in avail:
            ordered.append("CUDAExecutionProvider")
        for prov in requested:
            if prov in avail and prov not in ordered:
                ordered.append(prov)
        if not ordered and "CPUExecutionProvider" in avail:
            ordered.append("CPUExecutionProvider")
        if not ordered:
            raise RuntimeError(f"No valid ONNXRuntime providers found. Available: {sorted(avail)}")
        logger.info(f"[INSIGHT] Using providers: {ordered}, available={sorted(avail)}")
        return ordered
    except Exception as e:
        logger.error(f"ONNXRuntime provider resolution failed: {e}")
        return ["CPUExecutionProvider"]


# ================== Image Utils ==================
_CLAHE_CACHE: Dict[Tuple[float, int], Any] = {}
_GAMMA_LUT: Dict[float, np.ndarray] = {}

if OPENCV_NUM_THREADS >= 0:
    try:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
        logger.info(f"[OpenCV] setNumThreads({OPENCV_NUM_THREADS})")
    except Exception as _e:
        logger.warning(f"[OpenCV] setNumThreads failed: {_e}")

cv2.setUseOptimized(True)


def _get_clahe(cl: float, ts: int):
    key = (cl, ts)
    c = _CLAHE_CACHE.get(key)
    if c is None:
        c = cv2.createCLAHE(cl, (ts, ts))
        _CLAHE_CACHE[key] = c
    return c


def _gamma_lut(g: float) -> np.ndarray:
    lut = _GAMMA_LUT.get(g)
    if lut is None:
        ig = 1.0 / max(1e-6, g)
        lut = np.array([((i / 255.0) ** ig) * 255 for i in range(256)], dtype=np.uint8)
        _GAMMA_LUT[g] = lut
    return lut


def _apply_clahe(img: np.ndarray, cl: float = 2.0, ts: int = 8) -> np.ndarray:
    try:
        if img.ndim < 3 or (img.ndim == 3 and img.shape[2] == 1):
            l = img if img.ndim == 2 else img[:, :, 0]
            lc = _get_clahe(cl, ts).apply(l)
            return cv2.cvtColor(lc, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else cv2.merge([lc, lc, lc])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _get_clahe(cl, ts).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except cv2.error as e:
        logger.warning(f"[_apply_clahe] Err: {e}. Returning original.")
        return img


def _apply_gamma(img: np.ndarray, g: float = 1.2) -> np.ndarray:
    return cv2.LUT(img, _gamma_lut(g))


def _apply_denoise(img: np.ndarray, h: float = 5.0) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def _apply_sharpen(img: np.ndarray, a: float = 0.5) -> np.ndarray:
    b = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1 + a, b, -a, 0)


def _preprocess_for_preview(img: np.ndarray) -> np.ndarray:
    p = img
    if PREPROC_CLAHE:
        p = _apply_clahe(p, PREPROC_CLAHE_CLIP, PREPROC_CLAHE_TILE)
    if PREPROC_GAMMA:
        p = _apply_gamma(p, PREPROC_GAMMA_VALUE)
    if PREPROC_DENOISE:
        p = _apply_denoise(p, PREPROC_DENOISE_H)
    if PREPROC_SHARPEN:
        p = _apply_sharpen(p, PREPROC_SHARPEN_ALPHA)
    return p


def _apply_horizontal_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def _apply_rotation(img: np.ndarray, ang: float) -> np.ndarray:
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(c, ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _apply_scale(img: np.ndarray, sf: float) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = max(1, int(w * sf)), max(1, int(h * sf))
    try:
        sc = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
        if nw <= w and nh <= h:
            dh, dw = h - nh, w - nw
            t, b = dh // 2, dh - (dh // 2)
            l, r = dw // 2, dw - (dw // 2)
            return cv2.copyMakeBorder(sc, t, b, l, r, cv2.BORDER_REFLECT)
        else:
            sx = (nw - w) // 2
            sy = (nh - h) // 2
            cr = sc[sy:sy + h, sx:sx + w]
            if cr.shape[:2] != (h, w):
                return cv2.resize(cr, (w, h), cv2.INTER_LINEAR)
            return cr
    except cv2.error as e:
        logger.warning(f"[_apply_scale] Err: {e}. Returning original.")
        return img


def _adjust_brightness(img: np.ndarray, d: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=d)


def _adjust_contrast(img: np.ndarray, f: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=f, beta=0)


def _laplacian_var_gray(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 and img.shape[2] == 3 else img
        return float(cv2.Laplacian(g, cv2.CV_64F, ksize=3).var())
    except cv2.error:
        return 0.0


def _assign_track_key(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return (cx // TRACK_GRID_SIZE, cy // TRACK_GRID_SIZE)


def _draw_text_pil(frame: np.ndarray, text: str, org: Tuple[int, int],
                   font_p: str, font_s: int,
                   fg=(0, 0, 0), bg=(255, 255, 255, 180), pad: int = 4) -> np.ndarray:
    """
    Draw multiline text with PIL if available and fallback to OpenCV putText.
    """
    if not _PIL_OK or not font_p or not os.path.exists(font_p):
        y0 = org[1]
        for i, l in enumerate(text.split("\n")):
            cv2.putText(frame, l, (org[0], y0 + i * int(max(18, font_s * 1.1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        return frame

    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(img, 'RGBA')
        try:
            ft = ImageFont.truetype(font_p, font_s)
        except Exception:
            ft = ImageFont.load_default()
        # multiline_textbbox for PIL >= 8
        bbox = dr.multiline_textbbox((0, 0), text, font=ft, spacing=2)
        l, t, r, b = bbox
        tw, th = r - l, b - t
        x, y = org
        yt = y - th - pad * 2
        if yt < pad:
            yt = y + pad
        if bg:
            dr.rounded_rectangle((x - pad, yt - pad, x + tw + pad, yt + th + pad), radius=6, fill=bg)
        dr.multiline_text((x, yt), text, fill=(fg[2], fg[1], fg[0]), font=ft, spacing=2)
        frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"[_draw_text_pil] Error: {e}. OpenCV fallback.")
        y0 = org[1]
        for i, l in enumerate(text.split("\n")):
            cv2.putText(frame, l, (org[0], y0 + i * int(max(18, font_s * 1.1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return frame


def _draw_sim_tag(frame: np.ndarray, box: Tuple[int, int, int, int], sim: Optional[float],
                  dec: int = 2, thr: Optional[float] = None) -> int:
    if sim is None:
        return 0
    x1, y1, x2, y2 = map(int, box)
    txt = f"S:{sim:.{dec}f}"
    thr = thr if thr else FACE_SIM_THRESH
    bg = (60, 220, 60) if sim >= thr else (0, 200, 255)
    fg = (0, 0, 0)
    ft = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.75
    tk = 2
    (tw, th), _ = cv2.getTextSize(txt, ft, sc, tk)
    pad = 4
    yt = max(0, y1 - th - pad * 2)
    xt = min(frame.shape[1], x1 + tw + pad * 2)
    tl = (x1, yt)
    br = (xt, y1)
    cv2.rectangle(frame, tl, br, bg, -1)
    cv2.putText(frame, txt, (x1 + pad, y1 - pad), ft, sc, fg, tk, cv2.LINE_AA)
    return th + pad * 2


def _fmt_metrics(det: Optional[float] = None, sim: Optional[float] = None,
                 blur: Optional[float] = None, dec: int = 2) -> str:
    parts = []
    if det is not None:
        parts.append(f"C:{det:.{dec}f}")
    if sim is not None:
        parts.append(f"S:{sim:.{dec}f}")
    if blur is not None:
        st = "คม" if blur >= 200 else ("พอใช้" if blur >= 120 else "เบลอ")
        parts.append(f"B:{blur:.0f}({st})")
    return " ".join(parts)


# ================== Math Helpers ==================
def _snap32(w: int, h: int) -> Tuple[int, int]:
    return max(256, (w // 32) * 32), max(256, (h // 32) * 32)


def _even_dim(x: int) -> int:
    return (x + 1) & ~1


# ================== Pose / Quality helpers ==================
def _estimate_roll_deg_from_kps5(kps: np.ndarray) -> float:
    if kps is None or kps.shape != (5, 2):
        return 0.0
    re, le = kps[0], kps[1]
    dy = float(le[1] - re[1])
    dx = float(le[0] - re[0]) + 1e-6
    return math.degrees(math.atan2(dy, dx))


def _estimate_yaw_deg_from_kps5(kps: np.ndarray) -> float:
    if kps is None or kps.shape != (5, 2):
        return 0.0
    re, le, n = kps[0], kps[1], kps[2]
    ec = (re + le) / 2
    ed = np.linalg.norm(re - le)
    if ed < 1e-6:
        return 0.0
    no = n[0] - ec[0]
    r = np.clip(no / (ed * 0.5 + 1e-6), -1.0, 1.0)
    yaw = math.degrees(math.asin(r))
    return float(np.clip(yaw, -90, 90))


def _anti_topdown_compress_fix(img: np.ndarray, fac: float = 1.06) -> np.ndarray:
    h, w = img.shape[:2]
    nh = max(1, int(h * fac))
    try:
        st = cv2.resize(img, (w, nh), cv2.INTER_LINEAR)
        s = (nh - h) // 2
        e = s + h
        if s < 0 or e > nh:
            logger.warning(f"[_anti_topdown_compress_fix] Crop fail. Returning original.")
            return img
        cr = st[s:e, :]
        if cr.shape[0] != h:
            return cv2.resize(cr, (w, h), cv2.INTER_AREA)
        return cr
    except cv2.error as e:
        logger.warning(f"[_anti_topdown_compress_fix] Err: {e}. Returning original.")
        return img


def _dynamic_sim_threshold(base: float, blur_var: float, det_score: float, yaw_deg: float) -> float:
    thr = base
    if blur_var < 120:
        thr -= 0.03
    if det_score < 0.7:
        thr -= 0.03
    if abs(yaw_deg) > 20:
        penalty = max(0, abs(yaw_deg) - 20) / 70.0
        thr -= 0.05 * penalty
    thr = max(0.75, thr)
    return min(thr, 0.95)


# ================== Stats ==================
@dataclass
class LoopStats:
    last_ts: float = field(default_factory=time.time)
    last_report_ts: float = field(default_factory=time.time)
    frame_count: int = 0
    det_cycles: int = 0
    dropped_frames: int = 0
    faces_total: int = 0
    matches: int = 0
    unknowns: int = 0
    avg_sim_accum: float = 0.0
    avg_sim_n: int = 0
    t_det_ms_acc: float = 0.0
    t_embed_ms_acc: float = 0.0
    t_loop_ms_acc: float = 0.0
    loops: int = 0

    def tick_frame(self):
        self.frame_count += 1

    def add_det_cycle(self, fc: int, tms: float):
        self.det_cycles += 1
        self.faces_total += fc
        self.t_det_ms_acc += tms

    def add_embed_ms(self, tms: float):
        self.t_embed_ms_acc += tms

    def add_match(self, sim: float):
        self.matches += 1
        self.avg_sim_accum += sim
        self.avg_sim_n += 1

    def add_unknown(self):
        self.unknowns += 1

    def add_loop_time(self, tms: float):
        self.t_loop_ms_acc += tms
        self.loops += 1

    def maybe_report(self, cam: str):
        now = time.time()
        if (now - self.last_report_ts) < LOG_EVERY_SEC:
            return
        el = max(1e-6, now - self.last_report_ts)
        fps = self.frame_count / el
        det_r = self.det_cycles / el
        face_r = self.faces_total / el
        avg_s = (self.avg_sim_accum / self.avg_sim_n) if self.avg_sim_n else 0.0
        avg_det = (self.t_det_ms_acc / max(1, self.det_cycles))
        avg_emb = (self.t_embed_ms_acc / max(1, self.faces_total)) if self.faces_total else 0.0
        avg_loop = (self.t_loop_ms_acc / max(1, self.loops))
        logger.info(
            f"[Stats {cam}] FPS:{fps:.1f}|Det/s:{det_r:.1f}|Faces/s:{face_r:.1f}|Match:{self.matches} "
            f"Unk:{self.unknowns} AvgSim:{avg_s:.3f}|T(Det:{avg_det:.1f}ms Emb:{avg_emb:.1f}ms Loop:{avg_loop:.1f}ms)|"
            f"Drop:{self.dropped_frames}"
        )
        # reset
        self.last_report_ts = now
        self.frame_count = self.det_cycles = self.faces_total = self.matches = self.unknowns = self.avg_sim_n = self.loops = self.dropped_frames = 0
        self.avg_sim_accum = self.t_det_ms_acc = self.t_embed_ms_acc = self.t_loop_ms_acc = 0.0


# ================== Worker ==================
class FaceCameraWorker:
    def __init__(self, id, name, camera_url, zone=None, enter=None, department=None, user_access=None):
        self.id = id
        self.name = str(name)
        self.camera_url = camera_url
        self.zone = zone or "face"
        self.enter = enter or "IN"
        self.department = department
        self.user_access = user_access or []
        self.running = False
        self.rec_q = None
        self.ana_q = None

        self._kn_lock = threading.RLock()
        self._log_cooldown: Dict[str, float] = {}
        self._id_locks: Dict[Any, Dict[str, Any]] = {}
        self._emb_hist = defaultdict(lambda: deque(maxlen=max(3, int(SIM_ROLL_SECONDS * 10))))
        self._update_cooldown: Dict[Any, float] = {}

        self._stream_thread = None
        self._stop_event = threading.Event()

        try:
            self.db = Database()
            assert self.db.conn
            logger.info(f"[Face {self.name}] DB initialized")
        except Exception as e:
            logger.error(f"[Face {self.name}] DB init failed: {e}")
            self.db = None

        providers = _get_ort_providers()
        self.face_app = None
        dw, dh = _snap32(*RUNTIME_DET_SIZE)
        if (dw, dh) != tuple(RUNTIME_DET_SIZE):
            logger.info(f"[INSIGHT] Adjusted det-size {RUNTIME_DET_SIZE}->{(dw, dh)}")

        try:
            self.face_app = FaceAnalysis(name=FACE_MODEL_NAME, providers=providers)
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            self.face_app.prepare(ctx_id=ctx_id, det_size=(dw, dh), det_thresh=FACE_DET_THRESH)
            prov_list = getattr(self.face_app, 'providers', ['Unknown'])
            logger.info(
                f"[Face {self.name}] FaceAnalysis using providers: {prov_list} | ctx={ctx_id} | det={dw}x{dh} | thr={FACE_DET_THRESH:.2f}"
            )
        except Exception as e:
            logger.exception(f"[Face {self.name}] DETAILED ERROR during FaceAnalysis init with {providers}:")
            logger.error(f"[Face {self.name}] Init providers {providers} failed -> fallback CPU: {e}")
            try:
                cpu_providers = ["CPUExecutionProvider"]
                self.face_app = FaceAnalysis(name=FACE_MODEL_NAME, providers=cpu_providers)
                self.face_app.prepare(ctx_id=-1, det_size=(dw, dh), det_thresh=FACE_DET_THRESH)
                logger.info(f"[Face {self.name}] FaceAnalysis fallback CPU OK | det={dw}x{dh} | thr={FACE_DET_THRESH:.2f}")
            except Exception as e2:
                logger.critical(f"[Face {self.name}] FATAL: Cannot init FaceAnalysis: {e2}", exc_info=True)
                self.face_app = None

        self.rec_model = self.face_app.models.get('recognition') if self.face_app and hasattr(self.face_app, 'models') else None
        if self.rec_model is None:
            logger.warning(f"[Face {self.name}] Rec model unavailable!")

        self.known_names: List[str] = []
        self.known_depts: List[str] = []
        self.known_ids: List[Any] = []
        self.known_embs = np.empty((0, 512), dtype=np.float32)
        self.reload_known_faces()

        self.preview_w, self.preview_h = PREVIEW_WIDTH, PREVIEW_HEIGHT
        self._raw_wh = None

        logger.info(
            f"[Face {self.name}] ColdStart OK | Known:{self.known_embs.shape[0]} | Thr:{FACE_SIM_THRESH:.2f} | "
            f"DynThr:{DYNAMIC_SIM_THRESH} | Roll:{SIM_ROLL_SECONDS:.1f}s | Lock:{FACE_ID_LOCK_SEC:.1f}s | TTA:{SIM_USE_TTA_MAX}"
        )

    # --- Methods ---
    def reload_known_faces(self):
        if not self.db:
            logger.warning(f"[Face {self.name}] DB unavailable, cannot reload faces.")
            return
        try:
            known = self.db.load_known_faces()  # Assumes returns List[(emp_id, avg_emb_norm, name, dept)]
            logger.info(f"[Face {self.name}] Reloading {len(known)} known faces (averaged).")
            with self._kn_lock:
                if not known:
                    self.known_ids, self.known_names, self.known_depts = [], [], []
                    self.known_embs = np.empty((0, 512), dtype=np.float32)
                else:
                    self.known_ids = [r[0] for r in known]
                    self.known_names = [r[2] for r in known]
                    self.known_depts = [(r[3] or "Unknown") for r in known]
                    self.known_embs = np.stack([r[1] for r in known]).astype(np.float32)
                logger.info(f"[Face {self.name}] Known embeddings updated: {self.known_embs.shape}")
        except Exception as e:
            logger.error(f"[Face {self.name}] reload_known_faces error: {e}")
            self._safe_log_error("reload_known_faces", str(e), self.name)

    def set_output_queues(self, rec_q, ana_q):
        self.rec_q = rec_q
        self.ana_q = ana_q

    def _safe_log_error(self, et: str, msg: str, cid: str):
        if self.db and hasattr(self.db, 'log_error'):
            try:
                self.db.log_error(et, msg, cid)
            except Exception as e:
                logger.error(f"[DB ERROR] Log error failed for {cid}: {e}")
        else:
            logger.error(f"[Face {cid}] DB unavailable, cannot log error: {et} - {msg}")

    def _draw_box(self, frame: np.ndarray, box, name, similarity=None, emp_id=None, department=None, det_score=None, blur_var=None):
        x1, y1, x2, y2 = map(int, box)
        hf, wf = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(wf - 1, x2), min(hf - 1, y2)
        if x1 >= x2 or y1 >= y2:
            return
        color = (0, 255, 0) if (similarity is not None and similarity >= FACE_SIM_THRESH) else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        parts = [p for p in (name, emp_id, department) if p]
        line1 = " | ".join(parts) if parts else "Unknown"
        line2 = _fmt_metrics(det_score, similarity, blur_var, METRICS_DECIMALS) if SHOW_FACE_METRICS else ""
        label = line1 if not line2 else f"{line1}\n{line2}"
        uh = _draw_sim_tag(frame, (x1, y1, x2, y2), similarity, METRICS_DECIMALS, FACE_SIM_THRESH) if similarity is not None else 0
        bp = (x1, max(0, y1 - uh - 6))
        _draw_text_pil(frame, label, bp, UI_FONT_PATH, UI_FONT_SIZE, fg=(0, 0, 0), bg=(255, 255, 255, 180), pad=4)

    def _embed_face(self, frame: np.ndarray, face) -> np.ndarray:
        """
        Return stacked embeddings (TTA samples x 512) or empty array if failed.
        """
        if self.rec_model is None:
            return np.empty((0, 512), dtype=np.float32)

        al = None
        if getattr(face, "kps", None) is not None:
            try:
                al = face_align.norm_crop(frame, landmark=face.kps, image_size=112)
            except Exception as ae:
                logger.debug(f"[_embed_face] norm_crop fail: {ae}")

        if al is None or not hasattr(al, "shape") or al.shape[:2] != (112, 112):
            if hasattr(face, 'embedding') and face.embedding is not None and getattr(face.embedding, 'size', 0) == 512:
                logger.debug("[_embed_face] Align fail, use direct emb.")
                return l2_normalize(face.embedding.astype(np.float32)).reshape(1, -1)
            else:
                logger.warning("[_embed_face] Align fail, no direct emb.")
                return np.empty((0, 512), dtype=np.float32)

        if RECTIFY_TOPDOWN:
            al = _anti_topdown_compress_fix(al, 1.06)

        v = [al]
        try:
            roll = _estimate_roll_deg_from_kps5(face.kps)
            yaw = _estimate_yaw_deg_from_kps5(face.kps)
            if POSE_AWARE_TTA and abs(roll) > 5.0:
                v.append(_apply_rotation(al, float(np.clip(-roll, -TTA_ROTATION_ANGLE, TTA_ROTATION_ANGLE))))
            if TTA_FLIP_H and abs(yaw) > 30.0:
                v.append(_apply_horizontal_flip(al))
            v.append(_adjust_brightness(al, 15.0))
            v.append(_adjust_contrast(al, 1.15))
            v.append(_apply_sharpen(al, 0.3))
            v = v[:TTA_NUM_SAMPLES]
        except Exception as e:
            logger.warning(f"[_embed_face] TTA gen fail: {e}")

        embs = []
        for i, img in enumerate(v):
            try:
                if img is not None and hasattr(img, "shape") and img.shape[:2] == (112, 112):
                    feat = self.rec_model.get_feat(img)
                    emb = l2_normalize(feat.reshape(-1).astype(np.float32))
                    embs.append(emb)
            except Exception as e:
                logger.debug(f"[_embed_face] feat fail {i}: {e}")

        if embs:
            return np.stack(embs, axis=0)
        return np.empty((0, 512), dtype=np.float32)

    def save_face(self, frame: np.ndarray, face, name: str, department: str = "Unknown") -> bool:
        if self.db is None or not hasattr(self.db, 'add_employee'):
            logger.error(f"[Face {self.name}] DB not ready/missing method.")
            return False
        try:
            embs = self._embed_face(frame, face)
            if embs.shape[0] == 0:
                logger.warning(f"[Face {self.name}] No emb for save.")
                return False
            emb = embs[0]
            x1, y1, x2, y2 = face.bbox.astype(int)
            hf, wf = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(wf - 1, x2), min(hf - 1, y2)
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"[Face {self.name}] Invalid crop box.")
                return False
            fi = frame[y1:y2, x1:x2]
            det = float(getattr(face, "det_score", 0.0))
            blur = _laplacian_var_gray(fi)
            if det < SAVE_MIN_DET_SCORE or blur < SAVE_MIN_BLUR_VAR:
                logger.info(f"[Face {self.name}] Skip save {name}: score={det:.2f}, blur={blur:.1f}")
                return False
            ok, buf = cv2.imencode(".jpg", fi, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                logger.warning(f"[Face {self.name}] Encode fail.")
                return False
            img_bytes = buf.tobytes()
            al_jpg = None
            try:
                al = face_align.norm_crop(frame, landmark=face.kps, image_size=112)
                if al is not None and al.shape[:2] == (112, 112):
                    ok_al, buf_al = cv2.imencode(".jpg", al, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    al_jpg = buf_al.tobytes() if ok_al else None
            except Exception:
                pass
            saved = self.db.add_employee(emp_id=name, name=name, department=department,
                                         image_data=img_bytes, embedding=emb.tobytes(),
                                         view_hint="center", aligned_image_data=al_jpg)
            if saved:
                logger.info(f"[Face {self.name}] Saved {name} via DB.")
                self.reload_known_faces()
                return True
            else:
                logger.error(f"[Face {self.name}] DB add fail {name}.")
                return False
        except Exception as e:
            logger.error(f"[Face {self.name}] Save error {name}: {e}", exc_info=True)
            self._safe_log_error("save_face", str(e), self.name)
            return False

    def run(self, frame_queue, log_queue, status_queue, save_face_callback=None):
        if self.face_app is None or self.db is None:
            logger.critical(f"[Face {self.name}] Worker cannot start - FaceApp/DB fail.")
            status_queue.put((self.name, "Init Fail"))
            return

        self.running = True
        self._stop_event.clear()
        q_frames: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=8)
        self._stream_thread = threading.Thread(
            target=utils.read_frames_stable,
            args=(self.camera_url, self._stop_event, q_frames),
            kwargs={"prefer_sub_url": None},
            daemon=True
        )
        self._stream_thread.start()

        status_queue.put((self.name, "เชื่อมต่อแล้ว"))
        logger.info(f"[Face {self.name}] Worker started.")
        stats = LoopStats()
        last_det_ts = 0.0
        last_dets: List[Tuple] = []

        try:
            if OPENCV_NUM_THREADS == -1:
                cv2.setNumThreads(0)
            elif OPENCV_NUM_THREADS >= 0:
                cv2.setNumThreads(OPENCV_NUM_THREADS)
        except Exception:
            pass

        save_dir = os.path.join("face_crops", self.name)
        os.makedirs(save_dir, exist_ok=True)

        while self.running and not self._stop_event.is_set():
            try:
                f: Optional[np.ndarray] = None
                try:
                    f = q_frames.get(timeout=3.0)
                    # drain to latest
                    while not q_frames.empty():
                        try:
                            f = q_frames.get_nowait()
                        except queue.Empty:
                            break
                except queue.Empty:
                    if self.running:
                        status_queue.put((self.name, "Timeout"))
                        logger.warning(f"[Face {self.name}] Stream timeout.")
                    continue

                if f is None or not isinstance(f, np.ndarray) or f.size == 0:
                    if self.running:
                        logger.warning(f"[Face {self.name}] Invalid frame.")
                        status_queue.put((self.name, "Invalid Frame"))
                        time.sleep(0.1)
                    continue

                loop_t0 = perf_counter()
                stats.tick_frame()
                frame_h, frame_w = f.shape[:2]
                if PREVIEW_FROM_CAMERA and self._raw_wh != (frame_w, frame_h):
                    self._raw_wh = (frame_w, frame_h)
                    self.preview_w = _even_dim(frame_w)
                    self.preview_h = _even_dim(frame_h)
                    logger.info(f"[Face {self.name}] Preview size set: {self.preview_w}x{self.preview_h}")

                now_perf = perf_counter()
                now_ts = time.time()
                do_detect = (now_perf - last_det_ts) >= FACE_DET_INTERVAL

                # Cleanup locks/history
                if self._id_locks:
                    expired_locks = [k for k, v in self._id_locks.items() if v.get("until", 0) <= now_ts]
                    for k in expired_locks:
                        self._id_locks.pop(k, None)
                if SIM_USE_ROLLING_MAX and self._emb_hist:
                    expired_hist = [k for k, dq in self._emb_hist.items() if not dq or (dq and (now_ts - dq[0][0]) > SIM_ROLL_SECONDS * 1.1)]
                    for k in expired_hist:
                        self._emb_hist.pop(k, None)

                if do_detect:
                    last_det_ts = now_perf
                    t_det0 = perf_counter()
                    try:
                        faces = self.face_app.get(f) if self.face_app else []
                    except Exception as e:
                        logger.error(f"[Face {self.name}] FaceApp.get error: {e}")
                        faces = []
                        last_det_ts = 0
                    t_det_ms = (perf_counter() - t_det0) * 1000.0
                    stats.add_det_cycle(len(faces), t_det_ms)
                    if LOG_PER_DET_DEBUG and len(faces):
                        logger.debug(f"[Face {self.name}] det={len(faces)} | t={t_det_ms:.1f}ms")

                    current_dets = []
                    for face in faces:
                        try:
                            if not hasattr(face, 'bbox'):
                                continue
                            x1, y1, x2, y2 = face.bbox.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
                            if x1 >= x2 or y1 >= y2:
                                continue
                            tk = _assign_track_key(x1, y1, x2, y2)
                            face_crop = f[y1:y2, x1:x2]
                            blur_var = _laplacian_var_gray(face_crop)
                            det_score = float(getattr(face, "det_score", 0.0))
                            kps = getattr(face, "kps", None)
                            yaw_deg = _estimate_yaw_deg_from_kps5(kps)

                            t_emb0 = perf_counter()
                            all_embs = self._embed_face(f, face)
                            stats.add_embed_ms((perf_counter() - t_emb0) * 1000.0)
                            if all_embs.shape[0] == 0:
                                continue

                            if SIM_USE_ROLLING_MAX:
                                self._emb_hist[tk].append((now_ts, all_embs[0]))
                                roll_embs = np.stack([e for (_, e) in self._emb_hist[tk]], axis=0) if self._emb_hist[tk] else all_embs[:1]
                            else:
                                roll_embs = all_embs

                            similarity = 0.0
                            idx = -1
                            raw_similarity = 0.0
                            with self._kn_lock:
                                kn_e, kn_n, kn_d, kn_i = self.known_embs, self.known_names, self.known_depts, self.known_ids
                                if kn_e.shape[0] > 0 and roll_embs.shape[0] > 0:
                                    sims = kn_e @ roll_embs.T
                                    # compute best per-known embedding (max over samples)
                                    best_sim_per_known = sims.max(axis=1)
                                    idx = int(np.argmax(best_sim_per_known))
                                    raw_similarity = float(best_sim_per_known[idx])
                                    if raw_similarity > 0.60:
                                        similarity = min(0.98, raw_similarity + (1.0 - raw_similarity) * 0.45)
                                    else:
                                        similarity = raw_similarity

                            lock = self._id_locks.get(tk)
                            lock_active = bool(lock and lock.get("until", 0) > now_ts)
                            thr_use = _dynamic_sim_threshold(FACE_SIM_THRESH, blur_var, det_score, yaw_deg)

                            if LOG_PER_DET_DEBUG:
                                logger.debug(
                                    f"[Face {self.name}] tk={tk} conf={det_score:.2f} sim={similarity:.4f}({raw_similarity:.4f}) "
                                    f"blur={blur_var:.1f} yaw={yaw_deg:.1f} thr={thr_use:.2f} lock={'Y' if lock_active else 'N'}"
                                )

                            name, dept, emp_id = "Unknown", "Unknown", None
                            is_match = similarity >= thr_use and idx >= 0 and idx < len(self.known_names)

                            if is_match:
                                cand_n, cand_d, cand_i = self.known_names[idx], self.known_depts[idx], self.known_ids[idx]
                                if lock_active:
                                    if lock["name"] == cand_n:
                                        lock["until"] = now_ts + FACE_ID_LOCK_SEC
                                        lock["bbox"] = (x1, y1, x2, y2)
                                        name, dept, emp_id = lock["name"], lock["dept"], lock["emp_id"]
                                    else:
                                        if lock["until"] - now_ts < 0.3:
                                            self._id_locks[tk] = {
                                                "name": cand_n, "dept": cand_d, "emp_id": cand_i,
                                                "until": now_ts + FACE_ID_LOCK_SEC, "bbox": (x1, y1, x2, y2), "since": now_ts
                                            }
                                            name, dept, emp_id = cand_n, cand_d, cand_i
                                        else:
                                            name, dept, emp_id = lock["name"], lock["dept"], lock["emp_id"]
                                            lock["bbox"] = (x1, y1, x2, y2)
                                else:
                                    self._id_locks[tk] = {
                                        "name": cand_n, "dept": cand_d, "emp_id": cand_i,
                                        "until": now_ts + FACE_ID_LOCK_SEC, "bbox": (x1, y1, x2, y2), "since": now_ts
                                    }
                                    name, dept, emp_id = cand_n, cand_d, cand_i

                                logger.info(f"[Face {self.name}] Match: {name}({emp_id}) Sim:{similarity:.3f} Thr:{thr_use:.3f}")
                                stats.add_match(similarity)

                                # AUTO-UPDATE
                                try:
                                    is_high_conf = similarity >= UPDATE_SIM_THRESH
                                    is_off_cool = (now_ts - self._update_cooldown.get(emp_id, 0)) > UPDATE_COOLDOWN_SEC
                                    is_high_qual = (blur_var >= UPDATE_MIN_BLUR_VAR) and (abs(yaw_deg) <= UPDATE_MAX_YAW_DEG)
                                    if is_high_conf and is_off_cool and is_high_qual:
                                        logger.info(f"[AUTO-UPDATE] High quality match for '{name}' (Sim:{similarity:.3f}). Adding.")
                                        best_emb = np.mean(all_embs, axis=0)
                                        if self.db and hasattr(self.db, 'add_face_embedding') and face_crop.size > 0:
                                            ok, buf = cv2.imencode(".jpg", face_crop)
                                            if ok:
                                                saved = self.db.add_face_embedding(emp_id=emp_id, embedding=best_emb,
                                                                                  image_data=buf.tobytes(),
                                                                                  source=f"auto-{similarity:.2f}")
                                                if saved:
                                                    logger.info(f"[AUTO-UPDATE] Added embedding for '{name}'.")
                                                    self._update_cooldown[emp_id] = now_ts
                                                    self.reload_known_faces()
                                                else:
                                                    logger.warning(f"[AUTO-UPDATE] DB add failed for '{name}'.")
                                            else:
                                                logger.warning(f"[AUTO-UPDATE] Encode failed for '{name}'.")
                                        else:
                                            logger.warning(f"[AUTO-UPDATE] DB unavailable/invalid crop for '{name}'.")
                                except Exception as e:
                                    logger.error(f"[AUTO-UPDATE] Error for '{name}': {e}", exc_info=True)

                            else:
                                # No match
                                if lock_active:
                                    name, dept, emp_id = lock["name"], lock["dept"], lock["emp_id"]
                                    lock["until"] = max(lock["until"], now_ts + 1.0)
                                    lock["bbox"] = (x1, y1, x2, y2)
                                else:
                                    name, dept, emp_id = "Unknown", "Unknown", None
                                    stats.add_unknown()

                                if name == "Unknown" and similarity > 0:
                                    logger.info(f"[Face {self.name}] No match (Sim:{similarity:.3f} < Thr:{thr_use:.3f})")

                                if name == "Unknown" and save_face_callback is not None:
                                    # save_face_callback should return (new_name, new_dept) or None
                                    save_info = save_face_callback(f, face.bbox)
                                    if save_info:
                                        new_n, new_d = save_info
                                        if new_n:
                                            if self.save_face(f, face, new_n, new_d or "Unknown"):
                                                self._id_locks[tk] = {
                                                    "name": new_n, "dept": new_d or "Unknown", "emp_id": new_n,
                                                    "until": now_ts + FACE_ID_LOCK_SEC, "bbox": (x1, y1, x2, y2), "since": now_ts
                                                }
                                                name, dept, emp_id = new_n, (new_d or "Unknown"), new_n

                            current_dets.append((x1, y1, x2, y2, name, emp_id, dept,
                                                 (similarity if (is_match or lock_active) else None),
                                                 det_score, blur_var))
                            # DB Logging
                            lk = self._id_locks.get(tk)    
                            # ✅ เงื่อนไขใหม่: log ทันทีถ้า confidence สูง + stable time สั้น
                            time_since_lock = now_ts - lk.get("since", now_ts) if lk else 0
                            stable = (
                                lk and 
                                lk.get("name") == name and 
                                time_since_lock >= FACE_LOG_STABLE_SEC  # 0.05s
                            )
                            
                            # ✅ LOG ทันทีถ้า similarity สูงมาก (> 0.85) แม้ไม่ stable
                            if similarity >= 0.85:
                                stable = True
                            
                            if stable:
                                t_last = self._log_cooldown.get(emp_id, 0.0)
                                cooldown_ok = (now_ts - t_last) >= FACE_MIN_LOG_INTERVAL  # 1.5s
                                
                                if similarity >= 0.75 and cooldown_ok:
                                    if self.db and hasattr(self.db, 'log_face_detection'):
                                        try:
                                            self.db.log_face_detection(
                                                self.name, det_score, (x1, y1, x2, y2),
                                                name, dept, emp_id,
                                                similarity=similarity
                                            )
                                            logger.info(f"[DB Log✅] {name}({emp_id}) Sim:{similarity:.3f}")
                                            self._log_cooldown[emp_id] = now_ts
                                        except Exception as log_e:
                                            logger.error(f"[DB Log❌] {log_e}")
                                            self._safe_log_error("log_face_detection", str(log_e), self.name)
                                elif similarity >= 0.75:
                                    logger.debug(f"[DB Skip] {name}({emp_id}) cooldown={now_ts-t_last:.1f}s")
                            else:
                                logger.debug(f"[DB Wait] {name}({emp_id}) stable={time_since_lock:.2f}s")
                        except Exception as face_err:
                            logger.error(f"[Face {self.name}] Error processing face: {face_err}", exc_info=True)
                            continue

                    last_dets = current_dets

                # End detection block

                frame_to_show = f.copy()
                for d in last_dets:
                    self._draw_box(frame_to_show, d[0:4], d[4], d[7], d[5], d[6], d[8], d[9])

                f_disp = _preprocess_for_preview(frame_to_show)
                f_show = utils.resize_letterbox(f_disp, self.preview_w, self.preview_h)
                f_show = utils.ensure_even(f_show)

                dets_json = []
                scale = min(self.preview_w / frame_w, self.preview_h / frame_h)
                pad_x = (self.preview_w - frame_w * scale) / 2
                pad_y = (self.preview_h - frame_h * scale) / 2
                for (x1, y1, x2, y2, name, emp_id, dept, sim, det, blur) in last_dets:
                    line1 = " | ".join(filter(None, [name, emp_id, dept])) or "Unknown"
                    line2 = _fmt_metrics(det, sim, blur, METRICS_DECIMALS) if SHOW_FACE_METRICS else ""
                    lab = f"{line1} | {line2}" if line2 else line1
                    color = (0, 255, 0) if name != "Unknown" else (0, 200, 255)
                    x1s = int(x1 * scale + pad_x)
                    y1s = int(y1 * scale + pad_y)
                    x2s = int(x2 * scale + pad_x)
                    y2s = int(y2 * scale + pad_y)
                    dets_json.append({"box": (x1s, y1s, x2s, y2s), "label": lab, "color": color})

                try:
                    frame_queue.put_nowait((self.name, f_show, dets_json, f))
                except queue.Full:
                    stats.dropped_frames += 1
                    try:
                        _ = frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait((self.name, f_show, dets_json, f))
                    except queue.Full:
                        pass

                stats.add_loop_time((perf_counter() - loop_t0) * 1000.0)
                stats.maybe_report(self.name)

            except Exception as loop_err:
                logger.error(f"[Face {self.name}] Unhandled loop error: {loop_err}", exc_info=True)
                status_queue.put((self.name, "Loop Error"))
                self._safe_log_error("face_worker_loop", str(loop_err), self.name)
                time.sleep(1.0)

        # --- Worker Exit ---
        self._stop_event.set()
        logger.info(f"[Face {self.name}] Worker stopping.")
        try:
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            while not q_frames.empty():
                q_frames.get_nowait()
        except Exception:
            pass
        status_queue.put((self.name, "หยุดทำงาน"))
        logger.info(f"[Face {self.name}] Worker stopped.")

    def stop(self):
        logger.info(f"[Face {self.name}] Stop requested.")
        self.running = False
        self._stop_event.set()
        try:
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=2.0)
        except Exception as e:
            logger.warning(f"[Face {self.name}] Error joining stream thread: {e}")



class FaceCameraWorkerMultiStream:
    def __init__(self, id, name, camera_url, sub_camera_url=None, zone=None, enter=None, department=None, user_access=None):
        self.id = id
        self.name = str(name)
        self.camera_url_main = camera_url
        self.camera_url_sub = sub_camera_url
        self.zone = zone or 'face'
        self.enter = enter or 'IN'
        self.department = department
        self.user_access = user_access or []
        self.running = False
        self._stop_event = threading.Event()
        self._q_main = queue.Queue(maxsize=8)
        self._q_sub = queue.Queue(maxsize=8)
        self._stream_thread_main = None
        self._stream_thread_sub = None
        try:
            self.db = Database()
        except Exception:
            self.db = None
        self.preview_mode = 'main'

    def set_preview_mode(self, mode):
        if mode not in ('main','sub'):
            raise ValueError('preview must be "main" or "sub"')
        self.preview_mode = mode

    def run(self, frame_queue, log_queue, status_queue, save_face_callback=None):
        if self.db is None:
            status_queue.put((self.name, 'Init Fail'))
            return
        self.running = True
        self._stop_event.clear()
        # spawn frame readers (uses utils.read_frames_stable from original file)
        self._stream_thread_main = threading.Thread(target=utils.read_frames_stable, args=(self.camera_url_main, self._stop_event, self._q_main), daemon=True)
        self._stream_thread_main.start()
        if self.camera_url_sub:
            self._stream_thread_sub = threading.Thread(target=utils.read_frames_stable, args=(self.camera_url_sub, self._stop_event, self._q_sub), daemon=True)
            self._stream_thread_sub.start()
        status_queue.put((self.name, 'เชื่อมต่อแล้ว'))
        logger.info(f'[Face {self.name}] Multi-stream worker started')
        while self.running and not self._stop_event.is_set():
            f_main = None
            f_sub = None
            try:
                try:
                    f_main = self._q_main.get(timeout=0.5)
                    while not self._q_main.empty():
                        f_main = self._q_main.get_nowait()
                except Exception:
                    pass
                if self.camera_url_sub:
                    try:
                        f_sub = self._q_sub.get(timeout=0.1)
                        while not self._q_sub.empty():
                            f_sub = self._q_sub.get_nowait()
                    except Exception:
                        pass
                # prefer preview_mode for which frame to emit first
                stream_used = None
                frame_to_show = None
                if self.preview_mode == 'main' and isinstance(f_main, np.ndarray):
                    stream_used = f'{self.name}-main'
                    frame_to_show = f_main
                elif self.preview_mode == 'sub' and isinstance(f_sub, np.ndarray):
                    stream_used = f'{self.name}-sub'
                    frame_to_show = f_sub
                else:
                    # fallback to any available
                    if isinstance(f_main, np.ndarray):
                        stream_used = f'{self.name}-main'
                        frame_to_show = f_main
                    elif isinstance(f_sub, np.ndarray):
                        stream_used = f'{self.name}-sub'
                        frame_to_show = f_sub
                if frame_to_show is None:
                    continue
                # simple emit: draw nothing heavy here, let consumer handle drawing using detection endpoints
                try:
                    frame_disp = utils.resize_letterbox(frame_to_show, PREVIEW_WIDTH, PREVIEW_HEIGHT)
                except Exception:
                    frame_disp = frame_to_show
                dets_json = []
                try:
                    frame_queue.put_nowait((stream_used, frame_disp, dets_json, frame_to_show))
                except Exception:
                    try:
                        _ = frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        frame_queue.put_nowait((stream_used, frame_disp, dets_json, frame_to_show))
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f'[Face {self.name}] Multi loop error: {e}')
                time.sleep(0.5)
        self._stop_event.set()
        logger.info(f'[Face {self.name}] Multi-stream worker stopped')
