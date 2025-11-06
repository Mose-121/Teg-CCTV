import os
import cv2
import time
import threading
import queue
import numpy as np
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import logging
import re
import warnings
import hashlib
from time import monotonic
from service.database import Database
from service import utils

# +++ NEW IMPORTS FOR OCR + API +++
import pytesseract
import io
from collections import Counter, defaultdict
import requests
from threading import Lock

# ================== NumPy/Warnings ==================
np.seterr(all='warn')
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== ENV / CONFIG ==================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Tesseract Path ---
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if not os.path.exists(TESSERACT_PATH):
    raise FileNotFoundError(f"Tesseract ไม่พบที่: {TESSERACT_PATH}\nกรุณาติดตั้ง Tesseract-OCR และตั้งค่า TESSERACT_PATH")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

MODEL_PLATE = os.getenv("MODEL_PLATE", "").strip()
UI_FONT_PATH = os.getenv("UI_FONT_PATH", "").strip()
CAR_IMGSZ = int(os.getenv("CAR_IMGSZ", "1280"))
CAR_IMGSZ_HI = int(os.getenv("CAR_IMGSZ_HI", "1536"))
CAR_CONF = float(os.getenv("CAR_CONF", "0.25"))
CAR_IOU = float(os.getenv("CAR_IOU", "0.45"))
CAR_DET_INTERVAL = float(os.getenv("CAR_DET_INTERVAL", "0.08"))
OCR_MIN_INTERVAL = float(os.getenv("OCR_MIN_INTERVAL", "180"))
PREVIEW_WIDTH = int(os.getenv("PREVIEW_W", "1280"))
PREVIEW_HEIGHT = int(os.getenv("PREVIEW_H", "720"))
PREVIEW_FROM_CAMERA = os.getenv("PREVIEW_FROM_CAMERA", "1").lower() in ("1", "true", "yes")
PREFER_SUBSTREAM = os.getenv("PREFER_SUBSTREAM", "0").lower() in ("1", "true", "yes")
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "3"))
PREPROC_DET_CLAHE = os.getenv("PREPROC_DET_CLAHE", "true").lower() in ("true", "1", "yes")
PREPROC_DET_CLAHE_CLIP = float(os.getenv("PREPROC_DET_CLAHE_CLIP", "2.5"))
PREPROC_DET_CLAHE_TILE = int(os.getenv("PREPROC_DET_CLAHE_TILE", "8"))
PREPROC_DET_SHARPEN = os.getenv("PREPROC_DET_SHARPEN", "true").lower() in ("true", "1", "yes")
PREPROC_DET_SHARPEN_ALPHA = float(os.getenv("PREPROC_DET_SHARPEN_ALPHA", "1.0"))
CROP_MARGIN = int(os.getenv("CROP_MARGIN", "12"))
OCR_MIN_W = int(os.getenv("OCR_MIN_W", "90"))
OCR_MIN_H = int(os.getenv("OCR_MIN_H", "45"))
SHARP_MIN = float(os.getenv("SHARP_MIN", "70"))

# ---- FIXED/STRIDE-SAFE IMG SIZE CONFIG ----
FIXED_IMGSZ = int(os.getenv("FIXED_IMGSZ", str(CAR_IMGSZ)))
STRIDE = int(os.getenv("MODEL_STRIDE", "32"))

def _make_stride_safe(imgsz: int, stride: int = 32) -> int:
    if imgsz <= 0:
        return stride
    rem = imgsz % stride
    if rem == 0:
        return imgsz
    return imgsz + (stride - rem)

IMG_SZ_TO_USE = _make_stride_safe(FIXED_IMGSZ if FIXED_IMGSZ else CAR_IMGSZ, stride=STRIDE)

# Tracker & Top-K
CAR_TOPK_OCR = int(os.getenv("CAR_TOPK_OCR", "5"))
TRACK_IOU_THRESH = float(os.getenv("TRACK_IOU_THRESH", "0.3"))
TRACK_STALE_SEC = float(os.getenv("TRACK_STALE_SEC", "0.8"))
TRACK_MIN_FRAMES = int(os.getenv("TRACK_MIN_FRAMES", "2"))
TRACK_CLEANUP_SEC = float(os.getenv("TRACK_CLEANUP_SEC", "6.0"))

# Enhanced Accuracy Pack
ENABLE_DESKEW = os.getenv("ENABLE_DESKEW", "1").lower() in ("1", "true", "yes")
MORPH_KERNEL_SIZE = int(os.getenv("MORPH_KERNEL_SIZE", "3"))
EDGE_ENHANCE = os.getenv("EDGE_ENHANCE", "true").lower() in ("true", "1", "yes")
BRIGHTNESS_VARIANTS = int(os.getenv("BRIGHTNESS_VARIANTS", "5"))

# === API OCR CONFIG ===
OCR_API_KEY = os.getenv("OCR_API_KEY", "").strip()
LPR_API_URL = "https://api.aiforthai.in.th/lpr-iapp"
API_TIMEOUT = 10
_api_lock = Lock()
_last_api_call = 0.0
MIN_DELAY_BETWEEN_API = 1.5
MAX_RETRY = 2

# ================== Logging ==================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"IMG_SZ_TO_USE (stride-safe) = {IMG_SZ_TO_USE}, stride={STRIDE}")
logger.info(f"OCR Priority: API → Local | API Key: {'YES' if OCR_API_KEY else 'NO'}")

torch.backends.cudnn.benchmark = True
_OCR_EXEC = ThreadPoolExecutor(max_workers=max(1, OCR_WORKERS))

# ================== Thai Plate Pattern ==================
_THAI2ARABIC = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
CHARS = "ก-ฮ"
DIGITS = r"\d"
re1 = f"[{CHARS}]{{1,3}}[{DIGITS}]{{1,4}}"
re2 = f"[{DIGITS}]{{1}}[{CHARS}]{{2}}[{DIGITS}]{{1,4}}"
_PLATE_RE = re.compile(f"^({re1}|{re2})$", re.IGNORECASE)

def _norm_plate_local(s: str) -> str:
    if not s: return ""
    s = str(s).replace(" ", "").replace("\u200b", "").strip()
    s = s.translate(_THAI2ARABIC).upper()
    return s

def _is_plausible_plate_local(s: str) -> bool:
    return bool(_PLATE_RE.match(s))

# ================== Preprocessing ==================
def _apply_clahe_bgr(img: np.ndarray, clip: float, tile: int) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def _apply_sharpen(img: np.ndarray, alpha: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1 + alpha, blur, -alpha, 0)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    out = img
    if PREPROC_DET_CLAHE:
        out = _apply_clahe_bgr(out, PREPROC_DET_CLAHE_CLIP, PREPROC_DET_CLAHE_TILE)
    if PREPROC_DET_SHARPEN:
        out = _apply_sharpen(out, PREPROC_DET_SHARPEN_ALPHA)
    return out

# ================== Province List ==================
KNOWN_PROVINCES = [
    "กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต", "ขอนแก่น", "ชลบุรี", "นครราชสีมา",
    "สงขลา", "สุราษฎร์ธานี", "นครศรีธรรมราช", "อุบลราชธานี", "พิษณุโลก",
    "เชียงราย", "ลำปาง", "นครปฐม", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ",
    "สมุทรสาคร", "อยุธยา", "สุพรรณบุรี", "ราชบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์",
    "กาญจนบุรี", "ตาก", "สุโขทัย", "อุตรดิตถ์", "แพร่", "น่าน", "พะเยา",
    "ลำพูน", "แม่ฮ่องสอน", "อุดรธานี", "หนองคาย", "เลย", "สกลนคร", "นครพนม",
    "มุกดาหาร", "หนองบัวลำภู", "ศรีสะเกษ", "สุรินทร์", "บุรีรัมย์", "ชัยภูมิ",
    "ยโสธร", "อำนาจเจริญ", "ร้อยเอ็ด", "มหาสารคาม", "กาฬสินธุ์", "เพชรบูรณ์",
    "นครสวรรค์", "อุทัยธานี", "ชัยนาท", "สิงห์บุรี", "อ่างทอง", "ลพบุรี",
    "สระบุรี", "นครนายก", "ปราจีนบุรี", "สระแก้ว", "ฉะเชิงเทรา", "สมุทรสงคราม",
    "ระยอง", "จันทบุรี", "ตราด", "กระบี่", "พังงา", "ระนอง", "ชุมพร",
    "พัทลุง", "ตรัง", "สตูล", "ยะลา", "ปัตตานี", "นราธิวาส"
]

# ================== Enhanced Accuracy Pack ==================
def _deskew_minarect(gray: np.ndarray) -> np.ndarray:
    try:
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return gray
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        if angle < -45: angle += 90
        if abs(angle) < 0.8: return gray
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except: return gray

def _sauvola_threshold(gray: np.ndarray, window_size: int = 17, k: float = 0.25, r: float = 128) -> np.ndarray:
    try:
        padded = cv2.copyMakeBorder(gray, window_size // 2, window_size // 2, window_size // 2, window_size // 2, cv2.BORDER_REPLICATE)
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        mean = cv2.filter2D(padded, -1, kernel)
        mean2 = cv2.filter2D(padded.astype(np.float32)**2, -1, kernel)
        variance = mean2 - mean ** 2
        std = np.sqrt(np.maximum(variance, 0))
        thresh = mean * (1 + k * (std / r - 1))
        return (gray > thresh[window_size//2:-window_size//2, window_size//2:-window_size//2]).astype(np.uint8) * 255
    except: return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

def _apply_morphology(binary: np.ndarray) -> np.ndarray:
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened
    except: return binary

def _enhance_edges(gray: np.ndarray) -> np.ndarray:
    if not EDGE_ENHANCE: return gray
    try:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        sobel = np.absolute(sobel)
        sobel = np.uint8(sobel / sobel.max() * 255) if sobel.max() > 0 else np.uint8(sobel)
        return cv2.addWeighted(gray, 0.6, sobel, 0.4, 0)
    except: return gray

_CHAR_FIX_MAP = str.maketrans({
    'O': '0', 'Q': '0', 'D': '0', 'B': '8', 'I': '1', 'l': '1', 'S': '5', 'Z': '2',
    'G': '6', 'T': '7', 'A': '4', '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
    '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
})

def _postprocess_plate(s: str) -> str:
    s = _norm_plate_local(s)
    s = s.translate(_CHAR_FIX_MAP)
    s = re.sub(r"[^A-Z0-9ก-ฮ]", "", s)
    return s[:8]

def _is_valid_th_plate(s: str) -> bool:
    return _is_plausible_plate_local(s) and len(s) >= 3

def extract_province(text: str) -> Optional[str]:
    if not text: return None
    clean = re.sub(r'[A-Za-z0-9\s]', '', text)
    for prov in KNOWN_PROVINCES:
        if prov in clean:
            return prov
    return None

# ---------------- Adaptive sharp threshold for various crop sizes ----------
def _required_sharp_for_width(w: int) -> float:
    # smaller crops tolerate lower sharpness; larger crops expect stronger sharpness
    if w <= 80:
        return max(8.0, SHARP_MIN * 0.45)
    if w <= 140:
        return max(12.0, SHARP_MIN * 0.65)
    if w <= 240:
        return max(18.0, SHARP_MIN * 0.9)
    if w <= 400:
        return SHARP_MIN
    return SHARP_MIN * 1.15

# ================== PREP: improved variant generator ==================
def _prep_plate_crops_enhanced(img: np.ndarray) -> List[bytes]:
    """
    Enhanced crop pipeline:
      - upscale to canonical height
      - color->gray (V channel), denoise, contrast stretch, CLAHE
      - deskew, bilateral denoise
      - generate threshold variants (adaptive, sauvola, otsu)
      - produce rotated & brightness variants, output PNGs (and a color JPEG)
    """
    try:
        h, w = img.shape[:2]
        if h < 12 or w < 40:
            return []
        target_h = 180
        scale = max(1.0, target_h / float(h))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_REPLICATE)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        v = cv2.fastNlMeansDenoising(v, None, h=15, templateWindowSize=7, searchWindowSize=21)

        lo = np.percentile(v, 2.0)
        hi = np.percentile(v, 98.0)
        if hi - lo > 10:
            v = np.clip((v - lo) * (255.0 / max(1.0, hi - lo)), 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v = clahe.apply(v)

        if ENABLE_DESKEW:
            v = _deskew_minarect(v)

        v = cv2.bilateralFilter(v, d=7, sigmaColor=75, sigmaSpace=7)

        thr1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        thr2 = _sauvola_threshold(v, window_size=19, k=0.3)
        _, thr3 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bases = [thr1, thr2, thr3]
        variants = [_apply_morphology(b) for b in bases]

        brightness_factors = np.linspace(0.85, 1.15, BRIGHTNESS_VARIANTS)
        outs = []

        angles = (-6, -3, 0, 3, 6)
        for base in variants:
            for f in brightness_factors:
                adj = cv2.convertScaleAbs(base, alpha=f, beta=0)
                for ang in angles:
                    if ang == 0:
                        rot = adj
                    else:
                        M = cv2.getRotationMatrix2D((adj.shape[1] // 2, adj.shape[0] // 2), ang, 1.0)
                        rot = cv2.warpAffine(adj, M, (adj.shape[1], adj.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    ok, buf = cv2.imencode(".png", rot, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if ok:
                        outs.append(buf.tobytes())

        # color jpeg (useful for API)
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (nw, nh)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            outs.insert(0, buf.tobytes())

        return outs
    except Exception:
        return []

# ================== OCR BEST: API FIRST + RATE LIMIT + RETRY ==================
def ocr_plate_best(jpeg_bytes_list: List[bytes]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    global _last_api_call

    if OCR_API_KEY and jpeg_bytes_list:
        # prefer largest/most-detailed crop for API
        best_crop = max(jpeg_bytes_list, key=len)
        now = time.time()
        elapsed = now - _last_api_call
        if elapsed < MIN_DELAY_BETWEEN_API:
            time.sleep(MIN_DELAY_BETWEEN_API - elapsed)
        _last_api_call = time.time()

        for attempt in range(MAX_RETRY + 1):
            try:
                with _api_lock:
                    response = requests.post(
                        LPR_API_URL,
                        files=[("file", ("plate.jpg", best_crop, "image/jpeg"))],
                        data={"crop": 1, "rotate": 1},
                        headers={"Apikey": OCR_API_KEY},
                        timeout=API_TIMEOUT
                    )
                if response.status_code == 200:
                    res = response.json()
                    api_plate = _postprocess_plate(res.get("lp_number", ""))
                    api_prov_raw = res.get("province", "")
                    api_prov = api_prov_raw.split(":", 1)[1].strip() if ":" in api_prov_raw else None
                    if _is_valid_th_plate(api_plate):
                        logger.info(f"[OCR] API Success: {api_plate} ({api_prov})")
                        return api_plate, api_prov, None
                elif response.status_code == 429:
                    if attempt < MAX_RETRY:
                        wait = (attempt + 1) * 3
                        logger.warning(f"[OCR] 429 Rate Limit! รอ {wait}s (ครั้งที่ {attempt+1})")
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning("[OCR] 429 เกิน retry → ใช้ Local")
                else:
                    logger.warning(f"[OCR] API Error {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"[OCR] API Exception: {e}")
                if attempt < MAX_RETRY:
                    time.sleep(2)
                    continue
                break

    # Local OCR fallback with voting
    logger.info("[OCR] ใช้ Local OCR")
    try:
        plate_votes = Counter()
        province_votes = defaultdict(Counter)
        whitelist = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for jb in jpeg_bytes_list:
            try:
                img = Image.open(io.BytesIO(jb)).convert("L")
            except Exception:
                continue
            for oem in (3, 1):
                for psm in (7, 8, 6, 3):
                    config = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist} -c load_system_dawg=0 -c load_freq_dawg=0"
                    raw = pytesseract.image_to_string(img, lang='tha+eng', config=config).strip()
                    norm = _postprocess_plate(raw)
                    if _is_valid_th_plate(norm):
                        prov = extract_province(raw)
                        plate_votes[norm] += 1
                        if prov:
                            province_votes[norm][prov] += 1

        if plate_votes:
            local_plate = plate_votes.most_common(1)[0][0]
            local_prov = province_votes[local_plate].most_common(1)[0][0] if province_votes[local_plate] else None
            logger.info(f"[OCR] Local Success: {local_plate} ({local_prov})")
            return local_plate, local_prov, None
    except Exception as e:
        logger.error(f"[OCR] Local Error: {e}")

    return None, None, "OCR failed"

# ================== Drawing ==================
def draw_text_pil(image: np.ndarray, text: str, position: Tuple[int, int], font_path: str,
                  font_size: int = 30, text_color=(255, 255, 255), bg_color=(0, 255, 0)) -> np.ndarray:
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0] + 10
        h = bbox[3] - bbox[1] + 10
        draw.rectangle([position[0], position[1] - h, position[0] + w, position[1]], fill=bg_color)
        draw.text((position[0] + 5, position[1] - h + 5), text, font=font, fill=text_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"Draw error: {e}")
        return image

# ================== Helpers ==================
def _even_dim(x: int) -> int:
    return (x + 1) & ~1

def _clip_box(x1, y1, x2, y2, w, h, margin=2):
    return max(0, int(x1 - margin)), max(0, int(y1 - margin)), min(w, int(x2 + margin)), min(h, int(y2 + margin))

def _score_crop(conf: float, w: int, h: int, sharp: float) -> float:
    r = w / max(1, h)
    ratio_score = 1.0 if 3.0 <= r <= 5.5 else 0.7 if 2.0 <= r <= 6.5 else 0.3
    size_bonus = min(w, 500) / 500.0 * 0.3
    sharp_bonus = min(sharp, 400) / 400.0 * 0.4
    return 0.3 * conf + 0.3 * ratio_score + size_bonus + sharp_bonus

def _calculate_iou(box1, box2):
    # fixed union area computation
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

# ================== Worker Class ==================
class CarCameraWorker:
    def __init__(self, id, name, url, zone, department=None, user_access=None, output_dir="output"):
        self.id = id
        self.name = str(name)
        self.url = url
        self.zone = zone or "car"
        self.department = department
        self.user_access = user_access or []
        self.output_dir = output_dir
        self.running = False
        self.model = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.use_half = torch.cuda.is_available()

        if not MODEL_PLATE or not os.path.exists(MODEL_PLATE):
            raise FileNotFoundError(f"MODEL_PLATE ไม่พบ: {MODEL_PLATE}")
        if not UI_FONT_PATH or not os.path.exists(UI_FONT_PATH):
            raise FileNotFoundError(f"Font ไม่พบ: {UI_FONT_PATH}")

        self._is_pt = MODEL_PLATE.lower().endswith(".pt")
        self._load_model()
        self.db = Database()
        self._plate_seen = {}
        self._stop_event = threading.Event()
        self.preview_w, self.preview_h = PREVIEW_WIDTH, PREVIEW_HEIGHT
        self._raw_wh = None
        self._active_tracks = {}
        self._next_track_id = 0

        os.makedirs(output_dir, exist_ok=True)

    def _load_model(self):
        self.model = YOLO(MODEL_PLATE)
        dev = f"cuda:{self.device}" if isinstance(self.device, int) and torch.cuda.is_available() else "cpu"
        self.device = dev
        if self._is_pt and torch.cuda.is_available():
            try: self.model.fuse()
            except: pass
        logger.info(f"[Car {self.name}] โมเดลโหลดแล้ว | {dev} | half={self.use_half}")

    def refine_detection_on_crop(self, crop_img: np.ndarray, up_scale: float = 2.0, conf_th: float = None, iou_th: float = None):
        conf_th = 0.2 if conf_th is None else conf_th
        iou_th = CAR_IOU if iou_th is None else iou_th
        boxes_out = []
        try:
            h, w = crop_img.shape[:2]
            if h < 10 or w < 30: return boxes_out
            nw, nh = int(w * up_scale), int(h * up_scale)
            big = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
            target_sz = min(max(nw, nh), IMG_SZ_TO_USE, CAR_IMGSZ_HI)
            target_sz = _make_stride_safe(int(target_sz), stride=STRIDE)
            results = self.model(big, verbose=False, device=self.device, conf=conf_th, iou=iou_th,
                                 imgsz=target_sz, half=self.use_half)
            if not results or not hasattr(results[0], "boxes"): return boxes_out
            res = results[0]
            try:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
            except:
                xyxy = res.boxes.xyxy.numpy()
                confs = res.boxes.conf.numpy()
            for bb, cf in zip(xyxy, confs):
                x1 = int(bb[0] / up_scale); y1 = int(bb[1] / up_scale)
                x2 = int(bb[2] / up_scale); y2 = int(bb[3] / up_scale)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes_out.append({"box": (x1, y1, x2, y2), "conf": float(cf)})
        except Exception as e:
            logger.debug(f"Refine error: {e}")
        return boxes_out

    def _draw_box(self, frame, box, label="Detecting", conf=None, color=(0, 255, 0)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return draw_text_pil(frame, label, (x1, y1 - 5), UI_FONT_PATH, 30, (255,255,255), color)

    def run(self, frame_queue, log_queue, status_queue):
        self.running = True
        self._stop_event.clear()
        logger.info(f"[CAR {self.name}] เริ่ม | {self.url}")
        status_queue.put((self.name, "เชื่อมต่อ"))

        q_frames = queue.Queue(maxsize=8)
        sub_url = self.url.replace("subtype=0", "subtype=1") if "subtype=0" in self.url and PREFER_SUBSTREAM else None
        rtsp_thread = threading.Thread(target=utils.read_frames_stable, args=(self.url, self._stop_event, q_frames),
                                       kwargs={"prefer_sub_url": sub_url}, daemon=True)
        rtsp_thread.start()
        time.sleep(1.0)

        frame_count = 0
        last_log = time.time()
        last_det_ts = 0.0

        try: cv2.setNumThreads(0)
        except: pass

        while self.running and not self._stop_event.is_set():
            try:
                frame = q_frames.get(timeout=2.0)
                while not q_frames.empty(): frame = q_frames.get_nowait()
                if frame is None: continue
                frame_count += 1
                try: frame_queue.put((self.name, frame.copy()), timeout=0.05)
                except: pass

                if time.time() - last_log > 10:
                    logger.info(f"[CAR {self.name}] Frame #{frame_count} | Q:{frame_queue.qsize()}")
                    last_log = time.time()

                now = time.perf_counter()
                do_detect = (now - last_det_ts) >= CAR_DET_INTERVAL
                if not do_detect: continue
                last_det_ts = now

                working_frame = frame.copy()
                f_preproc = preprocess_image(working_frame)
                results = self.model(f_preproc, verbose=False, conf=CAR_CONF, iou=CAR_IOU, imgsz=CAR_IMGSZ,
                                     device=self.device, half=self.use_half)
                boxes = results[0].boxes if results else None
                H, W = working_frame.shape[:2]
                new_detections = []

                if boxes is not None and len(boxes) > 0:
                    try: xyxy_all = boxes.xyxy.cpu().numpy(); confs_all = boxes.conf.cpu().numpy()
                    except: xyxy_all = boxes.xyxy.numpy(); confs_all = boxes.conf.numpy()

                    for bb, cf in zip(xyxy_all, confs_all):
                        x1 = int(bb[0]); y1 = int(bb[1]); x2 = int(bb[2]); y2 = int(bb[3])
                        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                        if x2c <= x1c or y2c <= y1c: continue

                        do_refine = (x2c - x1c) < 150 or cf < 0.35
                        refined = self.refine_detection_on_crop(working_frame[y1c:y2c, x1c:x2c]) if do_refine else []
                        if refined:
                            for r in refined:
                                bx1, by1, bx2, by2 = r["box"]
                                new_detections.append({"box": (bx1 + x1c, by1 + y1c, bx2 + x1c, by2 + y1c), "conf": r["conf"], "frame": working_frame})
                        else:
                            new_detections.append({"box": (x1c, y1c, x2c, y2c), "conf": float(cf), "frame": working_frame})

                # === Tracker Logic ===
                matched = set()
                new_tracks = []
                now_t = time.time()

                for det in new_detections:
                    best_iou, best_id = 0.0, -1
                    for tid, tr in self._active_tracks.items():
                        if tid in matched: continue
                        iou = _calculate_iou(det["box"], tr["box"])
                        if iou > TRACK_IOU_THRESH and iou > best_iou:
                            best_iou, best_id = iou, tid
                    if best_id != -1:
                        tr = self._active_tracks[best_id]
                        tr["box"] = det["box"]
                        tr["last_seen"] = now_t
                        if not tr["ocr_started"]:
                            tr["frames_seen"] += 1
                            tr["label"] = f"Tracking... {tr['frames_seen']}"
                            x1e, y1e, x2e, y2e = _clip_box(*det["box"], W, H, CROP_MARGIN)
                            if x2e > x1e and y2e > y1e:
                                crop = det["frame"][y1e:y2e, x1e:x2e]
                                if crop.shape[1] >= OCR_MIN_W and crop.shape[0] >= OCR_MIN_H:
                                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    sharp = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                                    required_sharp = _required_sharp_for_width(crop.shape[1])
                                    if sharp >= required_sharp:
                                        score = _score_crop(det["conf"], crop.shape[1], crop.shape[0], sharp)
                                        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                        if ok: tr["crops"].append((score, buf.tobytes()))
                                    else:
                                        # keep lower-sharp crops but penalize score
                                        score = _score_crop(det["conf"], crop.shape[1], crop.shape[0], sharp) * 0.45
                                        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                        if ok and crop.shape[1] >= (OCR_MIN_W // 2):
                                            tr["crops"].append((score, buf.tobytes()))
                        matched.add(best_id)
                    else:
                        new_tracks.append(det)

                for tid in list(self._active_tracks):
                    tr = self._active_tracks[tid]
                    if tid in matched: continue
                    if tr["ocr_started"]:
                        if now_t - tr["last_seen"] > TRACK_CLEANUP_SEC:
                            self._active_tracks.pop(tid, None)
                    elif now_t - tr["last_seen"] > TRACK_STALE_SEC and tr["frames_seen"] >= TRACK_MIN_FRAMES and tr["crops"]:
                        tr["ocr_started"] = True
                        tr["label"] = "OCR..."
                        tr["color"] = (0, 0, 255)
                        tr["last_seen"] = now_t
                        crops = sorted(tr["crops"], key=lambda x: x[0], reverse=True)[:CAR_TOPK_OCR]
                        jpegs = [c[1] for c in crops]
                        _OCR_EXEC.submit(self._do_ocr_tracked, jpegs, tid, log_queue)

                added_boxes = []
                for det in new_tracks:
                    if any(_calculate_iou(det["box"], b) > 0.8 for b in added_boxes): continue
                    tid = self._next_track_id; self._next_track_id += 1
                    crops = []
                    x1e, y1e, x2e, y2e = _clip_box(*det["box"], W, H, CROP_MARGIN)
                    if x2e > x1e and y2e > y1e:
                        crop = det["frame"][y1e:y2e, x1e:x2e]
                        if crop.shape[1] >= OCR_MIN_W and crop.shape[0] >= OCR_MIN_H:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            sharp = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                            required_sharp = _required_sharp_for_width(crop.shape[1])
                            if sharp >= required_sharp:
                                score = _score_crop(det["conf"], crop.shape[1], crop.shape[0], sharp)
                                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                if ok: crops.append((score, buf.tobytes()))
                            else:
                                score = _score_crop(det["conf"], crop.shape[1], crop.shape[0], sharp) * 0.45
                                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if ok and crop.shape[1] >= (OCR_MIN_W // 2):
                                    crops.append((score, buf.tobytes()))
                    self._active_tracks[tid] = {
                        "box": det["box"], "last_seen": now_t, "frames_seen": 1, "crops": crops,
                        "ocr_started": False, "label": "Tracking... 1", "color": (0, 200, 200)
                    }
                    added_boxes.append(det["box"])

                frame_show = frame.copy()
                dets_json = []
                for tid, tr in self._active_tracks.items():
                    frame_show = self._draw_box(frame_show, tr["box"], tr["label"], color=tr["color"])
                    dets_json.append({"box": tr["box"], "label": tr["label"], "color": tr["color"]})

                if PREVIEW_FROM_CAMERA:
                    fh, fw = frame_show.shape[:2]
                    if self._raw_wh != (fw, fh):
                        self._raw_wh = (fw, fh)
                        self.preview_w, self.preview_h = _even_dim(fw), _even_dim(fh)
                frame_resized = cv2.resize(frame_show, (self.preview_w, self.preview_h))
                try: frame_queue.put_nowait((self.name, frame_resized, dets_json))
                except: pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[CAR {self.name}] Error: {e}")
                time.sleep(0.5)

        self.stop()
        status_queue.put((self.name, "หยุด"))
        logger.info(f"[CAR {self.name}] หยุด | Frames: {frame_count}")

    def _do_ocr_tracked(self, jpegs: List[bytes], track_id: int, log_queue):
        try:
            all_variants = []
            for jb in jpegs:
                img = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    all_variants.extend(_prep_plate_crops_enhanced(img))

            if not all_variants:
                return

            plate, prov, err = ocr_plate_best(all_variants)
            if err or not plate:
                # mark as failed briefly
                if track_id in self._active_tracks:
                    self._active_tracks[track_id]["label"] = "OCR fail"
                    self._active_tracks[track_id]["color"] = (60, 60, 60)
                return

            t = monotonic()
            if (t - self._plate_seen.get(plate, 0)) < OCR_MIN_INTERVAL:
                if track_id in self._active_tracks:
                    self._active_tracks[track_id]["label"] = f"{plate} (ซ้ำ)"
                    self._active_tracks[track_id]["color"] = (255, 165, 0)
                return

            self._plate_seen[plate] = t
            self._plate_seen = {k: v for k, v in self._plate_seen.items() if t - v < OCR_MIN_INTERVAL * 2}

            full = plate + (f" {prov}" if prov and prov != "ไม่ทราบ" else "")
            if track_id in self._active_tracks:
                self._active_tracks[track_id]["label"] = full
                self._active_tracks[track_id]["color"] = (0, 255, 0)

            try:
                car = self.db.check_car(plate)
                status = "INTERNAL" if car else "EXTERNAL"
                province = car.get("province", prov) if car else (prov or "ไม่ทราบ")
            except Exception:
                status = "EXTERNAL"
                province = prov or "ไม่ทราบ"

            try:
                self.db.save_record(
                    plate=plate,
                    province=province,
                    camera_name=self.name,
                    status=status,
                    direction_or_status="IN"
                )
            except Exception as e:
                logger.warning(f"DB save_record failed: {e}")

            log_queue.put(f"[LPR] {plate} @Cam{self.name} → {status} (IN)")
            logger.info(f"[LPR] {plate} ({province}) → {status}")

        except Exception as e:
            logger.error(f"OCR Track Error: {e}")
        finally:
            self._active_tracks.pop(track_id, None)

    def stop(self):
        self.running = False
        self._stop_event.set()
        try:
            _OCR_EXEC.shutdown(wait=False)
        except Exception:
            pass
