import os
import time
import threading
import queue
from typing import Optional, Tuple, List

import cv2
import numpy as np
import logging
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import random
import hashlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================
# OpenCV / FFmpeg / RTSP ENV
# ==============================
def ensure_opencv_rtsp_env():
    """
    เซ็ตติ้ง FFmpeg สำหรับ RTSP ที่เน้นเสถียร (ไม่หวัง latency ต่ำสุด)
    """
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    opts = {
        "rtsp_transport": "tcp",           # ★ ใช้ TCP เสมอให้เสถียร
        "rtsp_flags": "prefer_tcp",
        "stimeout": "20000000",            # socket timeout 20s
        "rw_timeout": "20000000",          # บางรุ่นรองรับ (เสริม stimeout)
        "max_delay": "20000000",           # demux delay สูงขึ้นเพื่อกัน jitter
        "buffer_size": "33554432",         # 32MB
        "analyzeduration": "15000000",     # 15s
        "probesize": "15728640",           # 15MB
        "fflags": "nobuffer",              # ★ ไม่บัฟเฟอร์เพิ่ม (ลดเฟรมค้าง)
        "flags": "low_delay",
        "reorder_queue_size": "0",
        "threads": "1",                    # ★ บังคับ 1 เธรด ป้องกันค้างแปลก ๆ
        "sync": "ext",
        # หมายเหตุ: บาง build ไม่รองรับ reconnect สำหรับ RTSP โดยตรง
    }
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(f"{k};{v}" for k, v in opts.items())


# ==============================
# Utils
# ==============================
def _frame_hash(frame: Optional[np.ndarray]) -> Optional[str]:
    if frame is None or not hasattr(frame, "shape"):
        return None
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return None
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()

def even_size(w: int, h: int) -> Tuple[int, int]:
    return (w & ~1), (h & ~1)

def ensure_even(img: np.ndarray) -> np.ndarray:
    if img is None or not hasattr(img, "shape"):
        return img
    h, w = img.shape[:2]
    pad_h = h & 1
    pad_w = w & 1
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img

def resize_letterbox(img: np.ndarray, target_w: int, target_h: int, pad_color=(0, 0, 0)) -> np.ndarray:
    target_w &= ~1
    target_h &= ~1
    if img is None or not hasattr(img, "shape"):
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else (cv2.INTER_CUBIC if scale > 1.5 else cv2.INTER_LINEAR)
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=resized.dtype)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def safe_imencode(ext: str, image: np.ndarray, params: Optional[list] = None) -> Tuple[bool, bytes]:
    params = params or []
    try:
        ok, buf = cv2.imencode(ext, image, params)
        if ok:
            return True, buf.tobytes()
    except Exception:
        pass
    try:
        img2 = ensure_even(np.ascontiguousarray(image))
        ok, buf = cv2.imencode(ext, img2, params)
        if ok:
            return True, buf.tobytes()
    except Exception:
        pass
    return False, b""

def even_pad_bgr(frame):
    h, w = frame.shape[:2]
    pad_bottom = h & 1
    pad_right = w & 1
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    return frame


# ==============================
# Low-level open/read helpers
# ==============================
def _set_capture_props(cap: cv2.VideoCapture, width: Optional[int], height: Optional[int]) -> None:
    try:
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ★ ลดบัฟเฟอร์อ่านให้สั้นที่สุด
    except Exception:
        pass

def open_capture_rtsp(url: str, warmup: int = 6, width: Optional[int] = None, height: Optional[int] = None) -> Optional[cv2.VideoCapture]:
    if not url:
        return None
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"[open_capture_rtsp] Failed to open: {url}")
        return None
    _set_capture_props(cap, width, height)
    # ★ วอร์มอัพเฉพาะ grab() เร็วกว่า read() เพราะไม่ decode
    for _ in range(max(0, int(warmup))):
        cap.grab()
    return cap


# ==============================
# Reader (ฟังก์ชัน)
# ==============================
def read_frames_stable(
    url: str,
    stop_event: threading.Event,
    frame_q: "queue.Queue",
    max_fail: int = 50,
    prefer_sub_url: Optional[str] = None,
    target_fps: Optional[float] = None,
):
    """
    เสถียรสูง: backoff+jitter / stale-detect / adaptive drain / latest-only queue
    """
    try:
        ensure_opencv_rtsp_env()
    except Exception:
        pass

    urls: List[str] = [u for u in (prefer_sub_url, url) if u] or [url]
    idx = 0
    cap = open_capture_rtsp(urls[idx])
    drain_max_env = int(os.getenv("RTSP_DRAIN_MAX", "2"))
    stale_sec = float(os.getenv("RTSP_STALE_SEC", "7"))
    reconnect_max = float(os.getenv("RTSP_RECONNECT_MAX", "10.0"))

    frame_period = 1.0 / float(target_fps) if target_fps and target_fps > 0 else 0.0
    next_ts = 0.0
    last_ok = time.time()
    fail = 0
    reconnect_delay = 0.7
    reconnected = 0

    last_hash = None
    same_hash_cnt = 0
    fps_est = 20.0   # ★ เริ่มด้วยเดา 20fps
    same_hash_limit = int(max(30, fps_est * 2))  # ~2s

    # ★ วัด FPS คร่าว ๆ เพื่อตั้ง hash_limit ให้สัมพันธ์ความจริง
    tick_t = time.time()
    tick_n = 0

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            sleep_for = reconnect_delay + random.uniform(0.0, 0.2 * reconnect_delay)
            time.sleep(sleep_for)
            reconnect_delay = min(reconnect_delay * 1.6, reconnect_max)
            idx = (idx + 1) % len(urls)
            cap = open_capture_rtsp(urls[idx])
            fail = 0
            next_ts = 0.0
            last_ok = time.time()
            last_hash = None
            same_hash_cnt = 0
            reconnected += 1
            logger.info(f"[RTSP] Reconnecting to {urls[idx]} (count={reconnected})")
            continue

        # pace FPS ฝั่งอ่าน
        if frame_period > 0.0:
            now = time.time()
            if next_ts <= 0.0:
                next_ts = now
            if now < next_ts:
                time.sleep(min(0.004, next_ts - now))
                continue

        # ★ ใช้ grab()+retrieve() → สามารถ drain โดยใช้ grab() ไม่ต้อง decode ทิ้งทั้งหมด
        ok = cap.grab()
        if not ok:
            fail += 1
            if (time.time() - last_ok) > stale_sec or fail >= max_fail:
                try: cap.release()
                except Exception: pass
                cap = None
                fail = 0
                logger.warning(f"[RTSP] stale/no frame → reconnect: {urls[idx]}")
            else:
                time.sleep(0.01)
            continue

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            fail += 1
            if (time.time() - last_ok) > stale_sec or fail >= max_fail:
                try: cap.release()
                except Exception: pass
                cap = None
                fail = 0
                logger.warning(f"[RTSP] retrieve failed → reconnect: {urls[idx]}")
            else:
                time.sleep(0.005)
            continue

        last_ok = time.time()
        fail = 0
        reconnect_delay = 0.7

        # ★ อัปเดตประมาณการ FPS → ทำให้ same_hash_limit อิงจริง
        tick_n += 1
        if (last_ok - tick_t) >= 1.5:
            fps_est = max(5.0, min(60.0, tick_n / (last_ok - tick_t)))
            same_hash_limit = int(max(15, fps_est * 2.2))  # ประมาณ ~2.2 วินาที
            tick_t = last_ok
            tick_n = 0

        # ตรวจเฟรมค้างด้วย hash
        hsh = _frame_hash(frame)
        same_hash_cnt = same_hash_cnt + 1 if (hsh and hsh == last_hash) else 0
        last_hash = hsh or last_hash
        if same_hash_cnt >= same_hash_limit:
            try: cap.release()
            except Exception: pass
            cap = None
            same_hash_cnt = 0
            last_hash = None
            logger.warning("[RTSP] repeated identical frames → reconnect")
            continue

        # drain backlog แบบปรับตาม lag
        drained = 0
        drain_max = drain_max_env
        if frame_period > 0.0:
            lag = (time.time() - next_ts) if next_ts > 0 else 0.0
            if lag > (frame_period * 2.5):
                drain_max = max(drain_max_env, 4)  # ★ เพิ่ม drain ถ้าตามไม่ทัน
        while drained < drain_max:
            if not cap.grab():
                break
            drained += 1
        # retrieve เฟรมล่าสุดหลัง drain
        if drained > 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

        frame = ensure_even(frame)

        # latest-only queue
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            try: _ = frame_q.get_nowait()
            except Exception: pass
            try: frame_q.put_nowait(frame)
            except Exception: pass

        if frame_period > 0.0:
            next_ts += frame_period
            if (time.time() - next_ts) > (frame_period * 3):
                next_ts = time.time()


# ==============================
# High-level: RTSPReader class
# ==============================
class RTSPReader:
    """
    เสถียร + auto-switch main/sub เมื่อเปลี่ยนโหมด
    """
    def __init__(
        self,
        main_url: Optional[str],
        sub_url: Optional[str] = None,
        queue_size: int = 1,
        target_fps: Optional[float] = None,
        max_fail: int = 50,
    ):
        self.main_url = main_url or ""
        self.sub_url = sub_url or ""
        self._want_mode = "sub"
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, queue_size))
        self._target_fps = target_fps
        self._max_fail = max_fail

        self._reconnects = 0
        self._last_ok_ts = 0.0

    @property
    def current_mode(self) -> str:
        return self._want_mode

    def start(self, mode: str = "sub"):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._want_mode = "main" if str(mode).lower() == "main" else "sub"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        self._thread = None
        if t and t.is_alive():
            try: t.join(timeout=2.0)
            except Exception: pass
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass

    def switch(self, mode: str):
        self._want_mode = "main" if str(mode).lower() == "main" else "sub"

    def get_latest(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            frame = self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
            while not self._q.empty():
                try: frame = self._q.get_nowait()
                except Exception: break
            return frame
        except Exception:
            return None

    def _loop(self):
        ensure_opencv_rtsp_env()

        current_url = ""
        cap: Optional[cv2.VideoCapture] = None
        last_mode = None

        drain_max_env = max(0, int(os.getenv("RTSP_DRAIN_MAX", "2")))
        stale_sec = float(os.getenv("RTSP_STALE_SEC", "7"))
        reconnect_max = float(os.getenv("RTSP_RECONNECT_MAX", "8.0"))

        frame_period = 1.0 / float(self._target_fps) if self._target_fps and self._target_fps > 0 else 0.0
        next_ts = 0.0

        fail = 0
        reconnect_delay = 0.6

        last_hash = None
        same_hash_cnt = 0
        fps_est = 20.0
        same_hash_limit = int(max(30, fps_est * 2))

        tick_t = time.time()
        tick_n = 0

        def pick_url(mode: str) -> str:
            if mode == "main" and self.main_url:
                return self.main_url
            if mode == "sub" and self.sub_url:
                return self.sub_url
            return self.main_url or self.sub_url

        while not self._stop.is_set():
            want_mode = self._want_mode
            want_url = pick_url(want_mode)

            if want_mode != last_mode or (want_url and want_url != current_url):
                if cap:
                    try: cap.release()
                    except Exception: pass
                    cap = None
                current_url = want_url
                last_mode = want_mode
                fail = 0
                next_ts = 0.0
                self._last_ok_ts = time.time()
                reconnect_delay = 0.6
                last_hash = None
                same_hash_cnt = 0
                if current_url:
                    cap = open_capture_rtsp(current_url)
                    logger.info(f"[RTSPReader] open {want_mode} -> {bool(cap and cap.isOpened())} | {current_url}")

            if cap is None or not cap.isOpened():
                sleep_for = reconnect_delay + random.uniform(0.0, 0.2 * reconnect_delay)
                time.sleep(sleep_for)
                reconnect_delay = min(reconnect_delay * 1.6, reconnect_max)
                cap = open_capture_rtsp(current_url) if current_url else None
                if cap and cap.isOpened():
                    logger.info(f"[RTSPReader] reopened {last_mode}: {current_url}")
                continue

            if frame_period > 0.0:
                now = time.time()
                if next_ts <= 0.0:
                    next_ts = now
                if now < next_ts:
                    time.sleep(min(0.004, next_ts - now))
                    continue

            if not cap.grab():
                fail += 1
                if (time.time() - self._last_ok_ts) > stale_sec or fail >= self._max_fail:
                    try: cap.release()
                    except Exception: pass
                    cap = None
                    logger.warning(f"[RTSPReader] stale/no frame → reconnect ({last_mode}) : {current_url}")
                    continue
                time.sleep(0.01)
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                fail += 1
                if (time.time() - self._last_ok_ts) > stale_sec or fail >= self._max_fail:
                    try: cap.release()
                    except Exception: pass
                    cap = None
                    logger.warning(f"[RTSPReader] retrieve failed → reconnect ({last_mode}) : {current_url}")
                    continue
                time.sleep(0.005)
                continue

            self._last_ok_ts = time.time()
            fail = 0
            reconnect_delay = 0.6

            tick_n += 1
            if (self._last_ok_ts - tick_t) >= 1.5:
                fps_est = max(5.0, min(60.0, tick_n / (self._last_ok_ts - tick_t)))
                same_hash_limit = int(max(15, fps_est * 2.2))
                tick_t = self._last_ok_ts
                tick_n = 0

            hsh = _frame_hash(frame)
            same_hash_cnt = same_hash_cnt + 1 if (hsh and hsh == last_hash) else 0
            last_hash = hsh or last_hash
            if same_hash_cnt >= same_hash_limit:
                try: cap.release()
                except Exception: pass
                cap = None
                same_hash_cnt = 0
                last_hash = None
                logger.warning("[RTSPReader] repeated identical frames → reconnect")
                continue

            drained = 0
            drain_max = drain_max_env
            if frame_period > 0.0:
                lag = (time.time() - next_ts) if next_ts > 0 else 0.0
                if lag > (frame_period * 2.5):
                    drain_max = max(drain_max_env, 4)
            while drained < drain_max:
                if not cap.grab():
                    break
                drained += 1
            if drained > 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue

            try:
                self._q.put_nowait(ensure_even(frame))
            except queue.Full:
                try: _ = self._q.get_nowait()
                except Exception: pass
                try: self._q.put_nowait(ensure_even(frame))
                except Exception: pass

            if frame_period > 0.0:
                next_ts += frame_period
                if (time.time() - next_ts) > (frame_period * 3):
                    next_ts = time.time()


# ==============================
# URL helpers
# ==============================
def infer_rtsp_variants(rtsp_url: str) -> Tuple[str, str]:
    if not rtsp_url:
        return rtsp_url, rtsp_url
    try:
        u = urlparse(rtsp_url)
        qs = dict(parse_qsl(u.query))
        if "subtype" in qs:
            qs_main = qs.copy(); qs_main["subtype"] = "0"
            qs_sub  = qs.copy();  qs_sub["subtype"]  = "1"
            main = urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(qs_main), u.fragment))
            sub  = urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(qs_sub),  u.fragment))
            return main, sub
    except Exception:
        pass
    try:
        if "Streaming/Channels/" in rtsp_url:
            path = urlparse(rtsp_url).path
            if "/Streaming/Channels/" in path:
                tail = path.split("/Streaming/Channels/")[-1]
                if tail and tail.isdigit() and len(tail) == 3:
                    base = rtsp_url[:-3]; ch = tail
                    if ch.endswith("1"): return base + ch, base + ch[:-1] + "2"
                    if ch.endswith("2"): return base + ch[:-1] + "1", base + ch
    except Exception:
        pass
    return rtsp_url, rtsp_url


__all__ = [
    "ensure_opencv_rtsp_env",
    "open_capture_rtsp",
    "read_frames_stable",
    "even_size",
    "ensure_even",
    "resize_letterbox",
    "safe_imencode",
    "RTSPReader",
    "infer_rtsp_variants",
    "even_pad_bgr",
]
