import os
import time
import threading
import subprocess
import shutil
from typing import Optional, Tuple, Deque
from collections import deque
import cv2
import numpy as np
import psutil  # เผื่อใช้ในอนาคต (เช็คโหลดเครื่อง)

# ================== ENV / CONFIG ==================
RECORD_WIDTH   = int(os.getenv("RECORD_WIDTH",  "1920"))
RECORD_HEIGHT  = int(os.getenv("RECORD_HEIGHT", "1080"))
RECORD_FPS     = float(os.getenv("RECORD_FPS",  "25.0"))
RECORD_BITRATE = os.getenv("RECORD_BITRATE", "6000k")
RECORD_CRF     = os.getenv("RECORD_CRF", "").strip()
RECORD_MAXRATE = os.getenv("RECORD_MAXRATE", "").strip()
RECORD_BUFSIZE = os.getenv("RECORD_BUFSIZE", "").strip()
RECORD_CODEC   = os.getenv("RECORD_CODEC", "hevc_nvenc")  # hevc_nvenc | libx265 | hevc_qsv ...
RECORD_GOP     = int(os.getenv("RECORD_GOP", "0"))        # 0 = 2*FPS
RECORD_PRESET  = os.getenv("RECORD_PRESET", "veryfast")
USE_PART_FILE  = True

RECORD_STALE_MS  = int(os.getenv("RECORD_STALE_MS", "2000"))
RECORD_MODE      = os.getenv("RECORD_MODE", "overlay")
RECORD_SAFE      = os.getenv("RECORD_SAFE", "mp4")
MAX_AGE_DAYS     = int(os.getenv("MAX_AGE_DAYS", "0"))  # 0 = ไม่ลบอัตโนมัติ

# คิวเฟรมภายใน (กัน block) และการ drop สูงสุด/รอบ
Q_MAX            = int(os.getenv("REC_Q_MAX", "32"))
Q_DROP_BURST     = int(os.getenv("REC_Q_DROP_BURST", "4"))

def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

def _safe_replace(src: str, dst: str, attempts: int = 60) -> bool:
    """os.replace() แบบ retry (กันไฟล์กำลังถูกแตะ)"""
    delay = 0.05
    for _ in range(attempts):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            time.sleep(delay)
            delay = min(delay * 1.6, 1.5)
        except FileNotFoundError:
            return False
        except OSError:
            time.sleep(delay)
            delay = min(delay * 1.6, 1.5)
    return False

# =============== Letterbox helper (คงสัดส่วนภาพ) ===============
def _resize_letterbox_bgr(img: np.ndarray, target_w: int, target_h: int, pad_color=(0,0,0)) -> np.ndarray:
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
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

# ================== FFmpeg Pipe ==================
class _FFmpegPipeWriter:
    def __init__(self, out_path: str, w: int, h: int, fps: float):
        self.path = out_path
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        self.p = None
        self._opened = False
        self._start(out_path)

    def _start(self, out_path: str):
        ff = _ffmpeg_bin()
        gop = RECORD_GOP if RECORD_GOP > 0 else max(1, int(round(self.fps)))

        def _normalize_bitrate(v: str) -> str:
            v = (v or "").strip()
            if not v:
                return "4000k"
            if v[-1].lower() in ("k", "m"):
                return v
            return f"{v}k"

        def _rc_args_for(_codec_name: str):
            args = []
            if RECORD_CRF:
                args += ["-crf", str(RECORD_CRF)]
                if RECORD_MAXRATE:
                    args += ["-maxrate", RECORD_MAXRATE]
                if RECORD_BUFSIZE:
                    args += ["-bufsize", RECORD_BUFSIZE]
            else:
                bv = _normalize_bitrate(RECORD_BITRATE)
                args += ["-b:v", bv]
                if RECORD_MAXRATE:
                    args += ["-maxrate", RECORD_MAXRATE]
                if RECORD_BUFSIZE:
                    args += ["-bufsize", RECORD_BUFSIZE]
            return args

        codec = (RECORD_CODEC or "libx265").lower()
        if codec in ("libx265", "h265"):
            enc_args = ["-c:v", "libx265", "-preset", RECORD_PRESET, "-tune", "zerolatency",
                        "-g", str(gop), "-bf", "0", *_rc_args_for("libx265"), "-pix_fmt", "yuv420p"]
        elif codec == "hevc_qsv":
            enc_args = ["-c:v", "hevc_qsv", "-preset", "veryfast", "-look_ahead", "0",
                        "-g", str(gop), "-bf", "0", *_rc_args_for("hevc_qsv"), "-pix_fmt", "nv12"]
        elif codec in ("h265_qsv", "avc_qsv"):
            enc_args = ["-c:v", "h265_qsv", "-preset", "veryfast", "-look_ahead", "0",
                        "-g", str(gop), "-bf", "0", *_rc_args_for("h265_qsv"), "-pix_fmt", "nv12"]
        elif codec in ("hevc_nvenc", "h265_nvenc"):
            enc_args = ["-c:v", codec, "-preset", "p4", "-g", str(gop), "-bf", "0",
                        *_rc_args_for(codec), "-pix_fmt", "yuv420p"]
        else:
            enc_args = ["-c:v", "libx265", "-preset", RECORD_PRESET, "-tune", "zerolatency",
                        "-g", str(gop), "-bf", "0", *_rc_args_for("libx265"), "-pix_fmt", "yuv420p"]

        args = [
            ff, "-hide_banner", "-loglevel", os.getenv("FFMPEG_LOGLEVEL", "warning"),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
            "-r", f"{self.fps}", "-i", "-",
            "-an",
            *enc_args,
            "-movflags", "+faststart",
            "-f", "mp4", out_path
        ]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened and self.p and self.p.stdin

    def write(self, frame: np.ndarray):
        try:
            self.p.stdin.write(frame.tobytes())
        except Exception:
            self._opened = False

    def release(self) -> bool:
        ok = True
        try:
            if self.p and self.p.stdin:
                try:
                    self.p.stdin.flush()
                except Exception:
                    pass
                try:
                    self.p.stdin.close()
                except Exception:
                    pass
        except Exception:
            ok = False
        if self.p:
            deadline = time.time() + 30.0
            while time.time() < deadline:
                rc = self.p.poll()
                if rc is not None:
                    break
                time.sleep(0.05)
            if self.p.poll() is None:
                try:
                    self.p.terminate()
                except Exception:
                    pass
                try:
                    self.p.wait(timeout=5)
                except Exception:
                    try:
                        self.p.kill()
                        self.p.wait(timeout=2)
                    except Exception:
                        ok = False
        self._opened = False
        return ok

# ================== Recorder ==================
class VideoRecorder:
    """
    โครงสร้างไฟล์:
        <output_dir>/<department>/<zone>/<camera>/<YYYY-MM-DD>/<camera>_<YYYYmmdd_HHMMSS>.mp4
    """
    def __init__(
        self,
        output_dir: str = "recordings",
        department: str = "Unknown",
        zone: str = "face",
        segment_minutes: int = 15,
        max_file_size_mb: Optional[float] = None,
        camera_id: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.department = (department or "Unknown").strip() or "Unknown"
        self.zone = (zone or "face").strip()
        self.segment_minutes = max(1, int(segment_minutes))
        self.segment_seconds = self.segment_minutes * 60.0
        self.max_file_size_mb = max_file_size_mb

        self._fps = float(RECORD_FPS)
        self._target_w = int(RECORD_WIDTH)
        self._target_h = int(RECORD_HEIGHT)
        self._stale_ms_threshold = int(RECORD_STALE_MS)

        self._lock = threading.Lock()
        self._running = False
        self._pump: Optional[threading.Thread] = None
        self._camera_id = (camera_id or "Unknown").strip()
        self._w: Optional[_FFmpegPipeWriter] = None
        self._cur_tmp: Optional[str] = None
        self._cur_final: Optional[str] = None
        self._opened_ts = 0.0
        self._frames_written = 0

        # คิวเฟรมและตัวจับเวลาคงอัตราเฟรม
        self._q: Deque[np.ndarray] = deque(maxlen=Q_MAX)
        self._last_ts_ms: Optional[int] = None

        # ติดตามวันที่ปัจจุบันเพื่อสร้างโฟลเดอร์รายวัน
        self._current_day = self._today_str()

        os.makedirs(self._base_dir(), exist_ok=True)

        try:
            cv2.setNumThreads(0)
        except Exception:
            pass
        
    def update_segment_minutes(self, new_minutes: int):
            """
            (Thread-safe) อัปเดตระยะเวลาของไฟล์วิดีโอ (segment)
            """
            new_minutes = max(1, int(new_minutes))
            with self._lock:
                # อัปเดตเฉพาะเมื่อค่ามีการเปลี่ยนแปลง
                if new_minutes != self.segment_minutes:
                    print(f"[REC {self._camera_id}] Segment time updated: {self.segment_minutes} -> {new_minutes} mins")
                    self.segment_minutes = new_minutes
                    self.segment_seconds = new_minutes * 60.0
                    # การเปลี่ยนแปลงนี้จะมีผลในรอบการตัดไฟล์ครั้งต่อไป
    # ---------- Path helpers ----------
    def _today_str(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime())

    def _base_dir(self) -> str:
        # โฟลเดอร์รายวัน
        day_dir = os.path.join(self.output_dir, self.department, self.zone, self._camera_id, self._current_day)
        os.makedirs(day_dir, exist_ok=True)
        return day_dir

    @staticmethod
    def _ts_name() -> str:
        return time.strftime("%d%m%Y_%H%M%S", time.localtime())

    def _mk_paths(self) -> Tuple[str, str]:
        name = f"{self._camera_id}_{self._ts_name()}.mp4"
        final = os.path.join(self._base_dir(), name)
        tmp = final + ".part" if USE_PART_FILE else final
        return tmp, final

    # ---------- Public API ----------
    def is_recording(self) -> bool:
        return self._running

    def start_recording(self, camera_id: Optional[str] = None, department: Optional[str] = None, zone: Optional[str] = None):
        with self._lock:
            if self._running:
                return
            if camera_id:
                self._camera_id = str(camera_id)
            if department:
                self.department = (department or "Unknown").strip() or "Unknown"
            if zone:
                self.zone = (zone or "face").strip()

            self._running = True
            self._pump = threading.Thread(target=self._pump_loop, daemon=True)
            self._pump.start()
            print(f"[REC {self._camera_id}] START {self.department}/{self.zone} | CFR {self._fps:.1f} fps | seg={self.segment_minutes} min | codec={RECORD_CODEC}")

    def stop_recording(self):
        with self._lock:
            self._running = False
        if self._pump:
            try:
                self._pump.join(timeout=2.5)
            except Exception:
                pass
            self._pump = None
        self._close_writer()
        print(f"[REC {self._camera_id}] STOP")

    def write_frame(self, frame: np.ndarray, detections=None):
        """รับเฟรมจาก worker แล้ว push เข้า internal queue (non-blocking)"""
        if frame is None:
            return
        self._last_ts_ms = int(time.time() * 1000)
        # ถ้าคิวเต็ม ให้ดรอปแบบ burst เพื่อลด latency
        if len(self._q) >= self._q.maxlen - 1:
            for _ in range(min(Q_DROP_BURST, len(self._q))):
                try:
                    self._q.popleft()
                except IndexError:
                    break
        self._q.append(frame)

    # ---------- Internal loop ----------
    def _pump_loop(self):
        # ใช้ perf_counter เพื่อ timing ที่นิ่งกว่า
        fps = self._fps if self._fps > 0 else float(os.getenv("RECORD_FPS", "25"))
        period = 1.0 / fps
        next_t = time.perf_counter()

        while self._running:
            # เปลี่ยนวัน/โฟลเดอร์อัตโนมัติเมื่อข้ามวัน
            day_now = self._today_str()
            if day_now != self._current_day:
                self._current_day = day_now
                # ปิดไฟล์เดิมแล้วเปิดใหม่ในโฟลเดอร์วันใหม่
                self._close_writer()

            # ensure writer
            if self._w is None:
                W = self._target_w if self._target_w > 0 else None
                H = self._target_h if self._target_h > 0 else None
                if W is None or H is None:
                    # ยังไม่รู้ขนาด → รอเฟรมแรกในคิว
                    if not self._q:
                        time.sleep(0.005)
                        continue
                    h0, w0 = self._q[-1].shape[:2]
                    W, H = w0, h0
                self._open_writer(W, H)

            # จัด CFR: ถ้าช้ากว่ากำหนด ให้ “กระโดดรอบ” เพื่อลด drift
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            # ชดเชยกรณีตกเฟรมหนัก ๆ
            skip = int((now - next_t) // period)
            next_t += (skip + 1) * period

            # เฟรมล่าสุด (ลด latency)
            if not self._q:
                continue
            frame = self._q[-1]

            # stale protection
            if self._last_ts_ms and (int(time.time() * 1000) - self._last_ts_ms) > self._stale_ms_threshold:
                continue

            self._write_raw_bgr(frame)

            # หมุนไฟล์ตามเวลา/ขนาด
            if self._should_rotate():
                self._close_writer()
                self._maybe_cleanup_old()

    # ---------- Writer helpers ----------
    def _open_writer(self, W: int, H: int):
        self._cur_tmp, self._cur_final = self._mk_paths()
        self._w = _FFmpegPipeWriter(self._cur_tmp, int(W), int(H), self._fps)
        self._opened_ts = time.time()
        self._frames_written = 0
        print(f"[REC {self._camera_id}] OPEN {self.department}/{self.zone}/{self._current_day} -> {self._cur_tmp} {W}x{H}@{self._fps:.1f} ({RECORD_CODEC})")

    def _close_writer(self):
        w = self._w
        if not w:
            return
        tmp, final = self._cur_tmp, self._cur_final
        frames = self._frames_written
        ok = False
        try:
            ok = w.release()
        except Exception:
            ok = False
        self._w = None
        self._cur_tmp = self._cur_final = None
        self._opened_ts = 0.0
        self._frames_written = 0
        if not tmp:
            return
        if USE_PART_FILE and tmp != final:
            if _safe_replace(tmp, final):
                print(f"[REC {self._camera_id}] CLOSE -> {final} ({frames} frames)")
            else:
                print(f"[REC {self._camera_id}] CLOSE (rename failed, keep .part) ({frames} frames)")
        else:
            print(f"[REC {self._camera_id}] CLOSE -> {final} ({frames} frames)")

    def _write_raw_bgr(self, frame: np.ndarray):
        if not self._w or not self._w.isOpened():
            return
        Ht, Wt = self._w.h, self._w.w
        try:
            frame_out = _resize_letterbox_bgr(frame, Wt, Ht)
        except Exception:
            return
        try:
            self._w.write(frame_out)
            self._frames_written += 1
        except Exception:
            pass

    def _should_rotate(self) -> bool:
        if self._opened_ts <= 0:
            return False
        if self.segment_seconds and (time.time() - self._opened_ts) >= self.segment_seconds:
            return True
        if self.max_file_size_mb and self._cur_tmp:
            try:
                sz = os.path.getsize(self._cur_tmp) / (1024 * 1024)
                if sz >= self.max_file_size_mb:
                    return True
            except Exception:
                pass
        return False

    # ---------- Housekeeping ----------
    def _maybe_cleanup_old(self):
        """ลบไฟล์เก่ากว่า MAX_AGE_DAYS ภายใต้โฟลเดอร์กล้องปัจจุบัน (ถ้าตั้งค่าไว้)"""
        if MAX_AGE_DAYS <= 0:
            return
        base_cam_dir = os.path.join(self.output_dir, self.department, self.zone, self._camera_id)
        now = time.time()
        try:
            for day in os.listdir(base_cam_dir):
                day_path = os.path.join(base_cam_dir, day)
                if not os.path.isdir(day_path):
                    continue
                # อายุอิงจาก mtime ของโฟลเดอร์
                age_days = (now - os.path.getmtime(day_path)) / (60 * 60 * 24)
                if age_days > MAX_AGE_DAYS:
                    try:
                        shutil.rmtree(day_path, ignore_errors=True)
                        print(f"[REC {self._camera_id}] CLEAN old day folder: {day_path}")
                    except Exception:
                        pass
        except Exception:
            pass
