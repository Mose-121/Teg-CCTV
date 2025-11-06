import sys, os, json, time, threading, asyncio, requests, cv2, numpy as np
os.environ['QT_MULTIMEDIA_PREFERRED_PLUGINS'] = 'windowsmediafoundation'
import csv
import mimetypes
from datetime import datetime, date
from typing import Optional, List, Tuple ,Dict
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import logging
import websockets
import re
import base64
import pytz
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "SERVER_BASE": os.environ.get("SERVER_BASE", "http://192.168.5.48:8000"),#เปลี่ยนเป็น IP เครื่องที่ run server
    "MAX_TILES": 9,
}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
# NEW: Path for assets
ASSETS_DIR = os.path.join(APP_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")


def infer_rtsp_variants(rtsp_url: str) -> Tuple[str, str]:
    """Return (main_url, sub_url) heuristically; fallback to same URL if unknown."""
    if not rtsp_url:
        return rtsp_url, rtsp_url
    try:
        u = urlparse(rtsp_url)
        qs = dict(parse_qsl(u.query))
        path = u.path or ""

        if "realmonitor" in path and "channel" in qs:
            qs_main = dict(qs); qs_main["subtype"] = "0"
            qs_sub  = dict(qs); qs_sub["subtype"]  = "1"
            main = urlunparse((u.scheme, u.netloc, path, u.params, urlencode(qs_main), u.fragment))
            sub  = urlunparse((u.scheme, u.netloc, path, u.params, urlencode(qs_sub),  u.fragment))
            return main, sub

        if "/Streaming/Channels/" in path:
            try:
                ch = path.split("/Streaming/Channels/")[-1]
                if ch.isdigit() and len(ch) == 3:
                    base = rtsp_url[:-3]
                    if ch.endswith("1"):
                        return base + ch, base + ch[:-1] + "2"
                    if ch.endswith("2"):
                        return base + ch[:-1] + "1", base + ch
            except Exception:
                pass

        if "/unicast/" in path and "/s" in path and path.endswith("/live"):
            parts = path.split("/")
            s_idx = [i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit()]
            if s_idx:
                i = s_idx[-1]
                main_p = list(parts); main_p[i] = "s1"
                sub_p  = list(parts); sub_p[i]  = "s2"
                main = urlunparse((u.scheme, u.netloc, "/".join(main_p), u.params, u.query, u.fragment))
                sub  = urlunparse((u.scheme, u.netloc, "/".join(sub_p),  u.params, u.query, u.fragment))
                return main, sub

        if "h264Preview_" in path:
            if path.endswith("_main"):
                return rtsp_url, rtsp_url.replace("_main", "_sub")
            if path.endswith("_sub"):
                return rtsp_url.replace("_sub", "_main"), rtsp_url

        if "axis-media/media.amp" in path:
            qs_main = dict(qs); qs_sub = dict(qs)
            qs_main.setdefault("resolution", "1920x1080")
            qs_sub.setdefault("resolution", "640x360")
            main = urlunparse((u.scheme, u.netloc, path, u.params, urlencode(qs_main), u.fragment))
            sub  = urlunparse((u.scheme, u.netloc, path, u.params, urlencode(qs_sub),  u.fragment))
            return main, sub

    except Exception as e:
        logger.error(f"Error inferring RTSP variants: {e}")
        return rtsp_url, rtsp_url
    return rtsp_url, rtsp_url


def build_rtsp_url(brand: str, host: str, port: str, user: str, pw: str,
                   channel: str, stream: str, profile: str, subtype: str, custom_path: str) -> str:
    """Build brand-specific RTSP url; for Generic use custom_path."""
    auth = f"{user}:{pw}@" if user and pw else (f"{user}@" if user else "")
    hostport = f"{host}:{port}" if port else host
    b = (brand or "").lower()
    ch = channel or "1"
    st = (stream or "").lower()
    pf = (profile or "").lower()
    sb = subtype or "0"

    if b == "dahua":
        return f"rtsp://{auth}{hostport}/cam/realmonitor?channel={ch}&subtype={sb}"
    if b == "hikvision":
        chnum = int(ch) if ch.isdigit() else 1
        path = f"/Streaming/Channels/{chnum}01" if st != "sub" else f"/Streaming/Channels/{chnum}02"
        return f"rtsp://{auth}{hostport}{path}"
    if b == "uniview":
        s = "s1" if st != "sub" else "s2"
        return f"rtsp://{auth}{hostport}/unicast/c{ch}/{s}/live"
    if b == "axis":
        res = "1920x1080" if st != "sub" else "640x360"
        return f"rtsp://{auth}{hostport}/axis-media/media.amp?camera={ch}&videocodec=h264&resolution={res}"
    if b == "reolink":
        suffix = "main" if st != "sub" else "sub"
        idx = int(ch) if ch.isdigit() else 1
        return f"rtsp://{auth}{hostport}/h264Preview_{idx:02d}_{suffix}"
    if b == "ezviz":
        base = "/h264_stream" if pf != "live" else "/live"
        return f"rtsp://{auth}{hostport}{base}"
    if b == "onvif-generic":
        if pf == "live":
            return f"rtsp://{auth}{hostport}/live/ch{ch}/0"
        path = f"/Streaming/Channels/{ch}01" if st != "sub" else f"/Streaming/Channels/{ch}02"
        return f"rtsp://{auth}{hostport}{path}"
    if b == "generic" and custom_path:
        cp = custom_path if custom_path.startswith("/") else f"/{custom_path}"
        return f"rtsp://{auth}{hostport}{cp}"
    return f"rtsp://{auth}{hostport}"

FILENAME_TIME_RE = re.compile(r'_(\d{2})[-_]?(\d{2})[-_]?(\d{2})(?:\.\w+)?$')

def _parse_video_start_time(date_str: str, filename: str) -> Optional[datetime]:
    """
    Parses the start time from a video filename.
    Returns an AWARE datetime object (Asia/Bangkok).
    """
    try:
        bangkok_tz = pytz.timezone('Asia/Bangkok')
        match = FILENAME_TIME_RE.search(filename)
        
        start_dt_obj = None
        if match:
            hh, mm, ss = match.groups()
            start_str = f"{date_str} {hh}:{mm}:{ss}"
            start_dt_obj = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        else:
            # Fallback (เผื่อชื่อไฟล์ไม่มีเวลา)
            start_dt_obj = datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S")
            
        return bangkok_tz.localize(start_dt_obj)
    except Exception as e:
        logger.error(f"Could not parse time from {filename}: {e}")
        return None
class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None
        self.department = ""
        self.access = []
        self.is_admin = False
        self.username = ""

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({"token": self.token, "user": self.username}, f)
        except Exception:
            pass

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return None
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.token = data.get("token")
                self.username = data.get("user") or ""
                return data
        except Exception:
            return None

    def clear_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                os.remove(CONFIG_FILE)
        except Exception:
            pass

    def _headers(self):
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, path, params=None, stream=False):
        url = self.base_url + path
        return requests.get(url, headers=self._headers(), params=params, timeout=(8, 25), stream=stream)

    def _post(self, path, json=None, data=None, files=None, stream=False):
        url = self.base_url + path
        return requests.post(url, headers=self._headers(), json=json, data=data, files=files, timeout=(8, 25), stream=stream)

    def _put(self, path, json=None):
        url = self.base_url + path
        return requests.put(url, headers=self._headers(), json=json, timeout=(8, 25))

    def _delete(self, path):
        url = self.base_url + path
        return requests.delete(url, headers=self._headers(), timeout=(8, 25))

    def login(self, username: str, password: str, remember: bool=False, force: bool=False) -> dict:
        params = {}
        if remember: params["remember"] = "true"
        if force: params["force"] = "true"
        url = self.base_url + "/auth/login"
        r = requests.post(url,
                          headers=self._headers(),
                          json={"username": username, "password": password},
                          params=params or None,
                          timeout=(8, 25))
        if r.status_code != 200:
            if "application/json" in (r.headers.get("content-type") or ""):
                detail = r.json().get("detail")
            else:
                detail = r.text
            raise RuntimeError(f"{r.status_code} {detail or 'Unauthorized'}")
        js = r.json()
        self.token = js.get("access_token")
        self.department = js.get("department", "")
        self.access = js.get("access", [])
        self.is_admin = bool(js.get("is_admin"))
        self.username = username
        return js

    def login_temp(self, username: str, temp_password: str) -> dict:
        r = self._post("/auth/login-temp", json={"username": username, "temp_password": temp_password})
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Temp login failed: {msg}")
        js = r.json()
        self.token = js.get("access_token")
        self.username = username
        self.department, self.access, self.is_admin = "", [], False
        return js

    def change_password(self, new_password: str) -> dict:
        r = self._post("/auth/change-password", json={"new_password": new_password})
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Change password failed: {msg}")
        js = r.json()
        self.token = js.get("access_token")
        self.department = js.get("department", "")
        self.access = js.get("access", [])
        self.is_admin = bool(js.get("is_admin", False))
        return js
    
    def logout(self):
        if not self.token:
            self.clear_config()
            self.department = ""
            self.access = []
            self.is_admin = False
            return
        try:
            self._post("/auth/logout", json={})
        except Exception:
            pass
        finally:
            self.token = None
            self.department = ""
            self.access = []
            self.is_admin = False
            self.clear_config()

    # ---- Cameras & Reports & Recordings & Users ----
    def list_cameras(self) -> List[dict]:
        r = self._get("/cameras"); r.raise_for_status()
        cameras = r.json()
        for c in cameras:
            if "camera_code" not in c:
                c["camera_code"] = c.get("camera_name")
        return cameras

    def get_preview_mode(self, camera_name: str) -> dict:
        r = self._get(f"/cameras/{camera_name}/preview-mode")
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Get preview mode failed: {msg}")
        return r.json()

    def set_preview_mode(self, camera_name: str, mode: str) -> bool:
        r = self._post(f"/cameras/{camera_name}/preview-mode", json={"mode": mode})
        return r.status_code == 200

    def add_camera(self, code: str, name: str, url: str, zone: str, 
                   comp: Optional[str] = None,url2: Optional[str] = None) -> dict:
        body = {
            "camera_name": code or name,
            "url": url,
            "zone": zone,
            "comp": comp or None,
            "url2": url2 or None,
        }
        body = {k: v for k, v in body.items() if v is not None}
        r = self._post("/cameras", json=body)
        if r.status_code not in (200, 201):
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Add camera failed: {msg}")
        return r.json()

    def update_camera(self, code: str, url: Optional[str] = None, zone: Optional[str] = None,
                      comp: Optional[str] = None, url2: Optional[str] = None) -> dict:
        body = {}
        if url is not None: body["url"] = url
        if url2 is not None: body["url2"] = url2
        if zone is not None: body["zone"] = zone
        if comp is not None: body["comp"] = comp
        if not body: return {"ok": True}
        r = self._put(f"/cameras/{code}", json=body)
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Update camera failed: {msg}")
        return r.json()

    def delete_camera(self, code: str) -> dict:
        r = self._delete(f"/cameras/{code}")
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Delete camera failed: {msg}")
        return r.json()

    # --- ⭐️ โค้ด Employee ที่ถูกต้อง (รวม 5 Methods) ⭐️ ---
    
    def list_employees(self) -> List[dict]:
        """ดึงรายชื่อพนักงานทั้งหมด"""
        try:
            r = self._get("/employees")
            r.raise_for_status()
            return r.json() 
        except Exception as e:
            logger.error(f"Failed to list employees: {e}")
            raise RuntimeError(f"Failed to list employees: {e}")

    def get_employee_details(self, emp_id: str) -> dict:
        """ดึงข้อมูลพนักงาน 1 คน (สำหรับหน้า Edit)"""
        try:
            r = self._get(f"/employees/{emp_id}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Failed to get employee details for {emp_id}: {e}")
            raise RuntimeError(f"Failed to get details: {e}")

    def delete_employee_slot(self, emp_id: str, slot_num: int) -> dict:
        """ลบรูปในช่อง (Slot) ที่กำหนด"""
        try:
            r = self._delete(f"/employees/{emp_id}/slot/{slot_num}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Failed to delete slot {slot_num} for {emp_id}: {e}")
            raise RuntimeError(f"Failed to delete slot: {e}")

    def update_employee_info(self, emp_id: str, name: str, department: str) -> dict:
        """อัปเดตเฉพาะ Info (Name/Dept) โดยไม่ส่งไฟล์"""
        try:
            payload = {"name": name, "department": department}
            r = self._put(f"/employees/{emp_id}/info", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Failed to update info for {emp_id}: {e}")
            raise RuntimeError(f"Failed to update info: {e}")

    def update_employee(self, emp_id: str, 
                        name: Optional[str] = None, 
                        department: Optional[str] = None, 
                        image_paths: Optional[List[str]] = None) -> dict:
        """[สำหรับเพิ่มรูป] เพิ่มรูปภาพใหม่ (ทีละ 1 รูป) ไปยังช่องที่ว่าง"""
        if not emp_id:
            raise RuntimeError("Employee ID is required for update.")
        if not image_paths:
             raise RuntimeError("Please select one new image to add.")

        data = {}
        if name is not None:
            data["name"] = name
        if department is not None:
            data["department"] = department
        
        files = []
        path = image_paths[0] # ใช้รูปแรกที่ส่งมา
        if not os.path.exists(path):
            raise RuntimeError(f"Image file not found: {path}")
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f: content = f.read()
        files.append(("files", (os.path.basename(path), content, mime)))
        
        # (เรียก API 'POST .../update' ที่บังคับว่าต้องมีไฟล์)
        r = self._post(f"/employees/{emp_id}/update", data=data, files=files)

        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Update employee failed: {msg}")
        
        return r.json()

    # --- (โค้ด Enroll/Delete เก่า) ---
    def enroll_employee(self, emp_id: str, name: str, department: str, image_paths: List[str]) -> dict:
        if not image_paths:
            raise RuntimeError("No image selected")
        files = []
        for path in image_paths:
            if not os.path.exists(path):
                raise RuntimeError(f"Image file not found: {path}")
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                raise RuntimeError("Please select a valid image file (.jpg, .jpeg, .png)")
            if os.path.getsize(path) / (1024 * 1024) > 10:
                raise RuntimeError("Image file is too large (max 10MB)")
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            with open(path, "rb") as f: content = f.read()
            files.append(("files", (os.path.basename(path), content, mime)))
        data = {"emp_id": emp_id, "name": name, "department": department}
        r = self._post("/employees/enroll", data=data, files=files)
        if r.status_code not in (200, 201):
            msg = r.json().get("detail") if "application/json" in r.headers.get("content-type", "") else r.text
            if r.status_code == 500:
                msg = f"Server error: {msg}. Please check server logs or contact admin."
            elif r.status_code == 400:
                msg = f"Invalid image: {msg}"
            elif r.status_code == 422:
                msg = f"No face found in image: {msg}"
            raise RuntimeError(msg)
        return r.json()

    def delete_employee(self, emp_id: str) -> dict:
        r = self._delete(f"/employees/{emp_id}")
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Delete employee failed: {msg}")
        return r.json()

    # --- (โค้ด User/Reports/Admin) ---
    def register_user(self, username: str, password: str, department: str, access: List[str], is_admin: bool) -> dict:
        body = {
            "user_name": username,
            "pass_user": password,
            "department": department,
            "access": access if isinstance(access, list) else [s.strip() for s in str(access or "").split(",") if s.strip()],
            "is_admin": bool(is_admin),
        }
        r = self._post("/users/register", json=body)
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Register failed: {msg}")
        return r.json()

    def list_reports(self, start=None, end=None, type_=None, department=None, q=None, limit=500) -> dict:
        params = {"limit": limit}
        if start: params["start"] = start
        if end: params["end"] = end
        if type_: params["type"] = type_
        if department is not None: params["department"] = department
        if q: params["q"] = q
        r = self._get("/reports", params=params)
        if r.status_code != 200:
            error_msg = r.json().get("detail") if "application/json" in r.headers.get("content-type", "") else r.text
            raise RuntimeError(error_msg)
        return r.json()

    def get_detections_for_file(self, filename: str, camera: str, zone: str, date: str) -> List[dict]:
        params = {
            "filename": filename,
            "camera": camera,
            "zone": zone,
            "date": date
        }
        try:
            r = self._get("/reports/by-video-file", params=params)
            r.raise_for_status()
            return r.json().get("items", [])
        except Exception as e:
            logger.error(f"Failed to get detections for {filename}: {e}")
            raise RuntimeError(f"API Error: {str(e)}")
            
    # --- ⭐️ [แก้ไข] list_recordings (ลบ Client-side Filter) ⭐️ ---
    def list_recordings(self, camera_name: str, zone: str, date: Optional[str] = None,
                        person_name: Optional[str] = None) -> List[dict]:
        
        params = {} # สร้าง params เสมอ
        
        if camera_name: params["camera"] = camera_name
        if zone: params["zone"] = zone
        if date: params["date"] = date
        if person_name and person_name.strip():
            params["person_name"] = person_name.strip()
            
        try:
            r = self._get("/recordings", params=params); r.raise_for_status()
            items = r.json()
        except Exception as e:
             logger.error(f"list_recordings failed: {e}")
             items = []

        def _build_display_name(it) -> str:
            cam = camera_name or it.get("camera") or ""
            date_raw = it.get("date") or ""
            try:
                dt = datetime.strptime(date_raw, "%Y-%m-%d")
                dmy = dt.strftime("%d%m%Y")
            except Exception:
                dmy = date_raw.replace("-", "")
            fn = (it.get("file") or it.get("filename") or "")
            m = re.search(r'(?<!\d)(\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)', fn)
            if m:
                hms = "".join(m.groups())
            else:
                mod = it.get("modified") or ""
                hms = mod.split("T")[1][:8].replace(":", "") if "T" in mod else ""
            parts = [cam]
            if dmy: parts.append(dmy)
            if hms: parts.append(hms)
            return "_".join(parts)

        out = []
        for it in items: # (วนลูป 'items' ที่ได้จาก Server โดยตรง)
            size_bytes = it.get("size_bytes", 0) or 0
            out.append({
                "file": it.get("file") or it.get("filename"),
                "modified": it.get("modified"),
                "department": it.get("department"),
                "zone": it.get("zone"),
                "camera": it.get("camera"),
                "date": it.get("date"),
                "size_bytes": size_bytes,
                "size_mb": round(float(size_bytes) / (1024*1024), 2),
                "display_name": _build_display_name(it),
            })
        return out
    # --- ⭐️ [สิ้นสุด] ⭐️ ---

    def download_recording(self, department: str, zone: str, camera_name: str, filename: str, save_path: str, date: Optional[str] = None):
        if date:
            path = f"/recordings/{department}/{zone}/{camera_name}/{date}/{filename}"
        else:
            path = f"/recordings/{department}/{zone}/{camera_name}/{filename}"
        r = self._get(path, stream=True)
        if r.status_code == 404 and date:
            r = self._get(f"/recordings/{department}/{zone}/{camera_name}/{filename}", stream=True)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk: f.write(chunk)

    def mjpg_ws_url(self, camera_name: str) -> str:
        u = urlparse(self.base_url)
        scheme = "wss" if u.scheme == "https" else "ws"
        base_netloc = u.netloc
        path = f"/ws/mjpg/{camera_name}"
        qs = urlencode({"token": self.token or ""})
        return urlunparse((scheme, base_netloc, path, "", qs, ""))

    def admin_reset_temp_password(self, username: str, expire_minutes: int = 30, temp_password: Optional[str] = None) -> dict:
        if not username: raise RuntimeError("Username is required")
        body = {"expire_minutes": int(expire_minutes)}
        if temp_password: body["temp_password"] = temp_password
        r = self._post(f"/admin/users/{username}/reset-password-temp", json=body)
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Reset temp password failed: {msg}")
        return r.json()
        
    def set_segment_minutes(self, minutes: int) -> dict:
        if minutes < 1 or minutes > 120:
            raise RuntimeError("Segment time must be between 1 and 120 minutes.")
        r = self._post("/admin/settings/segment", json={"minutes": minutes})
        if r.status_code != 200:
            try: msg = r.json().get("detail")
            except Exception: msg = r.text
            raise RuntimeError(f"Failed to set segment time: {msg}")
        return r.json()

    def get_camera_events(self, start_date: str, end_date: str) -> List[dict]:
        params = {"start": start_date, "end": end_date}
        try:
            r = self._get("/reports/camera-events", params=params)
            r.raise_for_status()
            return r.json().get("items", [])
        except Exception as e:
            logger.error(f"Failed to get camera events: {e}")
            raise RuntimeError(f"Failed to get events: {e}")

# --- ⭐️ [เพิ่มใหม่ 1/3] คลาสสำหรับจัดการ WS ของ UI ⭐️ ---
class UIWebSocketClient(QObject):
    """
    เชื่อมต่อกับ /ws/ui-updates ใน Thread แยก
    และส่ง Signal (status_updated) กลับมาเมื่อมีข้อมูลใหม่
    """
    status_updated = pyqtSignal(dict) # Signal (ส่ง dict)
    connection_lost = pyqtSignal(str) # Signal (ส่ง Error)

    def __init__(self, api: APIClient):
        super().__init__()
        self.api = api
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        self._thread = None
        if t and t.is_alive():
            try: t.join(timeout=1.0)
            except Exception: pass

    def _build_url(self) -> str:
        """สร้าง URL สำหรับ /ws/ui-updates พร้อม Token"""
        u = urlparse(self.api.base_url)
        scheme = "wss" if u.scheme == "https" else "ws"
        path = "/ws/ui-updates"
        qs = urlencode({"token": self.api.token or ""})
        return urlunparse((scheme, u.netloc, path, "", qs, ""))

    def _run_loop(self):
        """(Main loop ของ Thread)"""
        try:
            asyncio.run(self._ws_main())
        except Exception as e:
            logger.error(f"UI WS loop error: {e}")
            self.connection_lost.emit(f"WebSocket loop error: {e}")

    async def _ws_main(self):
        url = self._build_url()
        logger.info(f"Connecting UI WS: {url}")
        
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                logger.info("[UI WS] Connected successfully.")
                self.connection_lost.emit("") # (แจ้งว่าเชื่อมต่อสำเร็จ)
                
                while not self._stop.is_set():
                    try:
                        msg_str = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg_str)
                        
                        # ⭐️ ถ้าได้รับ "health_status" -> ส่ง Signal ⭐️
                        if data.get("type") == "health_status":
                            self.status_updated.emit(data.get("data", {}))
                            
                        # (คุณสามารถเพิ่ม 'type' อื่นๆ ที่นี่ได้ เช่น 'log')
                            
                    except asyncio.TimeoutError:
                        continue # (ปกติ, แค่เช็ค stop event)
                    except Exception as e:
                        logger.warning(f"[UI WS] Message error: {e}")
                        
        except Exception as e:
            logger.error(f"[UI WS] Connection failed: {e}")
            self.connection_lost.emit(f"Connection failed: {e}")
            time.sleep(5) # (รอ 5 วิ ก่อน Thread จะพยายามต่อใหม่)
# --- ⭐️ [สิ้นสุดคลาสใหม่] ⭐️ ---

# ===============================
#   MJPEG over WebSocket Player
# ===============================
class MJPGWebSocketPlayer:
    def __init__(self, api: APIClient, camera_name: str, on_frame_cb):
        self.api = api
        self.camera_name = camera_name
        self.on_frame_cb = on_frame_cb
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        self._thread = None
        if t and t.is_alive():
            try: t.join(timeout=2.0)
            except Exception: pass

    def _run_loop(self):
        try:
            asyncio.run(self._ws_main())
        except Exception as e:
            logger.error(f"MJPG ws loop error: {e}")
            self.on_frame_cb(None, f"WebSocket error: {e}")

    @staticmethod
    def _extract_jpeg(payload: bytes) -> Optional[bytes]:
        try:
            sep = payload.find(b"\r\n\r\n")
            if sep == -1: return None
            header = payload[:sep]
            body = payload[sep+4:]
            clen = None
            for line in header.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    try: clen = int(line.split(b":", 1)[1].strip())
                    except Exception: clen = None
                    break
            if clen is None: return body
            if len(body) < clen: return None
            return body[:clen]
        except Exception:
            return None

    async def _ws_main(self):
        url = self.api.mjpg_ws_url(self.camera_name)
        logger.info(f"Connecting MJPG WS: {url}")
        try:
            async with websockets.connect(url, max_size=None, ping_interval=20, ping_timeout=20) as ws:
                while not self._stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(msg, (bytes, bytearray)):
                        jpeg = self._extract_jpeg(msg)
                        if not jpeg: continue
                        arr = np.frombuffer(jpeg, np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is None: continue
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        self.on_frame_cb(rgb, None)
                    else:
                        try: self.on_frame_cb(None, str(msg))
                        except Exception: pass
        except Exception as e:
            self.on_frame_cb(None, f"WS connect error: {e}")

class DeepBlueLoginDialog(QDialog):
    def __init__(self, api: APIClient, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("CCTV Login")
        self.setFixedSize(420, 480)
        self.setStyleSheet(
            """
            QDialog { background-color: #0d1b2a; color: #E6EDF3; font-family: 'Segoe UI'; font-size: 14px; }
            QLineEdit { border: 1px solid #1b263b; border-radius: 6px; padding: 8px; background-color: #1b263b; color: #E6EDF3; }
            QLineEdit:focus { border: 1px solid #00aaff; background-color: #16222f; }
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0088ee, stop: 1 #0055aa);
                color: #FFFFFF; border-radius: 6px; padding: 8px 12px; font-weight: bold; border: 1px solid #00aaff;
            }
            QPushButton:hover { background-color: #0099ff; }
            QPushButton:pressed { background-color: #0066bb; }
            QCheckBox { color: #E6EDF3; }
            QLabel#titleLabel { font-size:18px; font-weight:bold; color:#F0F6FC; margin-bottom:10px; }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(16)

        logo_label = QLabel()
        if os.path.exists(LOGO_PATH):
            pixmap = QPixmap(LOGO_PATH)
            logo_label.setPixmap(pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("LOGO")
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QLabel("CCTV Secure Login", objectName="titleLabel"); title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.username = QLineEdit(); self.username.setPlaceholderText("Username")
        self.password = QLineEdit(); self.password.setPlaceholderText("Password"); self.password.setEchoMode(QLineEdit.Password)
        self.remember_cb = QCheckBox("Remember me")
        form.addRow("Username:", self.username)
        form.addRow("Password:", self.password)
        form.addRow("", self.remember_cb)
        layout.addLayout(form)

        self.status = QLabel(""); self.status.setAlignment(Qt.AlignCenter); self.status.setStyleSheet("color:#FFB86C;")
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.login_btn = QPushButton("Login")
        self.login_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) 
        layout.addWidget(self.login_btn)
        layout.addSpacing(20)
        self.login_btn.clicked.connect(self.try_login)
        self.username.returnPressed.connect(self.try_login)
        self.password.returnPressed.connect(self.try_login)

        QTimer.singleShot(200, self.auto_login_if_possible)

    def auto_login_if_possible(self):
        if self.api.load_config():
            self.status.setText("🔄 Checking saved session...")
            QTimer.singleShot(300, self._do_auto_login)

    def _do_auto_login(self):
        try:
            r = self.api._get("/cameras")
            if r.status_code == 200:
                self.accept(); return
        except Exception:
            pass
        self.status.setText("⚠️ Session expired. Please login again.")
        self.api.clear_config()

    def try_login(self):
        user = self.username.text().strip()
        pwd  = self.password.text().strip()
        if not user or not pwd:
            self.status.setText("Please enter username and password.")
            return
    
        self.status.setText("Logging in...")
        QApplication.processEvents()
    
        def _open_change_password_and_finish():
            QMessageBox.information(self, "Change Required",
                                    "ระบบตรวจพบว่าเป็นรหัสชั่วคราว กรุณาตั้งรหัสผ่านใหม่")
            dlg = ChangePasswordDialog(self.api, self)
            if dlg.exec_() == QDialog.Accepted:
                if self.remember_cb.isChecked():
                    self.api.save_config()
                self.accept()
            else:
                self.status.setText("ยกเลิกการเปลี่ยนรหัสผ่าน")
    
        try:
            js = self.api.login(user, pwd, remember=self.remember_cb.isChecked(), force=False)
    
            if js.get("must_change") is True:
                _open_change_password_and_finish()
                return
    
            if self.remember_cb.isChecked():
                self.api.save_config()
            self.accept()
            return
    
        except Exception as e:
            msg = str(e)

            try:
                self.api.login_temp(user, pwd)
                _open_change_password_and_finish()
                return
            except Exception:
                pass 
            
            if "already" in msg.lower() or "409" in msg or "logged in" in msg.lower():
                ret = QMessageBox.question(
                    self, "Duplicate Login",
                    "ผู้ใช้นี้กำลังล็อกอินอยู่อีกเครื่อง ต้องการบังคับเข้าใช้งาน (force login) ไหม?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if ret == QMessageBox.Yes:
                    try:
                        js2 = self.api.login(user, pwd, remember=self.remember_cb.isChecked(), force=True)
                        if js2.get("must_change") is True:
                            _open_change_password_and_finish(); return
                        if self.remember_cb.isChecked(): self.api.save_config()
                        self.accept(); return
                    except Exception as e2:
                        QMessageBox.critical(self, "Login Failed", str(e2))
                else:
                    QMessageBox.information(self, "Cancelled", "ยกเลิกการบังคับล็อกอิน")
            else:
                QMessageBox.critical(self, "Login Failed", msg)

            self.status.setText("❌ Login failed.")

class AdminHub(QDialog):
    def __init__(self, api: APIClient, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Admin Tools")
        self.setMinimumSize(520, 500) # ⭐️ (ปรับความสูงเล็กน้อยสำหรับแถวใหม่)

        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel { color: #E6EDF3; }
            QScrollArea, QScrollArea > QWidget, QWidget#qt_scrollarea_viewport { background-color: #0d1b2a; }
            QPushButton {
                background-color: #1c2b3a; border: 1px solid #2a3f54;
                border-radius: 8px; padding: 10px; text-align: left;
            }
            QPushButton:hover { background-color: #2a3f54; border: 1px solid #00aaff; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        title = QLabel("Administrative Tools")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #E6EDF3; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(15)

        def create_admin_button(text, description, icon_enum):
            btn = QPushButton()
            btn.setMinimumHeight(72)
            btn.setIcon(self.style().standardIcon(icon_enum))
            btn.setIconSize(QSize(32, 32))
            row = QHBoxLayout(btn)
            row.setContentsMargins(12, 10, 12, 10)
            row.setSpacing(12)
            icon_label = QLabel()
            icon_label.setPixmap(self.style().standardIcon(icon_enum).pixmap(QSize(32, 32)))
            icon_label.setAttribute(Qt.WA_TranslucentBackground)
            text_container = QWidget()
            text_container.setAttribute(Qt.WA_TranslucentBackground)
            v = QVBoxLayout(text_container)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            title_label = QLabel(text)
            title_label.setStyleSheet("font-size:12pt; font-weight:bold; color:#E6EDF3; background: transparent;")
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color:#B6C2CF; background: transparent;")
            v.addWidget(title_label)
            v.addWidget(desc_label)
            row.addWidget(icon_label, 0, Qt.AlignLeft)
            row.addWidget(text_container, 1, Qt.AlignLeft)
            return btn

        # --- ⭐️ [แก้ไข] สร้างปุ่มให้ถูกต้อง (ลบปุ่มซ้ำ) ⭐️ ---
        self.btn_register = create_admin_button("Register User", "Create a new user account", QStyle.SP_DialogYesButton)
        self.btn_addemp   = create_admin_button("Enroll Employee", "Add employee for face recognition", QStyle.SP_FileIcon)
        self.btn_editemp  = create_admin_button("Edit Employee", "Update name, department, or add photos", QStyle.SP_FileLinkIcon)
        self.btn_delemp   = create_admin_button("Delete Employee", "Remove an employee from the system", QStyle.SP_TrashIcon)
        self.btn_addcam   = create_admin_button("Add Camera", "Register a new camera stream", QStyle.SP_DesktopIcon)
        self.btn_editcam  = create_admin_button("Edit Camera", "Modify existing camera settings", QStyle.SP_FileDialogStart)
        self.btn_delcam   = create_admin_button("Delete Camera", "Remove a camera from the system", QStyle.SP_DialogCancelButton)
        self.btn_temp     = create_admin_button("Temp Password", "Generate a temporary login password", QStyle.SP_MessageBoxWarning)
        
        self.btn_event_log = create_admin_button("Event Log", "View camera status history (OK/DOWN)", QStyle.SP_FileDialogDetailedView)

        grid.addWidget(self.btn_register, 0, 0)
        grid.addWidget(self.btn_temp,     0, 1)
        grid.addWidget(self.btn_addemp,   1, 0)
        grid.addWidget(self.btn_editemp,  1, 1)
        grid.addWidget(self.btn_delemp,   2, 0)
        grid.addWidget(self.btn_addcam,   2, 1)
        grid.addWidget(self.btn_editcam,  3, 0)
        grid.addWidget(self.btn_delcam,   3, 1)
        grid.addWidget(self.btn_event_log, 4, 0)

        layout.addLayout(grid)

        # --- ⭐️ [แก้ไข] เชื่อมต่อ Signals (ลบปุ่มซ้ำ) ⭐️ ---
        self.btn_register.clicked.connect(lambda: RegisterDialog(self.api, self).exec_())
        self.btn_addemp.clicked.connect(lambda: AddEmployeeDialog(self.api, self).exec_())
        self.btn_delemp.clicked.connect(lambda: DeleteEmployeeDialog(self.api, self).exec_())
        self.btn_editemp.clicked.connect(self.open_edit_employee)
        self.btn_addcam.clicked.connect(lambda: AddCameraDialog(self.api, self).exec_())
        self.btn_editcam.clicked.connect(lambda: EditCameraDialog(self.api, self).exec_())
        self.btn_delcam.clicked.connect(lambda: DeleteCameraDialog(self.api, self).exec_())
        self.btn_temp.clicked.connect(lambda: ResetTempPasswordDialog(self.api, self).exec_())
        self.btn_event_log.clicked.connect(lambda: CameraEventLogDialog(self.api, self).exec_())
        # --- ⭐️ [สิ้นสุด] ⭐️ ---

    def open_edit_employee(self):
        # Slot ใหม่สำหรับเปิด Dialog
        dlg = EditEmployeeDialog(self.api, self)
        dlg.exec_()

class ChangePasswordDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Set New Password")
        self.setFixedSize(360, 180)
        f = QFormLayout(self)
        self.new1 = QLineEdit(); self.new1.setEchoMode(QLineEdit.Password)
        self.new2 = QLineEdit(); self.new2.setEchoMode(QLineEdit.Password)
        f.addRow("New password:", self.new1)
        f.addRow("Confirm:", self.new2)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.handle_change); btns.rejected.connect(self.reject)
        f.addRow(btns)

    def handle_change(self):
        p1 = self.new1.text().strip()
        p2 = self.new2.text().strip()
        if not p1 or len(p1) < 6:
            QMessageBox.warning(self, "Invalid", "Password must be at least 6 characters."); return
        if p1 != p2:
            QMessageBox.warning(self, "Mismatch", "Passwords do not match."); return
        try:
            self.api.change_password(p1)
            QMessageBox.information(self, "Success", "Password changed. Logged in.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class RegisterDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Register User")
        self.setFixedSize(420, 420)

        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                padding: 8px;
                background-color: #1b263b;
                color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b;
                color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QGroupBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)

        v = QVBoxLayout(self)
        f = QFormLayout()

        self.username = QLineEdit()
        self.password = QLineEdit(); self.password.setEchoMode(QLineEdit.Password)
        self.department = QComboBox()
        self.department.addItems(["IT", "HR", "Finance", "Security"]) 
        self.access_box = QGroupBox("Access")
        access_layout = QGridLayout(self.access_box)
        self.access_checks = []
        ACCESS_OPTIONS = ["CENTER","CONDO", "EPO", "EQR", "OFFICE1-2&FIN", "R&D", "TETSO", "TEBP", "TEI", "TER"]
        for i, name in enumerate(ACCESS_OPTIONS):
            cb = QCheckBox(name)
            self.access_checks.append(cb)
            access_layout.addWidget(cb, i // 3, i % 3)

        self.is_admin = QCheckBox("Is Admin")

        f.addRow("Username:", self.username)
        f.addRow("Password:", self.password)
        f.addRow("Department:", self.department)
        f.addRow(self.access_box)
        f.addRow("Status:", self.is_admin)
        v.addLayout(f)

        self.btn = QPushButton("Create User"); v.addWidget(self.btn)
        self.msg = QLabel(""); self.msg.setAlignment(Qt.AlignCenter); v.addWidget(self.msg)

        self.btn.clicked.connect(self.handle_submit)

    def _selected_access(self) -> list:
        return [cb.text() for cb in self.access_checks if cb.isChecked()]

    def handle_submit(self):
        username = self.username.text().strip()
        password = self.password.text().strip()
        department = self.department.currentText().strip()
        access_list = self._selected_access()
        is_admin = self.is_admin.isChecked()

        if not username or not password:
            self.msg.setText("Please fill in username and password"); return
        if not department:
            self.msg.setText("Please select a department"); return

        try:
            result = self.api.register_user(username, password, department, access_list, is_admin)
            if result.get("ok"):
                self.msg.setText("User created successfully")
                QMessageBox.information(self, "Success", "User created successfully")
                self.accept()
            else:
                self.msg.setText(f"Failed: {result.get('detail', '')}")
        except Exception as e:
            self.msg.setText(f"Error: {e}")

class AddEmployeeDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Enroll Employee")
        self.setMinimumWidth(500) 
        
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54; border-radius: 6px;
                padding: 8px; background-color: #1b263b; color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b; color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81; color: #FFFFFF;
                border-radius: 6px; padding: 8px 12px; font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
            QPushButton:disabled { background-color: #2a3f54; color: #778899; }
            QPushButton#removeBtn { background-color: #992233; }
            QPushButton#removeBtn:hover { background-color: #bb3344; }
            QListWidget {
                background-color: #1b263b;
                border: 1px solid #2a3f54;
                border-radius: 6px;
                color: #E6EDF3;
            }
        """)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.emp_id = QLineEdit()
        self.name = QLineEdit()
        self.department = QComboBox()
        initial_departments = ["(พิมพ์เพื่อเพิ่ม)", "IT", "HR", "Finance", "Security"]
        self.department.addItems(initial_departments)
        
        self.department.setEditable(True)
        self.department.lineEdit().setPlaceholderText("เลือกหรือพิมพ์แผนกใหม่...")
        
        def handle_text_change(text):
            if text == "(พิมพ์เพื่อเพิ่ม)":
                self.department.lineEdit().setText("")
        self.department.currentTextChanged.connect(handle_text_change)
        
        self.image_paths = []
        
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.IconMode)
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setResizeMode(QListWidget.Adjust)
        self.image_list.setWordWrap(True)
        
        # --- ⭐️ [แก้ไข 2/4] จำกัดความสูงของช่อง Images ⭐️ ---
        self.image_list.setMinimumHeight(130)
        self.image_list.setMaximumHeight(150) # (บังคับไม่ให้ยืด)
        
        # --- ⭐️ [สิ้นสุด] ⭐️ ---
        
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Select Images (Max 5)")
        self.add_btn.clicked.connect(self.choose_images)
        button_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("❌ Remove Selected")
        self.remove_btn.setObjectName("removeBtn")
        self.remove_btn.clicked.connect(self.remove_selected_image)
        button_layout.addWidget(self.remove_btn)
        
        button_layout.addStretch()
        
        form.addRow("Employee ID:", self.emp_id)
        form.addRow("Name:", self.name)
        form.addRow("Department:", self.department)
        form.addRow("Images:", self.image_list)
        form.addRow("", button_layout)
        
        v.addLayout(form)
        
        # --- ⭐️ [แก้ไข 3/4] เพิ่ม Stretch ⭐️ ---
        v.addStretch(1) # (ดันปุ่ม Enroll ลงล่าง)
        
        self.submit_btn = QPushButton("Enroll"); self.submit_btn.clicked.connect(self.submit); v.addWidget(self.submit_btn)
        self.msg = QLabel(""); self.msg.setAlignment(Qt.AlignCenter); v.addWidget(self.msg)

    def choose_images(self):
        current_count = len(self.image_paths)
        if current_count >= 5:
            QMessageBox.warning(self, "Limit Reached", "You can only add a maximum of 5 images.")
            return

        remaining_slots = 5 - current_count
        
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            f"Select Images (Up to {remaining_slots} more)",
            "", 
            "Images (*.jpg *.jpeg *.png)"
        )
        
        if not paths:
            return

        paths_to_add = paths
        if len(paths) > remaining_slots:
            QMessageBox.warning(self, "Limit Exceeded", 
                                f"You selected {len(paths)} images, but only {remaining_slots} slots are available.\n\nOnly the first {remaining_slots} images will be added.")
            paths_to_add = paths[:remaining_slots]
        
        self.image_paths.extend(paths_to_add) 
        
        for path in paths_to_add:
            filename = os.path.basename(path)
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(QSize(100, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon = QIcon(pixmap)
            item = QListWidgetItem(filename)
            item.setIcon(icon)
            self.image_list.addItem(item)
        
        if len(self.image_paths) >= 5:
            self.add_btn.setEnabled(False)
            self.add_btn.setText("Image slots are full (5/5)")

    # --- ⭐️ [แก้ไข 4/4] แก้ไขฟังก์ชันนี้ (เพิ่ม .api) ⭐️ ---
    def remove_selected_image(self):
        current_item = self.image_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select an image from the list to remove.")
            return

        ret = QMessageBox.question(self, "Confirm Remove", 
                                   f"Are you sure you want to remove this image?\n\n{current_item.text()}",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if ret == QMessageBox.Yes:
            row = self.image_list.row(current_item)
            self.image_list.takeItem(row)
            
            if 0 <= row < len(self.image_paths):
                removed_path = self.image_paths.pop(row)
                logger.info(f"Removed image from enroll list: {removed_path}")
            else:
                logger.warning(f"Could not find path at index {row} to remove (List mismatch!).")

            if len(self.image_paths) < 5:
                self.add_btn.setEnabled(True)
                self.add_btn.setText("Select Images (Max 5)")

    def submit(self):
        emp_id = self.emp_id.text().strip()
        name = self.name.text().strip()
        department = self.department.currentText().strip()
        
        image_paths = self.image_paths 
        
        if not (emp_id and name and department and image_paths):
            self.msg.setText("Please fill in all fields"); return
        try:
            self.api.enroll_employee(emp_id, name, department, image_paths) 
            QMessageBox.information(self, "Success", "Employee enrolled successfully")
            self.accept()
        except Exception as e:
            self.msg.setText(f"Error: {e}")
# --- ⭐️ [สิ้นสุดการวางทับ] ⭐️ ---
            
class DeleteEmployeeDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Delete Employee")
        self.setFixedSize(300, 150)
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                padding: 8px;
                background-color: #1b263b;
                color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b;
                color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.emp_id = QLineEdit(); self.emp_id.setPlaceholderText("Employee ID")
        form.addRow("Employee ID:", self.emp_id); v.addLayout(form)
        self.btn = QPushButton("Delete Employee"); self.btn.clicked.connect(self.handle_delete); v.addWidget(self.btn)
        self.msg = QLabel(""); self.msg.setAlignment(Qt.AlignCenter); v.addWidget(self.msg)

    def handle_delete(self):
        emp_id = self.emp_id.text().strip()
        if not emp_id:
            self.msg.setText("Please enter employee ID"); return
        try:
            self.api.delete_employee(emp_id)
            QMessageBox.information(self, "Success", f"Employee {emp_id} deleted successfully")
            self.accept()
        except Exception as e:
            self.msg.setText(f"Error: {e}")

class EditEmployeeDialog(QDialog):
    employeeUpdated = pyqtSignal()

    def __init__(self, api: APIClient, parent=None):
        super().__init__(parent)
        self.api = api
        self.current_emp_id: Optional[str] = None
        self.current_emp_data: Optional[dict] = None
        
        self.setWindowTitle("Edit Employee Information")
        self.setMinimumSize(700, 450) 
        
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54; border-radius: 6px;
                padding: 8px; background-color: #1b263b; color: #E6EDF3;
            }
            /* ⭐️ [เพิ่ม] สไตล์สำหรับ Dropdown ที่ค้นหาได้ ⭐️ */
            QComboBox QAbstractItemView {
                background: #1b263b; color: #E6EDF3; selection-background-color: #2a3f54;
            }
            QComboBox::drop-down { border: none; }
            QLineEdit { selection-background-color: #0f4c81; }
            /* ⭐️ [สิ้นสุด] ⭐️ */
            QGroupBox {
                border: 1px solid #2a3f54; border-radius: 8px; margin-top: 12px;
                font-weight: 600; color: #E6EDF3;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px;
                color: #E6EDF3; background: #0f4c81; border-radius: 4px;
            }
            QPushButton {
                background-color: #0f4c81; color: #FFFFFF;
                border-radius: 6px; padding: 8px 12px; font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
            QPushButton#deleteBtn { background-color: #992233; }
            QPushButton#deleteBtn:hover { background-color: #bb3344; }
            QPushButton:disabled { background-color: #2a3f54; color: #778899; }
            QLabel#slotLabel {
                background-color: #1b263b; border: 1px dashed #2a3f54;
                border-radius: 6px; min-height: 100px; max-height: 100px;
                min-width: 100px; max-width: 100px;
                alignment: AlignCenter;
            }
        """)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # --- 1. Dropdown (NOW SEARCHABLE) ---
        self.emp_select = QComboBox()
        self.emp_select.setEditable(True) # ⭐️ 1. ทำให้พิมพ์ได้
        self.emp_select.lineEdit().setPlaceholderText("Loading employees...") # ⭐️ 2. ตั้ง Placeholder
        self.emp_select.setEnabled(False)
        
        # ⭐️ 3. สร้างระบบค้นหา (Completer)
        self.completer = QCompleter(self)
        self.completer_model = QStringListModel()
        self.completer.setModel(self.completer_model)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.emp_select.setCompleter(self.completer)

        form_layout.addRow("Select Employee:", self.emp_select)
        
        layout.addLayout(form_layout) # ⭐️ 4. (แก้บั๊กหน้าว่าง)

        # --- 2. ช่องสำหรับแก้ไขข้อมูล (เหมือนเดิม) ---
        self.info_groupbox = QGroupBox("Employee Information")
        info_layout = QFormLayout(self.info_groupbox)
        self.emp_id_label = QLineEdit() 
        self.emp_id_label.setReadOnly(True)
        self.name_edit = QLineEdit()
        self.department_edit = QComboBox()
        self.department_edit.setEditable(True)
        initial_departments = ["(พิมพ์เพื่อเพิ่ม)", "IT", "HR", "Finance", "Security"]
        self.department_edit.addItems(initial_departments)
        info_layout.addRow("Employee ID:", self.emp_id_label)
        info_layout.addRow("Full Name:", self.name_edit)
        info_layout.addRow("Department:", self.department_edit)
        layout.addWidget(self.info_groupbox)

        # --- 3. ส่วนสำหรับ "จัดการ" 5 ช่อง (เหมือนเดิม) ---
        self.slots_groupbox = QGroupBox("Image Slots (Max 5)")
        self.slots_layout = QGridLayout(self.slots_groupbox)
        self.slots_layout.setSpacing(10)
        for i in range(1, 6):
            slot_widget = QWidget()
            slot_vbox = QVBoxLayout(slot_widget)
            slot_vbox.setContentsMargins(0,0,0,0)
            img_label = QLabel(f"Slot {i}\n(Empty)")
            img_label.setObjectName("slotLabel")
            setattr(self, f"slot_img_{i}", img_label) 
            slot_vbox.addWidget(img_label)
            add_btn = QPushButton("➕ Add")
            add_btn.clicked.connect(lambda _, s=i: self._on_add_image(s))
            setattr(self, f"slot_add_btn_{i}", add_btn)
            slot_vbox.addWidget(add_btn)
            del_btn = QPushButton("❌ Delete")
            del_btn.setObjectName("deleteBtn")
            del_btn.clicked.connect(lambda _, s=i: self._on_delete_slot(s))
            setattr(self, f"slot_del_btn_{i}", del_btn)
            slot_vbox.addWidget(del_btn)
            self.slots_layout.addWidget(slot_widget, 0, i - 1)
        layout.addWidget(self.slots_groupbox)
        
        # --- 4. ปุ่ม Save (สำหรับ Name/Dept) และ Status ---
        self.submit_btn = QPushButton("Save Info Changes (Name/Dept)")
        self.submit_btn.clicked.connect(self.submit_info_only)
        layout.addWidget(self.submit_btn)
        
        self.msg = QLabel("")
        self.msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.msg)
        layout.addStretch(1)
        
        self.info_groupbox.hide()
        self.slots_groupbox.hide()
        self.submit_btn.hide()

        # --- 6. เชื่อมต่อ Signal และเริ่มโหลด ---
        self.emp_select.activated.connect(self.on_employee_selected) 
        QTimer.singleShot(100, self.load_employees)

    def load_employees(self):
        """[แก้ไข] เรียก API /employees เพื่อดึงรายชื่อมาใส่ ComboBox และ Completer"""
        try:
            employees = self.api.list_employees()
            
            self.emp_select.clear()
            self.completer_model.setStringList([]) # ⭐️ 6. เคลียร์ Model ของระบบค้นหา
            
            self.emp_select.addItem("", None) # (แก้) เพิ่ม Item ว่าง (สำหรับ Placeholder)
            employee_display_list = [] # ⭐️ 7. สร้าง List ว่างสำหรับเก็บชื่อ
            
            for emp in employees:
                emp_name = emp.get("full_name") or "Unnamed" 
                emp_id = emp.get("emp_id")
                display_text = f"{emp_name} ({emp_id})"
                self.emp_select.addItem(display_text, emp_id)
                employee_display_list.append(display_text)

            self.completer_model.setStringList(employee_display_list) # ⭐️ 9. ตั้งค่า Model ให้ระบบค้นหา
            
            self.emp_select.setEnabled(True)
            self.emp_select.lineEdit().setPlaceholderText("Type to search employee...") # ⭐️ 10. เปลี่ยน Placeholder
            self.msg.setText("Please select an employee to edit.")
            
        except Exception as e:
            # (แก้) ถ้า Error ให้ใช้ Placeholder
            self.emp_select.lineEdit().setPlaceholderText("Failed to load employees.") 
            self.msg.setText(f"Error loading: {e}")

    def on_employee_selected(self, index):
        """เมื่อเลือกพนักงาน, ให้ดึงข้อมูล chi tiết (slots)"""
        self.current_emp_id = self.emp_select.itemData(index)
        
        if not self.current_emp_id:
            self.info_groupbox.hide()
            self.slots_groupbox.hide()
            self.submit_btn.hide()
            self.current_emp_data = None
            return
            
        self.info_groupbox.show()
        self.slots_groupbox.show()
        self.submit_btn.show()
        
        self.refresh_employee_details()

    def refresh_employee_details(self):
        """[อัปเกรด 5 ช่อง] ดึงข้อมูล slots และอัปเดต UI"""
        if not self.current_emp_id:
            return
            
        try:
            self.msg.setText("Loading details...")
            QApplication.processEvents()
            
            data = self.api.get_employee_details(self.current_emp_id)
            self.current_emp_data = data
            
            self.emp_id_label.setText(data.get("emp_id", ""))
            self.name_edit.setText(data.get("full_name", ""))
            dept = data.get("department", "")
            if dept:
                idx = self.department_edit.findText(dept)
                if idx == -1:
                    self.department_edit.addItem(dept)
                    self.department_edit.setCurrentText(dept)
                else:
                    self.department_edit.setCurrentIndex(idx)
            else:
                self.department_edit.setCurrentIndex(0)

            slots_data = data.get("slots", {})
            for i in range(1, 6):
                img_label = getattr(self, f"slot_img_{i}")
                add_btn = getattr(self, f"slot_add_btn_{i}")
                del_btn = getattr(self, f"slot_del_btn_{i}")
                img_b64 = slots_data.get(str(i))
                
                if img_b64:
                    try:
                        pixmap = QPixmap()
                        pixmap.loadFromData(base64.b64decode(img_b64), "JPG")
                        img_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        del_btn.show()
                        add_btn.hide()
                    except Exception as e:
                        logger.error(f"Error decoding base64 for slot {i}: {e}")
                        img_label.setText(f"Slot {i}\n(Load Error)")
                        del_btn.show()
                        add_btn.hide()
                else:
                    img_label.setText(f"Slot {i}\n(Empty)")
                    img_label.setPixmap(QPixmap()) 
                    del_btn.hide()
                    add_btn.show()
            
            self.msg.setText("Employee details loaded.")

        except Exception as e:
            self.msg.setText(f"Error loading details: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load details:\n{str(e)}")

    def _on_delete_slot(self, slot_num: int):
        if not self.current_emp_id:
            return
            
        ret = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the image in Slot {slot_num} for {self.current_emp_id}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if ret == QMessageBox.Yes:
            try:
                self.msg.setText(f"Deleting slot {slot_num}...")
                QApplication.processEvents()
                self.api.delete_employee_slot(self.current_emp_id, slot_num)
                self.msg.setText(f"Slot {slot_num} deleted. Refreshing...")
                self.refresh_employee_details()
            except Exception as e:
                self.msg.setText(f"Error deleting slot: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete slot:\n{str(e)}")

    def _on_add_image(self, slot_num: int):
        if not self.current_emp_id:
            return
        
        path, _ = QFileDialog.getOpenFileName(self, "Select 1 New Image to Add", "", "Images (*.jpg *.jpeg *.png)")
        
        if not path:
            return 
            
        try:
            self.msg.setText(f"Adding new image to next available slot...")
            QApplication.processEvents()

            self.api.update_employee(
                emp_id=self.current_emp_id,
                name=self.name_edit.text(), 
                department=self.department_edit.currentText(),
                image_paths=[path]
            )
            
            self.msg.setText(f"New image added. Refreshing...")
            self.refresh_employee_details()
            
        except Exception as e:
            self.msg.setText(f"Error adding image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add new image:\n{str(e)}")

    # --- ⭐️ [วางทับ] ฟังก์ชันนี้ (แก้บั๊กปุ่ม Save) ⭐️ ---
    def submit_info_only(self):
        """
        [แก้ไข] เรียก API ใหม่สำหรับอัปเดต Name/Dept เท่านั้น
        """
        if not self.current_emp_id:
            return
            
        new_name = self.name_edit.text().strip()
        new_dept = self.department_edit.currentText().strip()
        
        if not new_name:
            QMessageBox.warning(self, "Error", "Full Name cannot be empty.")
            return

        if new_dept == "(พิมพ์เพื่อเพิ่ม)":
            new_dept = "" 

        try:
            self.msg.setText("Saving info changes...")
            QApplication.processEvents()
            
            self.api.update_employee_info(
                emp_id=self.current_emp_id,
                name=new_name,
                department=new_dept
            )
            
            self.msg.setText("Information saved.")
            QMessageBox.information(self, "Success", "Employee information updated.")
            
            self.load_employees()
            self.emp_select.setCurrentText(f"{new_name} ({self.current_emp_id})")

        except Exception as e:
            self.msg.setText(f"Error saving info: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save info:\n{str(e)}")
            
class AddCameraDialog(QDialog):
    cameraAdded = pyqtSignal()

    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Add Camera")
        self.setMinimumSize(520, 460)

        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget, QWidget#qt_scrollarea_viewport {
                background-color: #0d1b2a;
            }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54; border-radius: 6px; padding: 8px;
                background-color: #1b263b; color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b; color: #E6EDF3; selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81; color: #FFFFFF; border-radius: 6px;
                padding: 8px 12px; font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        wrap = QWidget()
        v = QVBoxLayout(wrap)
        f = QFormLayout()

        self.name_code = QLineEdit()
        self.name_code.setPlaceholderText("Camera Code / Display Name (e.g., B101-Front / B101 Front Door)")
        
        self.brand = QComboBox(); self.brand.addItems(["Generic", "Dahua", "Hikvision", "Uniview", "Axis", "Reolink", "Ezviz", "ONVIF-Generic"])
        self.host = QLineEdit(); self.host.setPlaceholderText("e.g., 192.168.1.100")
        self.port = QLineEdit(); self.port.setPlaceholderText("554")
        self.user = QLineEdit()
        self.pw = QLineEdit(); self.pw.setEchoMode(QLineEdit.Password)
        self.channel = QLineEdit(); self.channel.setText("1")
        self.custom_path = QLineEdit(); self.custom_path.setPlaceholderText("e.g., /h264_stream or /Streaming/Channels/101")
        self.zone = QComboBox(); self.zone.addItems(["building", "car"])
        self.department = QComboBox()

        initial_departments = [
            "(พิมพ์เพื่อเพิ่ม)", "CENTER", "CONDO", "EQR", "TER", "OFFICE1-2&FIN", 
            "TETSO", "TEI", "EPO", "STORE", "TEBP", "R&D"
        ]
        unique_departments = sorted(list(set(initial_departments) - {"(พิมพ์เพื่อเพิ่ม)"}))
        self.department.addItems(["(พิมพ์เพื่อเพิ่ม)"] + unique_departments)

        self.department.setEditable(True)
        self.department.lineEdit().setPlaceholderText("เลือกหรือพิมพ์แผนกใหม่...")
        def handle_text_change(text):
            if text == "(พิมพ์เพื่อเพิ่ม)":
                self.department.lineEdit().setText("")
        self.department.currentTextChanged.connect(handle_text_change)
        
        self.preview_url = QLineEdit(); self.preview_url.setReadOnly(True)
        self.preview_url2 = QLineEdit(); self.preview_url2.setReadOnly(True)

        self.btn_preview = QPushButton("Preview RTSP URL")
        self.btn_test = QPushButton("Test URL (2 seconds)")
        self.btn_add = QPushButton("Add Camera")

        f.addRow("Camera Code / Display Name:", self.name_code)
        f.addRow("Brand:", self.brand)
        f.addRow("Host/IP:", self.host)
        f.addRow("Port:", self.port)
        f.addRow("User:", self.user)
        f.addRow("Password:", self.pw)
        f.addRow("Channel:", self.channel)
        f.addRow("Custom Path:", self.custom_path)
        f.addRow("Zone:", self.zone)
        f.addRow("Department:", self.department)
        f.addRow("Main RTSP URL:", self.preview_url)
        f.addRow("Sub RTSP URL:", self.preview_url2)

        h = QHBoxLayout()
        h.addWidget(self.btn_preview)
        h.addWidget(self.btn_test)
        f.addRow("", h)

        v.addLayout(f)
        v.addWidget(self.btn_add)

        self.help = QLabel(
            "Tips:\n"
            "- Dahua: channel=1\n"
            "- Hikvision: uses 101 per channel\n"
            "- Uniview: /unicast/c{ch}/s1/live\n"
            "- Axis: uses resolution\n"
            "- Reolink: h264Preview_01_main\n"
            "- Ezviz: some models use /h264_stream or /live\n"
            "- Generic: provide custom path"
        )
        self.help.setStyleSheet("color:#C7D6E2;")
        v.addWidget(self.help)

        scroll.setWidget(wrap)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        self.msg = QLabel("")
        self.msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.msg)

        widgets = [self.brand, self.host, self.port, self.user, self.pw, self.channel, self.custom_path]
        for w in widgets:
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.update_preview_url)
            else:
                w.textChanged.connect(self.update_preview_url)

        self.btn_preview.clicked.connect(self.update_preview_url)
        self.btn_test.clicked.connect(self.test_preview_url)
        self.btn_add.clicked.connect(self.submit)

        self.port.setText("554")
        self.channel.setText("1")
        self.apply_brand_visibility()
        self.brand.currentIndexChanged.connect(self.apply_brand_visibility)
        self.update_preview_url()

    def apply_brand_visibility(self):
        b = (self.brand.currentText() or "").lower()
        self.custom_path.setVisible(b == "generic")

    def update_preview_url(self):
        try:
            url = build_rtsp_url(
                self.brand.currentText(), self.host.text().strip(), self.port.text().strip(),
                self.user.text().strip(), self.pw.text().strip(), self.channel.text().strip(),
                "", "", "", 
                self.custom_path.text().strip()
            )
            main, sub = infer_rtsp_variants(url)
            self.preview_url.setText(main)
            self.preview_url2.setText(sub)
        except Exception as e:
            logger.error(f"Error updating preview URL: {e}")
            self.preview_url.setText("")
            self.preview_url2.setText("")

    def test_preview_url(self):
        url = self.preview_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Test URL", "No RTSP URL provided")
            return
        self.btn_test.setEnabled(False)
        self.btn_test.setText("Testing...")
        QApplication.processEvents()
        ok = False
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            t0 = time.time()
            while time.time() - t0 < 2.0:
                ret, _ = cap.read()
                if ret:
                    ok = True
                    break
                time.sleep(0.05)
            cap.release()
        except Exception as e:
            logger.error(f"RTSP test failed: {e}")
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("Test URL (2 seconds)")
        QMessageBox.information(self, "Test Result",
                                "Stream opened successfully ✅" if ok else "Failed to open stream ❌ (Check user/pw/host/port)")

    def submit(self):
        name = self.name_code.text().strip()
        url = self.preview_url.text().strip()
        url2 = self.preview_url2.text().strip()
        zone = self.zone.currentText()
        comp = self.department.currentText().strip() if self.department.currentText().strip() else None

        if not name or not url:
            QMessageBox.warning(self, "ข้อมูลไม่ครบ", "กรุณากรอกชื่อกล้องและ RTSP URL")
            return

        try:
            self.api.add_camera("", name, url, zone, comp, url2=url2)
            QMessageBox.information(self, "✅ สำเร็จ", "เพิ่มกล้องเรียบร้อยแล้ว!")
            self.cameraAdded.emit()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "❌ เกิดข้อผิดพลาด", f"Failed to add camera:\n{str(e)}")

class ResetTempPasswordDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Create Temporary Password")
        self.setFixedSize(420, 320)
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                padding: 8px;
                background-color: #1b263b;
                color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b;
                color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.username = QLineEdit(); self.username.setPlaceholderText("target username")
        self.expire = QSpinBox(); self.expire.setRange(5, 1440); self.expire.setValue(30)
        self.custom_pw_chk = QCheckBox("Use custom temp password")
        self.custom_pw = QLineEdit(); self.custom_pw.setPlaceholderText("leave blank for auto-generated"); self.custom_pw.setEnabled(False)

        form.addRow("Username:", self.username)
        form.addRow("Expires (minutes):", self.expire)
        form.addRow("", self.custom_pw_chk)
        form.addRow("Custom password:", self.custom_pw)
        v.addLayout(form)

        self.custom_pw_chk.toggled.connect(self.custom_pw.setEnabled)

        self.create_btn = QPushButton("Create Temporary Password"); v.addWidget(self.create_btn)

        self.result = QLabel(""); self.result.setWordWrap(True); self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("QLabel { background:#e0e6ed; color:#0D1117; font-weight:600; border-radius: 6px; padding: 10px; }")

        hb = QHBoxLayout()
        self.copy_btn = QPushButton("Copy to Clipboard"); self.copy_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        hb.addStretch(1); hb.addWidget(self.copy_btn); hb.addWidget(self.close_btn); v.addLayout(hb)

        self.create_btn.clicked.connect(self.handle_create)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.close_btn.clicked.connect(self.accept)

    def handle_create(self):
        username = self.username.text().strip()
        if not username:
            QMessageBox.warning(self, "Missing", "Please enter username"); return

        temp_pw = self.custom_pw.text().strip() if self.custom_pw_chk.isChecked() else None
        try:
            js = self.api.admin_reset_temp_password(username, self.expire.value(), temp_pw)
            pw = js.get("temp_password", ""); mins = js.get("expire_minutes", 30)
            self.result.setText(f"Temporary password for <b>{username}</b>:<br>"
                                f"<span style='font-size:16px;'>{pw}</span><br>"
                                f"(valid for {mins} minutes)")
            self.copy_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def copy_to_clipboard(self):
        try:
            pw = self.result.text().split("<span")[1].split(">")[1].split("<")[0]
        except Exception:
            pw = self.result.text()
        QApplication.clipboard().setText(pw)
        QMessageBox.information(self, "Copied", "Temporary password copied.")

class EditCameraDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Edit Camera")
        self.setFixedSize(480, 400)
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                padding: 8px;
                background-color: #1b263b;
                color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b;
                color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)
        v = QVBoxLayout(self); f = QFormLayout()
        try:
            self.cameras = self.api.list_cameras()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cameras: {e}"); self.cameras = []
        self.cam_cb = QComboBox()
        for cam in self.cameras:
            code = cam.get("camera_name", ""); name = cam.get("camera_name", "")
            self.cam_cb.addItem(f"{code} | {name}", cam)
        self.url = QLineEdit()
        self.url2 = QLineEdit()
        self.zone = QComboBox(); self.zone.addItems(["face", "car"])
        self.department = QComboBox()
        initial_departments = [
            "(พิมพ์เพื่อเพิ่ม)", "CENTER", "CONDO", "EPO", "EQR", "OFFICE1-2&FIN", 
            "R&D", "TETSO", "TEBP", "TEI", "TER"
        ]
        unique_departments = sorted(list(set(initial_departments) - {"(พิมพ์เพื่อเพิ่ม)"}))
        self.department.addItems(["(พิมพ์เพื่อเพิ่ม)"] + unique_departments)

        self.department.setEditable(True)
        self.department.lineEdit().setPlaceholderText("เลือกหรือพิมพ์แผนกใหม่...")

        def handle_text_change(text):
            if text == "(พิมพ์เพื่อเพิ่ม)":
                self.department.lineEdit().setText("")
        self.department.currentTextChanged.connect(handle_text_change)
        f.addRow("Select Camera:", self.cam_cb)
        f.addRow("RTSP URL (Main):", self.url)
        f.addRow("RTSP URL (Sub):", self.url2)
        f.addRow("Zone:", self.zone)
        f.addRow("Department:", self.department)
        v.addLayout(f)
        self.msg = QLabel(""); self.msg.setAlignment(Qt.AlignCenter); v.addWidget(self.msg)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel); v.addWidget(btns)
        btns.accepted.connect(self.handle_save); btns.rejected.connect(self.reject)
        self.cam_cb.currentIndexChanged.connect(self.populate_fields)
        if self.cameras: self.populate_fields(0)

    def populate_fields(self, idx):
        cam = self.cam_cb.itemData(idx)
        if not cam: return
        self.url.setText(cam.get("url", ""))
        self.url2.setText(cam.get("url2", ""))
        self.zone.setCurrentText(cam.get("zone", "face") or "face")
        self.department.setCurrentText(cam.get("department", "") or "")

    def handle_save(self):
        current_index = self.cam_cb.currentIndex()
        if current_index < 0:
             QMessageBox.warning(self, "ผิดพลาด", "กรุณาเลือกกล้องที่ต้องการแก้ไข")
             return

        selected_camera_data = self.cam_cb.itemData(current_index)
        if not selected_camera_data:
            QMessageBox.critical(self, "ผิดพลาด", "ไม่สามารถดึงข้อมูลกล้องที่เลือกได้")
            return

        code = selected_camera_data.get("camera_name")
        if not code:
             QMessageBox.critical(self, "ผิดพลาด", "ไม่พบ Camera Name/Code ในข้อมูลกล้อง")
             return

        url = self.url.text().strip()
        url2 = self.url2.text().strip()
        zone = self.zone.currentText()
        comp = self.department.currentText().strip()
        if comp == "(พิมพ์เพื่อเพิ่ม)" or not comp:
            comp = None

        try:
            self.api.update_camera(code, url=url, zone=zone, comp=comp)
            QMessageBox.information(self, "สำเร็จ", "แก้ไขกล้องเรียบร้อย")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "เกิดข้อผิดพลาด", f"Failed to update camera: {str(e)}")

class DeleteCameraDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Delete Camera")
        self.setFixedSize(300, 150)
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                padding: 8px;
                background-color: #1b263b;
                color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b;
                color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.camera_code = QLineEdit(); self.camera_code.setPlaceholderText("Camera code (camera_name)")
        form.addRow("Camera Code:", self.camera_code); v.addLayout(form)
        self.btn = QPushButton("Delete Camera"); self.btn.clicked.connect(self.handle_delete); v.addWidget(self.btn)
        self.msg = QLabel(""); self.msg.setAlignment(Qt.AlignCenter); v.addWidget(self.msg)

    def handle_delete(self):
        camera_code = self.camera_code.text().strip()
        if not camera_code:
            self.msg.setText("Please enter camera code"); return
        try:
            self.api.delete_camera(camera_code)
            QMessageBox.information(self, "Success", f"Camera {camera_code} deleted successfully")
            self.accept()
        except Exception as e:
            self.msg.setText(f"Error: {e}")
class CameraEventLogDialog(QDialog):
    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Camera Status Event Log")
        self.setMinimumSize(800, 500)
        
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QDateEdit {
                border: 1px solid #2a3f54; border-radius: 6px;
                padding: 8px; background-color: #1b263b; color: #E6EDF3;
            }
            QTableWidget {
                gridline-color: #2a3f54; font-size: 12px;
                background-color: #1b263b; alternate-background-color: #243447;
                color: #E6EDF3; 
            }
            QHeaderView::section {
                background: #243447; color: #E6EDF3; padding: 10px;
                font-weight: 600; border: 1px solid #2a3f54;
            }
            QPushButton {
                background-color: #0f4c81; color: #FFFFFF;
                border-radius: 6px; padding: 8px 12px; font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)
        
        layout = QVBoxLayout(self)
        
        control_bar = QHBoxLayout()
        control_bar.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit(QDate.currentDate().addDays(-7))
        self.start_date.setCalendarPopup(True)
        control_bar.addWidget(self.start_date)
        
        control_bar.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        control_bar.addWidget(self.end_date)
        
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.fetch_events)
        control_bar.addWidget(self.btn_refresh)
        control_bar.addStretch(1)
        layout.addLayout(control_bar)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Event Time", "Camera Name", "Status"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)
        
        QTimer.singleShot(50, self.fetch_events)

    def fetch_events(self):
        start_str = self.start_date.date().toString("yyyy-MM-dd")
        end_str = self.end_date.date().toString("yyyy-MM-dd")
        
        try:
            self.btn_refresh.setText("Loading...")
            QApplication.processEvents()
            
            events = self.api.get_camera_events(start_str, end_str) 
            
            self.table.setSortingEnabled(False)
            self.table.setRowCount(0)
            
            for row, item in enumerate(events):
                self.table.insertRow(row)
                
                self.table.setItem(row, 0, QTableWidgetItem(item.get('time_str', '')))
                
                self.table.setItem(row, 1, QTableWidgetItem(item.get('camera_name', '')))
                
                status_str = item.get('status', 'N/A')
                status_item = QTableWidgetItem(status_str)
                if status_str == "DOWN":
                    status_item.setForeground(QColor("#FF9999"))
                elif status_str == "OK":
                    status_item.setForeground(QColor("#99FF99"))
                self.table.setItem(row, 2, status_item)

            self.table.setSortingEnabled(True)
            self.table.sortItems(0, Qt.DescendingOrder)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load events:\n{str(e)}")
        finally:
            self.btn_refresh.setText("Refresh")

class ReportsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Reports (Face / Car)")
        self.setMinimumSize(980, 640)
        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            QLabel  { color: #E6EDF3; }
            QLineEdit, QComboBox {
                border: 1px solid #2a3f54;
                border-radius: 6px;
                padding: 8px;
                background-color: #1b263b;
                color: #E6EDF3;
            }
            QLineEdit::placeholder { color: #9FB2C1; }
            QComboBox QAbstractItemView {
                background: #1b263b;
                color: #E6EDF3;
                selection-background-color: #2a3f54;
            }
            QCheckBox { color: #E6EDF3; }
            QPushButton {
                background-color: #0f4c81;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1565a7; }
        """)

        root = QVBoxLayout(self)

        form = QFormLayout()
        self.start_dt = QDateTimeEdit(); self.start_dt.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.start_dt.setCalendarPopup(True)
        self.start_dt.setDateTime(QDateTime(QDate.currentDate(), QTime(0, 0, 0)))
        self.end_dt = QDateTimeEdit(); self.end_dt.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.end_dt.setCalendarPopup(True)
        self.end_dt.setDateTime(QDateTime.currentDateTime())

        quick = QHBoxLayout()
        self.btn_today = QPushButton("Today"); self.btn_24h = QPushButton("Last 24h"); self.btn_7d  = QPushButton("Last 7d")
        for b in (self.btn_today, self.btn_24h, self.btn_7d): b.setFixedHeight(26); quick.addWidget(b)
        quick.addStretch(1)

        self.dept_cb = QComboBox(); self.dept_cb.setEditable(False)
        self.type_cb = QComboBox(); self.type_cb.addItems(["ทั้งหมด", "Face", "Car"]); self.type_cb.setCurrentIndex(0)
        self.camera_code = QLineEdit(); self.camera_code.setPlaceholderText("e.g., CAM01 (optional)")
        self.q_line = QLineEdit(); self.q_line.setPlaceholderText("Search: name / plate / department / province / camera / emp_id / status")
        self.limit_spin = QSpinBox(); self.limit_spin.setRange(1, 5000); self.limit_spin.setValue(500)

        self._populate_departments()

        form.addRow("Start:", self.start_dt); form.addRow("End:", self.end_dt); form.addRow("", quick)
        form.addRow("Department (camera):", self.dept_cb)
        form.addRow("Type:", self.type_cb)
        form.addRow("Camera:", self.camera_code); form.addRow("Search:", self.q_line); form.addRow("Row Limit:", self.limit_spin)
        root.addLayout(form)

        actions = QHBoxLayout()
        self.btn_refresh = QPushButton("🔎 Search / Refresh"); self.btn_export = QPushButton("📁 Export CSV")
        actions.addWidget(self.btn_refresh); actions.addStretch(1); actions.addWidget(self.btn_export); root.addLayout(actions)

        self.summary = QLabel("")
        self.summary.setStyleSheet("QLabel { background: #f7f9fc; border: 1px solid #dfe6f0; border-radius: 8px; padding: 8px 10px; color: #0D1117; font-weight: 600; }")
        root.addWidget(self.summary)

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["Time", "Camera", "Zone", "Type", "Name/Plate", "Dept/Province/Status"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("QTableWidget { gridline-color: #e6ebf2; font-size: 12px; } QHeaderView::section { background: #243447; color: #E6EDF3; padding: 6px; font-weight: 600; }")
        root.addWidget(self.table, stretch=1)

        self.status = QLabel("Ready"); self.status.setStyleSheet("color:#E6EDF3;"); root.addWidget(self.status)

        self.btn_refresh.clicked.connect(self.fetch_data); self.btn_export.clicked.connect(self.export_csv)
        self.typing_timer = QTimer(self); self.typing_timer.setInterval(350); self.typing_timer.setSingleShot(True); self.typing_timer.timeout.connect(self.fetch_data)
        self.q_line.textChanged.connect(self.typing_timer.start)
        self.btn_today.clicked.connect(self._quick_today); self.btn_24h.clicked.connect(self._quick_24h); self.btn_7d.clicked.connect(self._quick_7d)

        self.fetch_data()

    def _populate_departments(self):
        self.dept_cb.clear()
        self.dept_cb.addItem("ทั้งหมด", None)
        self.dept_cb.addItem("ไม่มีแผนก", "")
        try:
            cams = self.parent().api.list_cameras()
            comps = sorted(set((c.get("comp") or "") for c in cams))
            for comp in comps:
                if comp == "": continue
                self.dept_cb.addItem(comp, comp)
        except Exception:
            pass

    def _quick_today(self):
        today = QDate.currentDate()
        self.start_dt.setDateTime(QDateTime(today, QTime(0, 0, 0)))
        self.end_dt.setDateTime(QDateTime.currentDateTime()); self.fetch_data()

    def _quick_24h(self):
        end = QDateTime.currentDateTime(); start = end.addSecs(-24 * 3600)
        self.start_dt.setDateTime(start); self.end_dt.setDateTime(end); self.fetch_data()
    def _quick_7d(self):
        end = QDateTime.currentDateTime(); start = end.addDays(-7)
        self.start_dt.setDateTime(start); self.end_dt.setDateTime(end); self.fetch_data()

    def fetch_data(self):
        self.status.setText("Fetching data..."); self.btn_refresh.setEnabled(False)
        try:
            start = self.start_dt.dateTime().toString("yyyy-MM-dd HH:mm:ss")
            end = self.end_dt.dateTime().toString("yyyy-MM-dd HH:mm:ss")

            params = {
                "start": start, 
                "end": end, 
                "limit": self.limit_spin.value()
            }

            dept = self.dept_cb.itemData(self.dept_cb.currentIndex())
            if dept is not None:
                params["department"] = dept if dept != "" else ""

            type_val = self.type_cb.currentText()
            if type_val == "Face":
                params["type_"] = "face"
            elif type_val == "Car":
                params["type_"] = "car"

            q_main = self.q_line.text().strip()
            q_cam = self.camera_code.text().strip()

            all_q_parts = [t for t in q_main.split() if t]
            if q_cam:
                all_q_parts.append(q_cam)

            if all_q_parts:
                params["q"] = " ".join(all_q_parts) 

            print(f"[DEBUG] Params sent: {params}")
            data = self.parent().api.list_reports(**params)
            print(f"[DEBUG] Items received: {len(data.get('items', []))}")


            items = data.get("items", [])

            self.populate_table(items)

            face_count = sum(1 for it in items if it.get("type") == "face")
            car_count = len(items) - face_count
            self.summary.setText(f"Total: {len(items)} | Face: {face_count} | Car: {car_count}")

            self.status.setText(f"Done • {len(items)} rows")
        except Exception as e:
            self.status.setText(f"Error: {e}"); QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_refresh.setEnabled(True)

    def populate_table(self, items):
        self.table.setRowCount(0); self.table.setSortingEnabled(False)
        for it in items:
            r = self.table.rowCount(); self.table.insertRow(r)
            
            ts = it.get("timestamp", "")
            dt_obj = datetime.fromisoformat(ts.replace("Z", "+00:00")) if "T" in ts else datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            self.table.setItem(r, 0, QTableWidgetItem(dt_obj.strftime("%d %b %H:%M")))
            
            self.table.setItem(r, 1, QTableWidgetItem(it.get("camera_name", "")))
            
            self.table.setItem(r, 2, QTableWidgetItem(it.get("zone", "")))
            
            type_icon = "👤" if it.get("type") == "face" else "🚗"
            self.table.setItem(r, 3, QTableWidgetItem(type_icon))
            
            if it.get("type") == "face":
                name = it.get("full_name", "")
                emp_id = it.get("emp_id", "")
                display = f"{name}" + (f" ({emp_id})" if emp_id else "")
            else:
                display = it.get("plate", "")
            self.table.setItem(r, 4, QTableWidgetItem(display))
            
            if it.get("type") == "face":
                dept = it.get("department", "")
                conf = it.get("confidence")
                sim = it.get("similarity")
                extra = []
                if conf:
                    extra.append(f"Conf:{conf:.1%}")
                if sim:
                    extra.append(f"Sim:{sim:.1%}")
                display = dept + (f" | {' | '.join(extra)}" if extra else "")
            else:
                prov = it.get("province", "")
                status = it.get("status", "")
                display = prov + (f" | {status}" if status else "")
            self.table.setItem(r, 5, QTableWidgetItem(display))
            
        self.table.setSortingEnabled(True)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", f"report_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv", "CSV Files (*.csv)")
        if not path: return
        
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            headers = ["Time", "Camera", "Zone", "Type", "Name/Plate", "Department/Province"]
            w.writerow(headers)
            
            for r in range(self.table.rowCount()):
                row = []
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    row.append(item.text() if item else "")
                w.writerow(row)
        
        QMessageBox.information(self, "Success", f"CSV saved to {path}")

class SelectCameraDialog(QDialog):
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera for Recordings")
        self.setMinimumSize(400, 480)
        v = QVBoxLayout(self)
        self.search = QLineEdit(); self.search.setPlaceholderText("Search by camera code or name..."); v.addWidget(self.search)
        self.listw = QListWidget(); v.addWidget(self.listw)
        self.cameras = cameras; self.filtered = cameras; self.populate_list()
        self.search.textChanged.connect(self.on_search)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); btns.accepted.connect(self.accept); btns.rejected.connect(self.reject); v.addWidget(btns)

    def on_search(self, text):
        t = text.strip().lower()
        if not t: self.filtered = self.cameras
        else:
            self.filtered = [cam for cam in self.cameras if t in (cam.get("camera_code", "") or cam.get("camera_name", "")).lower() or t in cam.get("camera_name", "").lower()]
        self.populate_list()

    def populate_list(self):
        self.listw.clear()
        for cam in self.filtered:
            item = QListWidgetItem(f"{cam.get('camera_code', cam.get('camera_name'))} | {cam.get('camera_name')}")
            item.setData(Qt.UserRole, cam); self.listw.addItem(item)

    def get_selected_camera(self):
        it = self.listw.currentItem(); return it.data(Qt.UserRole) if it else None

class YouTubeLikePlayer(QWidget):
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(False)

        self.player = QMediaPlayer(self)
        self.video = QVideoWidget(self)
        self.player.setVideoOutput(self.video)

        self.video.installEventFilter(self)

        self.thumbnail_label = QLabel(self.video)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(13,27,42,0.95), stop:1 rgba(27,38,59,0.95));
                border-radius: 8px; color: #E6EDF3; font-size: 14px; padding: 20px;
            }""")
        self.thumbnail_label.setText("🎥 เลือกวิดีโอเพื่อดูตัวอย่าง")
        self.thumbnail_label.hide()

        self.bar = QWidget(self)
        self.bar.setStyleSheet("background: rgba(0,0,0,180); border-radius: 0 0 8px 8px;")
        bar_layout = QHBoxLayout(self.bar)
        bar_layout.setContentsMargins(12, 8, 12, 8)
        bar_layout.setSpacing(12)

        self.btn_play = QToolButton(self.bar)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setStyleSheet("""
            QToolButton { width: 36px; height: 36px; border-radius: 18px; background: rgba(255,255,255,0.2); }
            QToolButton:hover { background: rgba(255,255,255,0.3); }
        """)

        self.lbl_time = QLabel("0:00 / 0:00", self.bar)
        self.lbl_time.setStyleSheet("color: #E6EDF3; font-family: 'Segoe UI', sans-serif; font-size: 12px; min-width: 100px;")

        self.slider = QSlider(Qt.Horizontal, self.bar)
        self.slider.setRange(0, 1000) 
        self.slider.sliderMoved.connect(self.seek_slider_moved)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px; }
            QSlider::handle:horizontal { background: #00aaff; border: none; border-radius: 8px; width: 16px; margin: -6px 0; }
            QSlider::handle:horizontal:hover { background: #33bbff; }
        """)

        self.btn_vol = QToolButton(self.bar)
        self.btn_vol.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.btn_vol.clicked.connect(self._toggle_mute)
        self._is_muted = False
        self._last_volume = 80
        
        self.vol_slider = QSlider(Qt.Horizontal, self.bar)
        self.vol_slider.setFixedWidth(90)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(self._last_volume)
        self.vol_slider.valueChanged.connect(self.set_volume)

        self.btn_speed = QToolButton(self.bar)
        self.btn_speed.setText("1.0x")
        speed_menu = QMenu(self.btn_speed)
        for sp in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            act = QAction(f"{sp}x", speed_menu, triggered=lambda _, s=sp: self.set_rate(s))
            speed_menu.addAction(act)
        self.btn_speed.setMenu(speed_menu)
        self.btn_speed.setPopupMode(QToolButton.InstantPopup)
        self.btn_speed.setStyleSheet("""
            QToolButton { padding: 6px 10px; border-radius: 6px; background: rgba(255,255,255,0.1); color: #E6EDF3; }
            QToolButton:hover { background: rgba(255,255,255,0.2); }
        """)

        self.btn_fs = QToolButton(self.bar)
        self.btn_fs.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.btn_fs.clicked.connect(self.toggle_fullscreen)
        self.btn_fs.setStyleSheet("""
            QToolButton { width: 36px; height: 36px; border-radius: 18px; background: rgba(255,255,255,0.1); }
            QToolButton:hover { background: rgba(255,255,255,0.2); }
        """)

        bar_layout.addWidget(self.btn_play)
        bar_layout.addWidget(self.slider, 1)
        bar_layout.addWidget(self.lbl_time)
        bar_layout.addSpacing(12)
        bar_layout.addWidget(self.btn_vol)
        bar_layout.addWidget(self.vol_slider)
        bar_layout.addSpacing(12)
        bar_layout.addWidget(self.btn_speed)
        bar_layout.addSpacing(8)
        bar_layout.addWidget(self.btn_fs)

        self.big_play = QToolButton(self.video) 
        self.big_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.big_play.setIconSize(QSize(96, 96))
        self.big_play.setStyleSheet("""
            QToolButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(0,122,255,0.9), stop:1 rgba(0,122,255,0.7)); 
                border-radius: 48px; 
                border: 3px solid rgba(255,255,255,0.9);
            }
            QToolButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(0,150,255,1), stop:1 rgba(0,122,255,0.9)); 
            }
            QToolButton:pressed { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(0,100,230,0.9), stop:1 rgba(0,100,230,0.7));
            }
        """)
        self.big_play.clicked.connect(self.toggle_play)
        self.big_play.hide()
        
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.video)
        
        self.bar_timer = QTimer(self)
        self.bar_timer.setInterval(2500)
        self.bar_timer.timeout.connect(self._maybe_hide_controls)
        
        self._mouse_active = True
        self._controls_visible = True
        self._slider_is_dragging = False
        
        self.player.positionChanged.connect(self._on_pos)
        self.player.durationChanged.connect(self._on_dur)
        self.player.stateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.error.connect(self._on_player_error)
        
        self.player.setVolume(self._last_volume)
        
        self._has_thumbnail = False
        self._current_media_url = None
        self._last_position = 0

    def eventFilter(self, source, event):
        if source is self.video and event.type() == QEvent.MouseButtonDblClick:
            self.doubleClicked.emit()
            return True
        return super().eventFilter(source, event)

    def set_media(self, url_or_path: str):
        if not url_or_path:
            return
        
        logger.info(f"QMediaPlayer: Setting media to {url_or_path}")
        self._current_media_url = url_or_path
        url = QUrl.fromLocalFile(url_or_path) if os.path.exists(url_or_path) else QUrl(url_or_path)
        
        self.player.stop()
        self._reset_ui()
        
        self.player.setMedia(QMediaContent(url))
        self.show_thumbnail(True)
        self.show_controls(True)

    def set_thumbnail(self, thumbnail_path: str = None, thumbnail_bytes: bytes = None):
        pixmap = None
        if thumbnail_path and os.path.exists(thumbnail_path):
            pixmap = QPixmap(thumbnail_path)
        elif thumbnail_bytes:
            pixmap = QPixmap()
            pixmap.loadFromData(thumbnail_bytes)
            
        if pixmap and not pixmap.isNull():
            self.thumbnail_label.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._has_thumbnail = True
            logger.info(f"✅ Thumbnail loaded: {pixmap.width()}x{pixmap.height()}")
        else:
            self.thumbnail_label.setText("🎥 ไม่พบตัวอย่าง")
            self.thumbnail_label.setPixmap(QPixmap())
            self._has_thumbnail = False
        self.show_thumbnail(True)


    def show_thumbnail(self, show: bool = True):
        if show and self._has_thumbnail:
            self.thumbnail_label.setVisible(True)
            self.thumbnail_label.raise_()
            self.big_play.setVisible(True)
            self.big_play.raise_()
        elif show and not self._has_thumbnail:
            self.thumbnail_label.setText("🎥 เลือกวิดีโอเพื่อดูตัวอย่าง")
            self.thumbnail_label.setVisible(True)
            self.thumbnail_label.raise_()
            self.big_play.setVisible(True)
            self.big_play.raise_()
        else:
            self.thumbnail_label.hide()
            self.big_play.hide()

    def play(self): 
        logger.debug("QMediaPlayer: Play command received.")
        if self.player.state() != QMediaPlayer.PlayingState:
            self.player.play()
        
    def pause(self): 
        logger.debug("QMediaPlayer: Pause command received.")
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        
    def stop(self): 
        logger.debug("QMediaPlayer: Stop command received.")
        self.player.stop()

    def toggle_play(self):
        state = self.player.state()
        if state == QMediaPlayer.PlayingState:
            self.pause()
        elif state == QMediaPlayer.PausedState:
            self.play()
        elif state == QMediaPlayer.StoppedState:
             if self.player.duration() > 0 and self._last_position >= (self.player.duration() - 1000):
                 logger.debug("Restarting media from beginning.")
                 self.player.setPosition(0)
                 self.play()
             else:
                 self.play()
        else: # NoMedia
             if self._current_media_url:
                 logger.debug("No media loaded, trying to set and play.")
                 self.set_media(self._current_media_url)
                 self.player.play()

    def set_rate(self, r: float):
        logger.debug(f"Setting playback rate to {r}")
        self.player.setPlaybackRate(r)
        self.btn_speed.setText(f"{r:.2f}x")

    def _slider_pressed(self):
        self._slider_is_dragging = True
        logger.debug("Slider pressed (dragging started)")

    def _slider_released(self):
        self._slider_is_dragging = False
        logger.debug("Slider released (dragging finished)")
        self.seek_slider_value(self.slider.value())

    def seek_slider_moved(self, v_int: int):
        if self.player.duration() > 0:
            pos_ms = int((v_int / 1000.0) * self.player.duration())
            self.lbl_time.setText(f"{self._fmt(pos_ms)} / {self._fmt(self.player.duration())}")
        
    def seek_slider_value(self, v_int: int):
        if self.player.duration() > 0 and self.player.isSeekable():
            pos_ms = int((v_int / 1000.0) * self.player.duration())
            logger.debug(f"Seeking to {pos_ms} ms (Slider: {v_int})")
            self.player.setPosition(pos_ms)
        else:
             logger.warning(f"Cannot seek (Duration: {self.player.duration()}, Seekable: {self.player.isSeekable()})")

    def set_volume(self, v_int: int):
        self.player.setVolume(v_int)
        if v_int > 0 and self._is_muted:
            self.player.setMuted(False)
            self._is_muted = False
        elif v_int == 0:
            self.player.setMuted(True)
            self._is_muted = True
        if not self._is_muted:
            self._last_volume = v_int
        self._update_volume_icon()

    def _toggle_mute(self):
        self._is_muted = not self._is_muted
        self.player.setMuted(self._is_muted)
        logger.debug(f"Mute toggled to: {self._is_muted}")
        
        if not self._is_muted and self.player.volume() == 0:
             restore_vol = self._last_volume if self._last_volume > 0 else 80
             self.vol_slider.setValue(restore_vol)
        
        self._update_volume_icon()

    def _update_volume_icon(self):
        if self._is_muted or self.player.volume() == 0:
            try:
                 icon = self.style().standardIcon(QStyle.SP_MediaVolumeMuted)
                 if icon.isNull(): raise AttributeError("Icon SP_MediaVolumeMuted is null")
            except AttributeError:
                 logger.warning("SP_MediaVolumeMuted not available, using SP_MediaVolume fallback.")
                 icon = self.style().standardIcon(QStyle.SP_MediaVolume) # Fallback
            self.btn_vol.setIcon(icon)
        else:
            self.btn_vol.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))

    def _on_pos(self, pos_ms):
        dur = self.player.duration()
        if not self._slider_is_dragging and dur > 0:
            pos_ratio = pos_ms / dur if dur > 0 else 0 
            self.slider.blockSignals(True)
            self.slider.setValue(int(pos_ratio * 1000))
            self.slider.blockSignals(False)
        
        self.lbl_time.setText(f"{self._fmt(pos_ms)} / {self._fmt(dur)}")
        self._last_position = pos_ms

    def _on_dur(self, dur_ms):
        logger.debug(f"QMediaPlayer: Duration changed to {dur_ms} ms")
        is_seekable = (dur_ms > 0 and self.player.isSeekable())
        self.slider.setEnabled(is_seekable)
        self._on_pos(self.player.position())

    def _on_state(self, state):
        logger.debug(f"QMediaPlayer: State changed to {state}")
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.show_thumbnail(False)
            self.start_auto_hide()
        elif state == QMediaPlayer.PausedState:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.bar_timer.stop()
            self.show_controls(True)
        elif state == QMediaPlayer.StoppedState:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self._reset_ui()
            self.show_thumbnail(True)
            self.show_controls(True)
    
    def _on_media_status(self, status):
        logger.debug(f"QMediaPlayer: MediaStatus changed to {status}")
        self._reposition_overlays()
        if status == QMediaPlayer.LoadedMedia:
            logger.info("Media successfully loaded by QMediaPlayer (WMF).")
            self._on_dur(self.player.duration())
        elif status == QMediaPlayer.InvalidMedia:
            self._on_player_error(self.player.error())
        elif status == QMediaPlayer.EndOfMedia:
             logger.info("QMediaPlayer: End of media reached.")
             self.player.stop()

    @pyqtSlot(QMediaPlayer.Error)
    def _on_player_error(self, error):
        error_string = self.player.errorString()
        if error != QMediaPlayer.NoError:
            logger.error(f"QMediaPlayer Error #{error}: {error_string} (URL: {self._current_media_url})")
            parent_dialog = self.parentWidget()
            if isinstance(parent_dialog, RecordingsDialog):
                 QMessageBox.warning(parent_dialog, "Playback Error",
                                     f"Could not play the video.\n\nError ({error}):\n{error_string}")
            else:
                 QMessageBox.warning(self, "Playback Error",
                                     f"Could not play the video.\n\nError ({error}):\n{error_string}")
            self.stop()

    def _reset_ui(self):
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.setEnabled(False)
        self.lbl_time.setText("0:00 / 0:00")
        self._last_position = 0

    def _fmt(self, ms):
        if ms < 0: ms = 0
        s = int(ms / 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def resizeEvent(self, e):
        self._reposition_overlays()
        super().resizeEvent(e)

    def _reposition_overlays(self):
        r = self.rect()
        video_rect = self.video.geometry()

        self.thumbnail_label.setGeometry(video_rect)
        
        self.big_play.setGeometry(
            int(video_rect.x() + (video_rect.width() - 96) / 2),
            int(video_rect.y() + (video_rect.height() - 96) / 2),
            96, 96
        )
        self.big_play.raise_()

        bar_height = 56
        self.bar.setGeometry(0, r.height() - bar_height, r.width(), bar_height)
        self.bar.raise_()

    def mouseMoveEvent(self, e: QMouseEvent):
        self._mouse_active = True
        self.show_controls(True)
        self.start_auto_hide()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self._mouse_active = False
        self.start_auto_hide()
        super().leaveEvent(e)

    def start_auto_hide(self):
        self.bar_timer.stop()
        if self.player.state() == QMediaPlayer.PlayingState:
            self.bar_timer.start()

    def _maybe_hide_controls(self):
        if (self.player.state() == QMediaPlayer.PlayingState and
                not self._mouse_active and
                self._controls_visible):
            self.show_controls(False)
        self._mouse_active = False

    def show_controls(self, show: bool):
        self.bar.setVisible(show)
        self._controls_visible = show
        self.setCursor(Qt.ArrowCursor if show else Qt.BlankCursor)
        if show:
            self.bar.raise_()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()
        e.accept()

    def toggle_fullscreen(self):
        vw = self.window()
        if vw:
            if not vw.isFullScreen():
                vw.showFullScreen()
            else:
                vw.showNormal()

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Space:
            self.toggle_play()
            return
        if key == Qt.Key_Right:
            if self.player.isSeekable():
                 self.player.setPosition(self.player.position() + 5000)
            return
        if key == Qt.Key_Left:
            if self.player.isSeekable():
                 self.player.setPosition(max(0, self.player.position() - 5000))
            return
        if key == Qt.Key_F:
            self.toggle_fullscreen()
            return
        if key == Qt.Key_M:
            self._toggle_mute()
            return
        super().keyPressEvent(e)

class RecordingsDialog(QDialog):
    PAGE_SIZE = None

    def __init__(self, parent=None, camera_name: Optional[str] = None, zone: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Recordings - {camera_name or 'Unknown Camera'}")
        self.setMinimumSize(1400, 780)

        self.setStyleSheet("""
            QDialog { background-color: #0d1b2a; color: #E6EDF3; }
            
            QMessageBox { background-color: #1b263b; }
            QMessageBox QLabel { color: #E6EDF3; min-width: 300px; }
            QMessageBox QPushButton { background-color: #0f4c81; color: #FFFFFF; border-radius: 6px; padding: 8px 16px; font-weight: 600; min-width: 80px; }
            QMessageBox QPushButton:hover { background-color: #1565a7; }

            QLabel { color: #E6EDF3; }
            QLineEdit, QComboBox, QDateEdit {
                border: 1px solid #2a3f54; border-radius: 6px; padding: 8px;
                background-color: #1b263b; color: #E6EDF3;
            }
            QLineEdit:focus, QDateEdit:focus { border: 1px solid #00aaff; }
            QLineEdit::placeholder { color: #9FB2C1; }
            QPushButton {
                background-color: #0f4c81; color: #FFFFFF; border-radius: 6px;
                padding: 8px 16px; font-weight: 600; border: none;
            }
            QPushButton:hover { background-color: #1565a7; }
            QPushButton:disabled { background-color: #2a3f54; color: #778899; }
            QTableWidget {
                gridline-color: #2a3f54; font-size: 12px;
                background-color: #1b263b; alternate-background-color: #243447;
                color: #E6EDF3; 
            }
            QHeaderView::section {
                background: #243447; color: #E6EDF3; padding: 10px;
                font-weight: 600; border: 1px solid #2a3f54;
            }
            QGroupBox {
                border: 1px solid #2a3f54; border-radius: 8px; margin-top: 12px;
                font-weight: 600; color: #E6EDF3;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px;
                color: #E6EDF3; background: #0f4c81; border-radius: 4px;
            }
            QToolBar {
                background-color: #112240; spacing: 12px; padding: 8px;
                border-bottom: 1px solid #2a3f54; border-radius: 8px;
            }
            QSplitter::handle { background: #112240; width: 6px; }
            QDoubleSpinBox {
                border: 1px solid #2a3f54; border-radius: 6px; padding: 6px;
                background-color: #1b263b; color: #E6EDF3;
            }
            QSlider::groove:horizontal:disabled { background: #2a3f54; }
            QSlider::handle:horizontal:disabled { background: #556677; }
        """)

        self.api = getattr(parent, 'api', None)
        self.camera_name = camera_name or "Unknown"
        self.zone = zone or "Unknown"

        self._all_items: list[dict] = []
        self._view_items: list[dict] = []
        self._files: list[dict] = []
        self._page: int = 1
        self._current_item: Optional[dict] = None
        self._current_video_start_dt: Optional[datetime] = None
        self._player_is_maximized = False

        self._build_ui()
        QTimer.singleShot(100, self.reload)

    def _build_ui(self):
        self.top_toolbar = QToolBar(self)
        self.top_toolbar.setMovable(False)
        self.top_toolbar.setIconSize(QSize(20, 20))
        lbl_cam = QLabel(f"📷 <b>{self.camera_name}</b>  |  Zone: <b>{self.zone}</b>")
        lbl_cam.setStyleSheet("font-size: 13px; padding: 4px 8px; background: rgba(15,76,129,0.3); border-radius: 6px;")
        self.top_toolbar.addWidget(lbl_cam)
        self.top_toolbar.addSeparator()
        self.top_toolbar.addWidget(QLabel("Date:"))
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setFixedWidth(140)
        self.date_edit.dateChanged.connect(self.reload)
        self.top_toolbar.addWidget(self.date_edit)
        btn_prev = QAction(self.style().standardIcon(QStyle.SP_ArrowLeft), "Previous Day", self)
        btn_today = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "Today", self)
        btn_next = QAction(self.style().standardIcon(QStyle.SP_ArrowRight), "Next Day", self)
        def _go_days(delta): self.date_edit.setDate(self.date_edit.date().addDays(delta))
        btn_prev.triggered.connect(lambda: _go_days(-1))
        btn_next.triggered.connect(lambda: _go_days(1))
        btn_today.triggered.connect(lambda: self.date_edit.setDate(QDate.currentDate()))
        self.top_toolbar.addAction(btn_prev)
        self.top_toolbar.addAction(btn_today)
        self.top_toolbar.addAction(btn_next)
        self.top_toolbar.addSeparator()
        
        self.top_toolbar.addWidget(QLabel("Search Person:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("ค้นหาบุคคล (Enter เพื่อค้นหา)...")
        self.search_edit.setFixedWidth(260)
        self.top_toolbar.addWidget(self.search_edit)
        
        self.btn_search = QPushButton("🔎 Search")
        self.top_toolbar.addWidget(self.btn_search)
        
        self.top_toolbar.addSeparator()
        self.top_toolbar.addWidget(QLabel("Size ≥ (MB):"))
        self.min_size = QDoubleSpinBox()
        self.min_size.setRange(0, 9999)
        self.min_size.setDecimals(1)
        self.min_size.setValue(0.0)
        self.min_size.setFixedWidth(80)
        self.top_toolbar.addWidget(self.min_size)
        btn_refresh = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "Refresh", self)
        btn_refresh.triggered.connect(self.reload)
        self.top_toolbar.addSeparator()
        self.top_toolbar.addAction(btn_refresh)
        
        self.split = QSplitter(self)
        self.split.setHandleWidth(6)

        self.left_wrap = QWidget()
        lv = QVBoxLayout(self.left_wrap)
        lv.setContentsMargins(12, 12, 12, 12)

        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setHandleWidth(6)

        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0,0,0,0)
        
        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(["Time", "Display Name", "Filename", "Size (MB)", "🖼️ Thumbnail", "▶ Play", "⤓ Download"])
        header = self.table.horizontalHeader()
        modes = [ QHeaderView.ResizeToContents, QHeaderView.Stretch, QHeaderView.ResizeToContents, QHeaderView.ResizeToContents, QHeaderView.Fixed, QHeaderView.Fixed, QHeaderView.Fixed ]
        for i, mode in enumerate(modes): header.setSectionResizeMode(i, mode)
        self.table.setColumnWidth(4, 90)
        self.table.setColumnWidth(5, 70)
        self.table.setColumnWidth(6, 70)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        table_layout.addWidget(self.table, 1)

        pag_layout = QHBoxLayout()
        self.lbl_count = QLabel("0 item(s)")
        pag_layout.addWidget(self.lbl_count)
        pag_layout.addStretch(1)
        self.btn_prev_page = QPushButton("◀ Previous")
        self.lbl_page = QLabel("Page 1/1")
        self.btn_next_page = QPushButton("Next ▶")
        pag_layout.addWidget(self.btn_prev_page)
        pag_layout.addWidget(self.lbl_page)
        pag_layout.addWidget(self.btn_next_page)
        table_layout.addLayout(pag_layout)

        self.detections_gb = QGroupBox("Detections in this Video")
        det_layout = QVBoxLayout(self.detections_gb)
        self.detection_list = QListWidget()
        self.detection_list.setStyleSheet("""
            QListWidget { background-color: #0d1b2a; border: 1px solid #2a3f54; }
            QListWidget::item { padding: 6px; }
            QListWidget::item:hover { background-color: #2a3f54; }
        """)
        self.detection_list.itemClicked.connect(self._on_detection_clicked)
        det_layout.addWidget(self.detection_list)

        self.left_splitter.addWidget(table_widget) 
        self.left_splitter.addWidget(self.detections_gb)
        self.left_splitter.setStretchFactor(0, 3) 
        self.left_splitter.setStretchFactor(1, 1)
        self.left_splitter.setSizes([600, 200]) 
        
        lv.addWidget(self.left_splitter)

        right_wrap = QWidget()
        rv = QVBoxLayout(right_wrap) 
        rv.setContentsMargins(12, 12, 12, 12)

        self.player = YouTubeLikePlayer(self)
        self.player.doubleClicked.connect(self.toggle_player_maximize)
        self.player.setMinimumHeight(380)
        rv.addWidget(self.player, 2) 

        self.info_gb = QGroupBox("Selected File Info")
        f = QFormLayout(self.info_gb)
        self.lbl_disp = QLabel("-")
        self.lbl_file = QLabel("-")
        self.lbl_url = QLineEdit()
        self.lbl_url.setReadOnly(True)
        self.btn_copy_url = QPushButton("📋 Copy URL")
        self.btn_copy_url.setFixedWidth(110)
        url_row = QHBoxLayout()
        url_row.addWidget(self.lbl_url, 1)
        url_row.addWidget(self.btn_copy_url)
        self.lbl_size = QLabel("-")
        f.addRow("Display Name:", self.lbl_disp)
        f.addRow("Filename:", self.lbl_file)
        f.addRow("URL:", url_row)
        f.addRow("Size:", self.lbl_size)
        rv.addWidget(self.info_gb)

        act_row_layout = QHBoxLayout()
        self.btn_open_external = QPushButton("🎬 Open External")
        self.btn_download = QPushButton("⤓ Download File")
        act_row_layout.addWidget(self.btn_open_external)
        act_row_layout.addStretch(1)
        act_row_layout.addWidget(self.btn_download)
        
        self.action_buttons_widget = QWidget() 
        self.action_buttons_widget.setLayout(act_row_layout)
        rv.addWidget(self.action_buttons_widget)

        self.split.addWidget(self.left_wrap)
        self.split.addWidget(right_wrap)
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 3)
        self.split.setSizes([600, 800])

        root = QVBoxLayout(self)
        root.addWidget(self.top_toolbar)
        root.addWidget(self.split, 1)
        root.setContentsMargins(0, 0, 0, 0)

        self.search_edit.returnPressed.connect(self.reload)
        self.btn_search.clicked.connect(self.reload)
        self.min_size.valueChanged.connect(self._refilter)
        
        self.table.cellClicked.connect(self._cell_clicked)
        self.btn_prev_page.clicked.connect(lambda: self._goto_page(self._page - 1))
        self.btn_next_page.clicked.connect(lambda: self._goto_page(self._page + 1))
        self.btn_copy_url.clicked.connect(self._copy_url)
        self.btn_open_external.clicked.connect(self._open_external)
        self.btn_download.clicked.connect(self._download_current)

        self._debounce = QTimer(self)
        self._debounce.setInterval(350)
        self._debounce.setSingleShot(True)
        self.show()
        self._debounce.timeout.connect(self._refilter)

    def toggle_player_maximize(self):
        self._player_is_maximized = not self._player_is_maximized
        is_maximized = self._player_is_maximized
        
        self.top_toolbar.setVisible(not is_maximized)
        self.left_wrap.setVisible(not is_maximized)
        self.info_gb.setVisible(not is_maximized)
        self.action_buttons_widget.setVisible(not is_maximized)
        
        handle = self.split.handle(1) 
        if handle:
            handle.setVisible(not is_maximized)
            
        if is_maximized:
            logger.info("Player maximized (Theater Mode)")
        else:
            logger.info("Player restored (Normal View)")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            if self._player_is_maximized:
                self.toggle_player_maximize()
                e.accept()
                return
        
        if self.player.hasFocus() or self.player.video.hasFocus() or self.player.bar.hasFocus():
            self.player.keyPressEvent(e)
            e.accept()
            return
            
        super().keyPressEvent(e)

    def _selected_date_str(self) -> str:
        return self.date_edit.date().toString("yyyy-MM-dd")

    def _build_stream_url(self, it: dict) -> str:
            department = it.get("department", "")
            date_str = it.get("date", self._selected_date_str())
            filename = it.get("file") or it.get("filename", "")
            base = CONFIG.get("SERVER_BASE", "").rstrip("/")
            if not base:
                logger.error("SERVER_BASE not set")
                return ""
            path_parts = ["recordings", department, self.zone, self.camera_name, date_str, filename]
            clean_path = "/".join(part for part in path_parts if part)
            base_url_with_path = f"{base}/{clean_path}"
            try:
                token = getattr(self.api, 'token', None)
                if not token:
                    logger.warning("API token is missing, URL will likely fail.")
                    return base_url_with_path
                u = urlparse(base_url_with_path)
                qs = dict(parse_qsl(u.query))
                qs["token"] = token
                final_url = urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(qs), u.fragment))
                return final_url
            except Exception as e:
                logger.error(f"Error building authenticated URL: {e}", exc_info=True)
                return base_url_with_path

    def reload(self):
        if not self.api:
            QMessageBox.critical(self, "Error", "API Client not available.")
            return
        self._clear_state()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            target_date = self._selected_date_str()
            
            person_query = self.search_edit.text().strip()
            
            all_server_items = self.api.list_recordings(
                self.camera_name, 
                self.zone, 
                date=target_date, 
                person_name=person_query
            ) or []
            
            self._all_items = all_server_items 
            self._all_items.sort(key=lambda x: x.get("modified", ""), reverse=True)
            
            log_msg = f"APIClient returned {len(self._all_items)} recordings for {self.camera_name} on {target_date}"
            if person_query:
                log_msg += f" (matching '{person_query}')"
            logger.info(log_msg)
            
        except Exception as e:
            logger.error(f"Failed to load recordings: {e}", exc_info=True)
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error Loading Recordings", f"Failed to load recordings list:\n{str(e)}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        
        self._refilter()

    def _clear_state(self):
        self._all_items = []
        self._view_items = []
        self._files = []
        self._page = 1
        self._current_item = None
        self.table.setRowCount(0)
        if hasattr(self.player, 'stop') and callable(self.player.stop):
            self.player.stop()
        self._clear_info_panel()
        
        if hasattr(self, 'detection_list') and self.detection_list:
            self.detection_list.clear()

    def _clear_info_panel(self):
        self.lbl_disp.setText("-")
        self.lbl_file.setText("-")
        self.lbl_url.setText("")
        self.lbl_size.setText("-")
        self.btn_copy_url.setEnabled(False)
        self.btn_open_external.setEnabled(False)
        self.btn_download.setEnabled(False)
        
        if hasattr(self, 'detection_list') and self.detection_list:
            self.detection_list.clear()

    def _refilter_debounced(self):
        self._debounce.start()

    def _refilter(self):
        min_mb = float(self.min_size.value())
        
        def matches(item: dict) -> bool:
            size_match = float(item.get("size_mb", 0.0)) >= min_mb
            return size_match
            
        self._view_items = [it for it in self._all_items if matches(it)]
        self._page = 1
        self._refresh_page()

    def _refresh_page(self):
        total = len(self._view_items)

        # ย้ายขึ้นก่อน if
        is_sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)

        if self.PAGE_SIZE is None:
            # แสดงทั้งหมด
            pages = 1
            self._page = 1
            start = 0
            end = total
            self._files = self._view_items
        else:
            # แบ่งหน้า
            pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
            self._page = max(1, min(self._page, pages))
            start = (self._page - 1) * self.PAGE_SIZE
            end = min(start + self.PAGE_SIZE, total)
            self._files = self._view_items[start:end]

        # --- ต่อไปนี้เหมือนเดิม ---
        self.table.setRowCount(len(self._files))
        for idx, it in enumerate(self._files):
            r = idx
            modified_iso = it.get("modified", "")
            time_display = it.get("date", "-")
            if modified_iso and "T" in str(modified_iso):
                try:
                    dt_obj = datetime.fromisoformat(modified_iso.replace("Z", "+00:00"))
                    time_display = dt_obj.strftime("%H:%M:%S") + f" ({it.get('date', '')})"
                except ValueError:
                    try:
                        time_part = str(modified_iso).split("T")[1][:8]
                        time_display = f"{time_part} ({it.get('date', '-')})"
                    except:
                        pass
            time_item = QTableWidgetItem(time_display)
            time_item.setData(Qt.UserRole, modified_iso)
            self.table.setItem(r, 0, time_item)
            self.table.setItem(r, 1, QTableWidgetItem(it.get("display_name") or "-"))
            self.table.setItem(r, 2, QTableWidgetItem(it.get("file") or it.get("filename", "-")))
            size_mb = it.get('size_mb', 0.0)
            size_item = QTableWidgetItem(f"{size_mb:.1f}")
            size_item.setData(Qt.UserRole, it.get('size_bytes', 0))
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 3, size_item)
            thumb_label = QLabel("film")
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setStyleSheet("background: #2a3f54; border: 1px solid #3a4f6a; border-radius: 6px;")
            thumb_label.setFixedSize(88, 50)
            self.table.setCellWidget(r, 4, thumb_label)
            btn_play = QPushButton("Play")
            btn_play.setProperty("row", r)
            btn_play.clicked.connect(self._play_row)
            btn_play.setFixedSize(65, 32)
            btn_play.setToolTip(f"Play {it.get('file')}")
            self.table.setCellWidget(r, 5, btn_play)
            btn_dl = QPushButton("Download")
            btn_dl.setProperty("row", r)
            btn_dl.clicked.connect(self._download_row)
            btn_dl.setFixedSize(65, 32)
            btn_dl.setToolTip(f"Download {it.get('file')}")
            self.table.setCellWidget(r, 6, btn_dl)

        # เปิด sorting กลับ
        self.table.setSortingEnabled(is_sorting_enabled)

        # อัปเดต UI
        self.lbl_count.setText(f"{total} item(s)")
        self.lbl_page.setText(f"Page {self._page}/{pages}")
        self.btn_prev_page.setEnabled(self._page > 1)
        self.btn_next_page.setEnabled(self._page < pages)

    def _goto_page(self, p: int):
        total_pages = max(1, (len(self._view_items) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self._page = max(1, min(p, total_pages))
        self._refresh_page()

    def _row_item(self, row: int) -> Optional[dict]:
        if 0 <= row < len(self._files):
            return self._files[row]
        return None

    def _cell_clicked(self, row: int, col: int):
        if col not in (5, 6):
            self._preview_row(row)

    def _preview_row(self, row: int):
        item_data = self._row_item(row)
        if not item_data:
            self._clear_info_panel()
            if hasattr(self.player, 'stop'): self.player.stop()
            if hasattr(self, 'detection_list') and self.detection_list:
                self.detection_list.clear()
                self._current_video_start_dt = None
            return

        self._current_item = item_data
        url = self._build_stream_url(item_data)

        self.lbl_disp.setText(item_data.get("display_name") or "-")
        self.lbl_file.setText(item_data.get("file") or item_data.get("filename", "-"))
        self.lbl_url.setText(url)
        self.lbl_size.setText(f"{item_data.get('size_mb', 0):.1f} MB")
        self.btn_copy_url.setEnabled(bool(url))
        self.btn_open_external.setEnabled(bool(url))
        self.btn_download.setEnabled(True)
        try:
            self._current_video_start_dt = _parse_video_start_time(
                item_data.get("date"),
                item_data.get("file")
            )
            logger.info(f"Video start time calculated: {self._current_video_start_dt}")
        except Exception as e:
            self._current_video_start_dt = None
            logger.error(f"Failed to parse video start time: {e}")
        try:
            logger.info(f"Setting QMediaPlayer URL to: {url}")
            self.player.set_media(url)
            logger.info(f"Preview loaded request sent for: {item_data.get('display_name', 'Unknown')}")
            self._fetch_detections_for_item(item_data)
        except Exception as e:
            logger.error(f"Error setting media in player: {e}", exc_info=True)
            QMessageBox.warning(self, "Player Error", f"Cannot load preview:\n{str(e)}")

    def _fetch_detections_for_item(self, item_data: dict):
        if not hasattr(self, 'detection_list'):
             return
        self.detection_list.clear()
        self.detection_list.addItem("🔄 Loading detections...")

        filename = item_data.get("file") or item_data.get("filename")
        camera = item_data.get("camera", self.camera_name)
        zone = item_data.get("zone", self.zone)
        date = item_data.get("date", self._selected_date_str())

        if not filename or not self.api:
            self.detection_list.clear()
            self.detection_list.addItem("❌ Error: Missing file info or API.")
            return

        def _fetch_task():
            try:
                detections = self.api.get_detections_for_file(filename, camera, zone, date)
                
                self.detection_list.clear() 

                if not detections:
                    self.detection_list.addItem("ℹ️ No detections found in this video.")
                    return

                for det in detections:
                    type_icon = "👤" if det.get("type") == "face" else "🚗"
                    name = det.get("full_name") or det.get("plate") or "Unknown"
                    ts = det.get("timestamp", "")
                    time_str = ""
                    
                    try:
                        if "T" in ts:
                            time_str = ts.split("T")[-1].split(".")[0]
                        elif " " in ts:
                            time_str = ts.split(" ")[-1]
                    except Exception:
                        pass
                    
                    item_text = f"{type_icon} [{time_str}] {name}"
                    item = QListWidgetItem(item_text)
                    # (เก็บ ISO Timestamp เต็มๆ ไว้ใน UserRole)
                    item.setData(Qt.UserRole, det.get("timestamp")) 
                    
                    # ⭐️ [แก้ไข] ⭐️
                    # (เหลือ addItem ไว้แค่บรรทัดเดียว)
                    self.detection_list.addItem(item)
            
            except Exception as e:
                logger.error(f"Failed to fetch detections for {filename}: {e}", exc_info=True)
                self.detection_list.clear()
                self.detection_list.addItem(f"❌ Error loading detections: {str(e)}")

        QTimer.singleShot(0, _fetch_task)
        
    def _play_row(self):
        sender_button = self.sender()
        if not sender_button: return
        row = sender_button.property("row")
        if row is not None and row >= 0:
             self._preview_row(row)
             if hasattr(self.player, 'play'):
                 QTimer.singleShot(50, self.player.play)

    def _download_row(self):
        sender_button = self.sender()
        if not sender_button: return
        row = sender_button.property("row")
        if row is not None and row >= 0:
            self._download_item(self._row_item(row))

    def _copy_url(self):
        url = self.lbl_url.text().strip()
        if url:
            QApplication.clipboard().setText(url)
            QMessageBox.information(self, "Copied", "URL copied to clipboard! 📋")

    def _open_external(self):
        url_str = self.lbl_url.text().strip()
        if url_str:
            if QDesktopServices.openUrl(QUrl(url_str)):
                logger.info(f"Opened external URL: {url_str}")
            else:
                logger.error(f"Failed to open external URL: {url_str}")
                QMessageBox.warning(self, "Error", f"Could not open URL:\n{url_str}")

    def _download_current(self):
        if not self._current_item:
            QMessageBox.information(self, "Download", "Please select a file first.")
            return
        self._download_item(self._current_item)

    def _download_item(self, it: Optional[dict]):
        if not it or not self.api:
            QMessageBox.warning(self, "Download Error", "Cannot download: Missing item data or API client.")
            return
        filename = it.get("file") or it.get("filename", "recording.mp4")
        default_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video File", os.path.join(default_dir, filename),
            "MP4 Files (*.mp4);;All Files (*)"
        )
        if not save_path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dept = it.get("department", "")
            zone = it.get("zone", self.zone)
            cam = it.get("camera", self.camera_name)
            file = it.get("file") or it.get("filename", "")
            date = it.get("date", self._selected_date_str())
            if not all([dept, zone, cam, file, date]):
                raise ValueError("Missing required information (dept, zone, camera, file, date) for download.")
            self.api.download_recording(dept, zone, cam, file, save_path, date=date)
            QMessageBox.information(self, "Download Complete", f"✅ Video saved successfully to:\n{save_path}")
        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Download Error", f"Failed to download file:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    @pyqtSlot(QListWidgetItem)
    def _on_detection_clicked(self, item: QListWidgetItem):
        """
        (SLOT) เมื่อคลิกที่ Detection 
        -> คำนวณ Milliseconds และสั่ง Player ให้ Seek
        """
        detection_ts_str = item.data(Qt.UserRole)
        
        if not detection_ts_str:
            logger.warning("Clicked detection has no timestamp data.")
            return
            
        if not self._current_video_start_dt:
            logger.warning("Video start time is not set. Cannot seek.")
            return
            
        try:
            # 1. แปลงเวลา Detection (Aware)
            # (เราต้อง import datetime as dt หรือแก้ไขตรงนี้)
            # (ถ้าใช้ from datetime import datetime, date)
            detection_dt = datetime.fromisoformat(detection_ts_str)
            
            # 2. คำนวณส่วนต่าง (เป็น Milliseconds)
            offset_sec = (detection_dt - self._current_video_start_dt).total_seconds()
            offset_ms = int(offset_sec * 1000)

            if offset_ms < 0:
                logger.warning(f"Seek failed: Detection time {detection_dt} is before video start {self._current_video_start_dt}")
                offset_ms = 0 # (ไปที่วินาทีที่ 0 แทน)
                
            # 3. สั่ง Player ให้ Seek
            if hasattr(self.player, 'player') and self.player.player.isSeekable():
                self.player.player.setPosition(offset_ms)
                logger.info(f"Seeking to detection at {offset_ms} ms (Offset: {offset_sec:.2f}s)")
                
                # (ถ้า Player หยุดอยู่ ให้ Play)
                if self.player.player.state() != QMediaPlayer.PlayingState:
                    self.player.play()
            else:
                logger.warning("Player is not seekable.")
                
        except Exception as e:
            logger.error(f"Error seeking to detection: {e}", exc_info=True)
    def closeEvent(self, event):
        logger.info("Closing RecordingsDialog, stopping player...")
        if hasattr(self.player, 'stop') and callable(self.player.stop):
             self.player.stop()
        super().closeEvent(event)
# ===============================
#   Camera Stream Tile (MODIFIED for Grid)
# ===============================
class CameraStreamTile(QWidget):
    doubleClicked = pyqtSignal()

    def __init__(self, api: APIClient, camera_name: str, parent=None):
        super().__init__(parent)
        self.api = api
        self.camera_name = camera_name
        self.mode = "sub"
        self.ws_player: Optional[MJPGWebSocketPlayer] = None

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet(
            "QLabel { background-color: #000; border: 2px solid #0d1b2a; border-radius: 10px; }"
        )
        self.video.installEventFilter(self)

        self.title = QLabel(f"📹 {camera_name}")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-weight:bold; color:#E6EDF3; margin-top:4px;")

        self.btn_toggle = QPushButton("Main/Sub")
        self.btn_toggle.setFixedHeight(26)
        self.btn_toggle.clicked.connect(self.toggle_quality)

        lay = QVBoxLayout(self)
        lay.addWidget(self.video, 1)
        lay.addWidget(self.title)
        lay.addWidget(self.btn_toggle)
        lay.setSpacing(6)

        QTimer.singleShot(10, self.start_play)

    def eventFilter(self, source, event):
        if source is self.video and event.type() == QEvent.MouseButtonDblClick:
            self.doubleClicked.emit()
            return True
        return super().eventFilter(source, event)

    def start_play(self):
        self.stop_play()
        try:
            self.api.set_preview_mode(self.camera_name, self.mode)
        except Exception as e:
            logger.warning(f"Failed to set preview mode: {e}")

        try:
            cams = self.api.list_cameras()
            cam = next((c for c in cams if c.get("camera_name") == self.camera_name), None)
            if cam and cam.get("url2") and self.mode == "sub":
                logger.info(f"Using url2 for {self.camera_name}")
        except Exception:
            pass

        self.ws_player = MJPGWebSocketPlayer(self.api, self.camera_name, self._on_frame)
        self.ws_player.start()

    def stop_play(self):
        if self.ws_player:
            try: self.ws_player.stop()
            except Exception: pass
        self.ws_player = None
        pix = QPixmap(self.video.size())
        pix.fill(Qt.black)
        self.video.setPixmap(pix)

    def toggle_quality(self):
        self.mode = "sub" if self.mode == "main" else "main"
        self.title.setText(f"{self.camera_name} ({self.mode.upper()})")
        self.start_play()

    def _on_frame(self, rgb_np, text_msg=None):
        if text_msg: self.title.setToolTip(text_msg); return
        if rgb_np is None: return
        h, w, ch = rgb_np.shape
        img = QImage(rgb_np.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def closeEvent(self, e: QCloseEvent):
        try:
            self.stop_play()
        except Exception:
            pass
        super().closeEvent(e)

# ===============================
#   Main Window (UI ENHANCED)
# ===============================
class DeepBlueGridUltimate(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api = APIClient(CONFIG["SERVER_BASE"])
        self.setWindowTitle("Deep Blue CCTV Client - Ultimate Grid Edition")
        self.setMinimumSize(1300, 800)
        self.open_dialogs: Dict[str, QDialog] = {}
        if os.path.exists(LOGO_PATH):
            icon = QIcon(LOGO_PATH)
            self.setWindowIcon(icon)
            QApplication.setWindowIcon(icon)
        
        self.maximized_tile: Optional[CameraStreamTile] = None
        self.green_icon = self._create_status_icon(QColor(80, 200, 80)) 
        self.red_icon = self._create_status_icon(QColor(220, 50, 50)) 
        self.gray_icon = self._create_status_icon(QColor(100, 100, 100)) 
        
        self.ui_ws_client: Optional[UIWebSocketClient] = None

        # อย่าสร้างปุ่ม Admin/Segment ที่นี่
        self.btn_segment = None
        self.act_admin = None
        self.btn_admin = None

        self._apply_styles()
        self._build_ui()
        self.employee_count_label = QLabel("Employees: (Loading...)")
        self.employee_count_label.setStyleSheet("color: #E6EDF3; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.employee_count_label)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0b1726; }
            QSplitter::handle { background: #112240; }
            QToolBar { background-color: #112240; color: #E6EDF3; padding: 6px; spacing: 10px; border-bottom: 1px solid #2a3f54; }
            QToolButton { background: transparent; color: #E6EDF3; padding: 5px; qproperty-iconSize: 20px; }
            QToolButton:hover { background: #2a3f54; border-radius: 4px; }
            QLabel { color: #E6EDF3; }
            QLineEdit { border: 1px solid #2a3f54; border-radius: 6px; padding: 8px; background-color: #1b263b; color: #E6EDF3; }
            QLineEdit:focus { border: 1px solid #00aaff; }
            QListWidget { border: 1px solid #2a3f54; background-color: #1b263b; color: #E6EDF3; }
            QListWidget::item { padding: 8px; }
            QListWidget::item:hover { background-color: #2a3f54; }
            QListWidget::item:selected { background-color: #0077cc; }
            QPushButton { background-color: #0077cc; color: #FFFFFF; border-radius: 6px; padding: 8px 12px; border: none; font-weight: bold; }
            QPushButton:hover { background-color: #0088ee; }
            QPushButton#clearButton { background-color: #4a5b6c; }
            QPushButton#clearButton:hover { background-color: #5a6b7c; }
            QStatusBar { color: #B6C2CF; }
            QHeaderView::section { background-color: #112240; color: #E6EDF3; padding: 5px; border: 1px solid #2a3f54; }

            /* แก้ตรงนี้: ข้อความใน dialog อ่านง่าย */
            QMessageBox QLabel {
                color: #FFFFFF !important;
                font-weight: bold;
                font-size: 14px;
                padding: 6px;
            }
            QMessageBox QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #005a99;
            }
        """)

    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.toolBar = tb

        style = self.style()
        self.act_refresh    = QAction(style.standardIcon(QStyle.SP_BrowserReload), "Refresh Cameras", self)
        self.act_reports    = QAction(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Reports", self)
        self.act_recordings = QAction(style.standardIcon(QStyle.SP_DriveDVDIcon), "Recordings", self)
        self.act_logout     = QAction(style.standardIcon(QStyle.SP_DialogCloseButton), "Logout", self)

        def add_btn(action):
            btn = QToolButton()
            btn.setDefaultAction(action)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            tb.addWidget(btn)
            return btn

        add_btn(self.act_refresh)
        tb.addSeparator()
        add_btn(self.act_reports)
        add_btn(self.act_recordings)

        # ไม่สร้าง Admin/Segment ที่นี่ → รอ _rebuild_toolbar()

        add_btn(self.act_logout)

        # --- UI อื่น ---
        splitter = QSplitter()
        splitter.setHandleWidth(6)
        self.setCentralWidget(splitter)

        left = QWidget()
        lv = QVBoxLayout(left)
        left.setContentsMargins(10, 10, 10, 10)

        self.lbl_user = QLabel("Welcome, Guest")
        self.lbl_user.setStyleSheet("font-size:16px; font-weight:600; color:#E6EDF3;")
        lv.addWidget(self.lbl_user)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search Cameras...")
        lv.addWidget(self.search)

        self.comp_filter_cb = QComboBox()
        self.comp_filter_cb.setStyleSheet("""
            QComboBox { border: 1px solid #2a3f54; border-radius: 6px; padding: 8px; background-color: #1b263b; color: #E6EDF3; }
            QComboBox QAbstractItemView { background: #1b263b; color: #E6EDF3; selection-background-color: #2a3f54; }
        """)
        lv.addWidget(self.comp_filter_cb)

        self.cam_list = QListWidget()
        self.cam_list.setSelectionMode(QAbstractItemView.NoSelection)
        lv.addWidget(self.cam_list, 1)

        hl = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All (≤9)")
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("clearButton")
        hl.addWidget(self.btn_select_all)
        hl.addWidget(self.btn_clear)
        lv.addLayout(hl)

        self.btn_apply = QPushButton("Apply Selection")
        self.btn_apply.setFixedHeight(34)
        lv.addWidget(self.btn_apply)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.grid_wrap = QWidget()
        self.grid = QGridLayout(self.grid_wrap)
        self.grid.setSpacing(10)
        self.grid.setContentsMargins(6, 6, 6, 6)
        rv.addWidget(self.grid_wrap, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 960])

        self.statusBar().showMessage("Ready")

        # Connect
        self.act_refresh.triggered.connect(self.load_cameras)
        self.btn_select_all.clicked.connect(self.select_all_9)
        self.btn_clear.clicked.connect(self.clear_checks)
        self.btn_apply.clicked.connect(self.apply_selection)
        self.search.textChanged.connect(self._filter_list)
        self.comp_filter_cb.currentIndexChanged.connect(self._filter_list)
        self.act_logout.triggered.connect(self.logout)
        self.act_reports.triggered.connect(self.show_reports)
        self.act_recordings.triggered.connect(self.show_recordings)

        QTimer.singleShot(50, self._ensure_login)

    def _rebuild_toolbar(self):
        """รีบิลด์ toolbar ใหม่ตาม is_admin – แก้ removeWidget error"""
        tb = self.toolBar
        if not tb:
            return

        # ลบ action ทั้งหมดก่อน
        for action in list(tb.actions()):
            if action != self.act_logout:
                tb.removeAction(action)

        # ลบ widget ผ่าน layout
        layout = tb.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget and widget.defaultAction() != self.act_logout:
                    widget.setParent(None)
                    widget.deleteLater()

        style = self.style()
        def add_btn(action):
            btn = QToolButton()
            btn.setDefaultAction(action)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            tb.addWidget(btn)
            return btn

        # ปุ่มหลัก
        add_btn(self.act_refresh)
        tb.addSeparator()
        add_btn(self.act_reports)
        add_btn(self.act_recordings)

        is_admin = bool(getattr(self.api, "is_admin", False))

        if is_admin:
            if not self.btn_segment:
                self.btn_segment = QToolButton()
                self.btn_segment.setText("Segment: 1 min")
                self.btn_segment.setIcon(style.standardIcon(QStyle.SP_FileDialogContentsView))
                self.btn_segment.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                self.btn_segment.setPopupMode(QToolButton.InstantPopup)
                self.btn_segment.setMenu(self._create_segment_menu())
            tb.addWidget(self.btn_segment)
            tb.addSeparator()

            if not self.act_admin:
                self.act_admin = QAction(style.standardIcon(QStyle.SP_ComputerIcon), "Admin", self)
                self.act_admin.triggered.connect(self.show_admin)
            if not self.btn_admin:
                self.btn_admin = add_btn(self.act_admin)
            else:
                tb.addWidget(self.btn_admin)
            tb.addSeparator()
        else:
            if self.btn_segment:
                self.btn_segment.setParent(None)
                self.btn_segment = None
            if self.btn_admin:
                self.btn_admin.setParent(None)
                self.btn_admin = None
            if self.act_admin:
                self.act_admin = None

        add_btn(self.act_logout)
        logger.info(f"Toolbar rebuilt. Admin: {'Yes' if is_admin else 'No'}")

    def _ensure_login(self):
        dlg = DeepBlueLoginDialog(self.api, self)
        if dlg.exec_() == QDialog.Accepted:
            self.lbl_user.setText(f"Welcome, {self.api.username or 'User'}")
            self._rebuild_toolbar()  # เรียกใหม่ ไม่ใช้ _apply_role_permissions
            self.load_cameras()
            self._start_ui_websocket()
        else:
            QApplication.quit()

    def logout(self):
        self._clear_grid_tiles()
        self.cam_list.clear()
        try:
            self.api.logout()
        except Exception:
            pass
        self.lbl_user.setText("Welcome, Guest")
        if hasattr(self, "employee_count_label"):
            self.employee_count_label.setText("Employees: 0")
        if hasattr(self, "ui_ws_client") and self.ui_ws_client:
            self.ui_ws_client.stop()

        self._rebuild_toolbar()  # ซ่อน Admin/Segment

        for dialog in list(self.open_dialogs.values()):
            dialog.close()
        self.open_dialogs.clear()

        QMessageBox.information(self, "Logout", "Logout Clear")
        self._ensure_login()

    def show_admin(self):
        if not bool(getattr(self.api, "is_admin", False)):
            QMessageBox.warning(self, "Access Denied", "คุณไม่มีสิทธิ์เข้าใช้งาน Admin Tools")
            return
        try:
            hub = AdminHub(self.api, self)
            hub.exec_()
            self.load_cameras()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _create_status_icon(self, color: QColor) -> QIcon:
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(3, 3, 10, 10)
        painter.end()
        return QIcon(pixmap)

    @pyqtSlot(dict)
    def on_health_status_update(self, status_dict: dict):
        self.statusBar().setToolTip(f"Last status update: {datetime.now().strftime('%H:%M:%S')}")
        for i in range(self.cam_list.count()):
            item = self.cam_list.item(i)
            if not item: continue
            cam_data = item.data(Qt.UserRole)
            if not isinstance(cam_data, dict): continue
            cam_name = cam_data.get("camera_name")
            if not cam_name: continue
            status = status_dict.get(cam_name, "UNKNOWN")
            if status == "OK":
                item.setIcon(self.green_icon)
                item.setToolTip("Status: OK (Running)")
            elif status == "DOWN":
                item.setIcon(self.red_icon)
                item.setToolTip("Status: DOWN (Connection Lost)")
            else:
                item.setIcon(self.gray_icon)
                item.setToolTip("Status: Unknown")

    def _start_ui_websocket(self):
        if self.ui_ws_client:
            self.ui_ws_client.stop()
        self.ui_ws_client = UIWebSocketClient(self.api)
        self.ui_ws_client.status_updated.connect(self.on_health_status_update)
        def on_connection_status(message):
            self.statusBar().showMessage("Real-time status: Connected" if not message else f"Real-time status: Disconnected ({message})")
        self.ui_ws_client.connection_lost.connect(on_connection_status)
        self.ui_ws_client.start()

    def _create_segment_menu(self) -> QMenu:
        menu = QMenu(self)
        for minutes in [1, 2, 3]:
            action = QAction(f"{minutes} minutes", self, triggered=lambda _, m=minutes: self._set_segment_time(m))
            menu.addAction(action)
        return menu

    def _set_segment_time(self, minutes: int):
        self.statusBar().showMessage(f"Setting segment time to {minutes} minutes...")
        QApplication.processEvents()
        try:
            result = self.api.set_segment_minutes(minutes)
            new_mins = result.get("new_segment_minutes", minutes)
            if self.btn_segment:
                self.btn_segment.setText(f"Segment: {new_mins} min")
            self.statusBar().showMessage(f"Segment time set to {new_mins} minutes.")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set segment time:\n{str(e)}")

    def _ensure_login(self):
        dlg = DeepBlueLoginDialog(self.api, self)
        if dlg.exec_() == QDialog.Accepted:
            self.lbl_user.setText(f"Welcome, {self.api.username or 'User'}")
            self._rebuild_toolbar()
            self.load_cameras()
            self._start_ui_websocket() # ⭐️ [เพิ่ม] เริ่ม WS หลัง Login
        else:
            QApplication.quit()
    
    def logout(self):
        self._clear_grid_tiles()
        self.cam_list.clear()
        try:
            self.api.logout()
        except Exception:
            pass

        self.lbl_user.setText("Welcome, Guest")
        if hasattr(self, "employee_count_label"):
            self.employee_count_label.setText("Employees: 0")
        if hasattr(self, "ui_ws_client") and self.ui_ws_client:
            self.ui_ws_client.stop()

        # ใช้แค่บรรทัดเดียว → ปลอดภัย + สะอาด
        self._rebuild_toolbar()  # ซ่อน Admin/Segment อย่างสมบูรณ์

        for dialog in list(self.open_dialogs.values()):
            dialog.close()
        self.open_dialogs.clear()

        QMessageBox.information(self, "Logout", "Logout Clear")
        self._ensure_login()
    @pyqtSlot(int)
    def _on_dialog_finished(self):
        sender_dialog = self.sender()
        if not isinstance(sender_dialog, QDialog):
            return
        
        dialog_key_to_remove = None
        for key, dialog_instance in self.open_dialogs.items():
            if dialog_instance is sender_dialog:
                dialog_key_to_remove = key
                break

        if dialog_key_to_remove and dialog_key_to_remove in self.open_dialogs:
            logger.info(f"Dialog '{dialog_key_to_remove}' closed and removed from references.")
            del self.open_dialogs[dialog_key_to_remove]
        else:
             logger.warning(f"Finished dialog {sender_dialog} not found in open_dialogs.")   
    def closeEvent(self, e: QCloseEvent):
            if hasattr(self, "ui_ws_client") and self.ui_ws_client:
                 self.ui_ws_client.stop() # ⭐️ [เพิ่ม] หยุด WS ตอนปิด
            
            try:
                for dialog in list(self.open_dialogs.values()):
                    dialog.close()
                self._clear_grid_tiles()
                if hasattr(self.api, 'logout'):
                    self.api.logout()
            except Exception as ex:
                logger.error(f"Error during closeEvent cleanup: {ex}")
            super().closeEvent(e)
    
    def load_cameras(self):
        self.cam_list.clear()
        self.comp_filter_cb.clear()
        try:
            cams = self.api.list_cameras()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cameras: {e}")
            return
        all_comps = sorted(list(set(c.get("comp") for c in cams if c.get("comp"))))
        self.comp_filter_cb.addItem("All Departments", None)
        for comp in all_comps:
            self.comp_filter_cb.addItem(comp, comp)
        for cam in cams:
            name = cam.get("camera_name") or cam.get("camera_code", "")
            item = QListWidgetItem(name)
            item.setIcon(self.gray_icon) # ⭐️ [เพิ่ม] ตั้งค่าไอคอนเริ่มต้น
            item.setToolTip("Status: Unknown") # ⭐️ [เพิ่ม]
            
            # --- ⭐️ [แก้ไข] ลบบรรทัดที่ซ้ำซ้อน ⭐️ ---
            # item.setData(Qt.UserRole, name) (บรรทัดนี้ถูกลบ)
            
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.cam_list.addItem(item)
            item.setData(Qt.UserRole, cam) # (เหลือบรรทัดนี้ไว้)
        try:
            employees = self.api.list_employees() 
            count = len(employees)
            self.employee_count_label.setText(f"👥 Employees: {count}")
        except Exception as e:
            logger.error(f"Failed to load employee count: {e}")
            self.employee_count_label.setText("👥 Employees: Error")
        self.statusBar().showMessage(f"Loaded {self.cam_list.count()} cameras.")
    
# ในคลาส DeepBlueGridUltimate

    def current_checked_names(self) -> List[str]:
        names = []
        for i in range(self.cam_list.count()):
            if self.cam_list.item(i).checkState() == Qt.Checked:
                cam_data = self.cam_list.item(i).data(Qt.UserRole)
                
                if isinstance(cam_data, dict):
                    name = cam_data.get("camera_name")
                    if name:
                        names.append(name)
                elif isinstance(cam_data, str):
                    names.append(cam_data)
        return names
    
    def select_all_9(self):
        for i in range(self.cam_list.count()):
            self.cam_list.item(i).setCheckState(Qt.Unchecked)
        for i in range(min(self.cam_list.count(), CONFIG["MAX_TILES"])):
            self.cam_list.item(i).setCheckState(Qt.Checked)
    
    def clear_checks(self):
        for i in range(self.cam_list.count()):
            self.cam_list.item(i).setCheckState(Qt.Unchecked)
    
    def _filter_list(self):
            search_text = self.search.text().strip().lower()
            selected_comp = self.comp_filter_cb.itemData(self.comp_filter_cb.currentIndex())
            
            for i in range(self.cam_list.count()):
                it = self.cam_list.item(i)
                
                cam_data = it.data(Qt.UserRole)
                
                if not isinstance(cam_data, dict):
                    it.setHidden(True)
                    continue

                cam_name = (cam_data.get("camera_name") or "").lower()
                name_match = (not search_text) or (search_text in cam_name)
                
                cam_comp = cam_data.get("comp")
                comp_match = (selected_comp is None) or (cam_comp == selected_comp)

                it.setHidden(not (name_match and comp_match))
    
    def apply_selection(self):
        if self.maximized_tile:
            self.handle_tile_double_click() 
        names = self.current_checked_names()
        if not names:
            QMessageBox.information(self, "Selection", "กรุณาเลือกกล้องอย่างน้อย 1 ตัว")
            return
        if len(names) > CONFIG["MAX_TILES"]:
            QMessageBox.warning(self, "Limit", f"เลือกได้สูงสุด {CONFIG['MAX_TILES']} ตัว")
            names = names[:CONFIG["MAX_TILES"]]
        self._clear_grid_tiles()
        row, col = 0, 0
        for name in names:
            tile = CameraStreamTile(self.api, name, self.grid_wrap)
            tile.doubleClicked.connect(self.handle_tile_double_click)
            self.grid.addWidget(tile, row, col)
            col += 1
            if col >= 3:
                col, row = 0, row + 1
            if row >= 3:
                break
        self.statusBar().showMessage(f"Showing {self.grid.count()} stream(s).")
    
    def _clear_grid_tiles(self):
        self.maximized_tile = None
        for i in reversed(range(self.grid.count())):
            w = self.grid.itemAt(i).widget()
            if isinstance(w, CameraStreamTile):
                w.stop_play()
            if w:
                w.setParent(None)
                w.deleteLater()
    
    def handle_tile_double_click(self):
        clicked_tile = self.sender()
        if not isinstance(clicked_tile, QWidget):
            return
            
        if self.maximized_tile:
            for i in range(self.grid.count()):
                widget = self.grid.itemAt(i).widget()
                if widget:
                    widget.show()
            self.maximized_tile = None
        else:
            self.maximized_tile = clicked_tile
            for i in range(self.grid.count()):
                widget = self.grid.itemAt(i).widget()
                if widget and widget != self.maximized_tile:
                    widget.hide()
                    
    def _show_dialog(self, dialog_key: str, dialog_class: type, *args, **kwargs):
        """Helper function to show or focus a non-modal dialog."""
        if dialog_key in self.open_dialogs:
            dialog = self.open_dialogs[dialog_key]
            dialog.raise_() # Bring window to front
            dialog.activateWindow() # Activate it
            logger.info(f"Bringing existing '{dialog_key}' dialog to front.")
        else:
            try:
                # สร้าง Dialog instance ใหม่
                # ส่ง self (QMainWindow) เป็น parent ให้ Dialog
                dialog = dialog_class(self, *args, **kwargs)

                # เก็บ Reference ไว้ใน Dictionary
                self.open_dialogs[dialog_key] = dialog

                # เชื่อมต่อ signal 'finished' เพื่อ cleanup ตอนปิด
                # finished signal ส่ง int (result code), ใช้ pyqtSlot(int) รับ
                dialog.finished.connect(self._on_dialog_finished)

                # ตั้งค่า Attribute ให้ลบตัวเองเมื่อปิด (สำคัญมากสำหรับ non-modal)
                dialog.setAttribute(Qt.WA_DeleteOnClose)

                # แสดงหน้าต่างแบบ Non-modal
                dialog.show()
                logger.info(f"Opened new '{dialog_key}' dialog.")

            except Exception as e:
                logger.error(f"Error opening dialog '{dialog_key}': {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Could not open {dialog_key} window:\n{str(e)}")
                # ลบ Reference ถ้าสร้างแล้ว Error ก่อน Show
                if dialog_key in self.open_dialogs:
                    del self.open_dialogs[dialog_key]
                    
    def show_reports(self):
        try:
            ReportsDialog(self).exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def show_recordings(self):
        try:
            cams = self.api.list_cameras()
            pick = SelectCameraDialog(cams, self)
            if pick.exec_():
                cam = pick.get_selected_camera()
                if cam:
                    dlg = RecordingsDialog(self, camera_name=cam.get("camera_name"), zone=cam.get("zone", "building"))
                    dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def show_admin(self):
        try:
            hub = AdminHub(self.api, self)
            hub.exec_()
            self.load_cameras()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    

# ===============================
#   Splash + App Entry (UI ENHANCED)
# ===============================
class SplashScreen(QSplashScreen):
    def __init__(self):
        pixmap = QPixmap(500, 300)
        pixmap.fill(QColor("#0d1b2a"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if os.path.exists(LOGO_PATH):
            logo_pix = QPixmap(LOGO_PATH).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(int(pixmap.width()/2 - logo_pix.width()/2), 40, logo_pix)
        
        painter.setPen(QColor("#00aaff"))
        painter.setFont(QFont("Segoe UI", 20, QFont.Bold))
        painter.drawText(pixmap.rect().adjusted(0, 100, 0, 0), Qt.AlignCenter, "Deep Blue CCTV Client")
        
        # ✅ เพิ่ม version/subtitle
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor("#90a4ae"))
        painter.drawText(pixmap.rect().adjusted(0, 140, 0, 0), Qt.AlignCenter, "Ultimate Grid Edition")
        
        painter.end()

        super().__init__(pixmap)
        
        # ✅ แสดง loading message แบบ animated
        self.showMessage("Initializing Client...", Qt.AlignCenter | Qt.AlignBottom, QColor("#e0e6ed"))
        
        # ✅ Animated dots
        self.dot_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_dots)
        self.timer.start(500)

    def _update_dots(self):
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count
        self.showMessage(f"Loading{dots}", Qt.AlignCenter | Qt.AlignBottom, QColor("#e0e6ed"))

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)

def main():

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    
    # ✅ ตั้งค่าไอคอนแอพทั้งระบบ
    if os.path.exists(LOGO_PATH):
        app_icon = QIcon(LOGO_PATH)
        app.setWindowIcon(app_icon)
    
    # ✅ ปรับ high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # ✅ ปรับ font rendering
    app.setFont(QFont("Segoe UI", 9))
    
    # ✅ สร้าง splash screen
    splash = SplashScreen()
    splash.show()
    
    # ✅ ซ่อน splash หลัง 2 วินาที
    QTimer.singleShot(2000, splash.close)
    
    # ✅ สร้าง main window
    window = DeepBlueGridUltimate()
    window.show()
    
    # ✅ รอ splash จบก่อนแสดง main window เต็ม
    QTimer.singleShot(2200, lambda: splash.finish(window))
    
    sys.exit(app.exec_())
if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        print(f"Created directory: {ASSETS_DIR}. Please add your 'logo.png' there.")

    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        QMessageBox.critical(None, "Fatal Error", str(e))