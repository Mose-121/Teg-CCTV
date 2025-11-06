import os
import re
import cv2
import bcrypt
import logging
import numpy as np
import psycopg2
from typing import Optional, Tuple, List, Dict, Any
from psycopg2 import Binary
from psycopg2.extras import Json
import datetime as dt
import pytz
# ---------------- load .env (optional) ----------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

# (optional) คำต้องห้ามถ้ามีในโปรเจกต์
try:
    from service import utils
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False

# ================= DB helper: read config from env =================
def _get_db_config_from_env() -> Dict[str, Any]:
    """
    อ่านค่าการเชื่อมต่อ DB จาก environment variables หรือ DATABASE_URL (ถ้ามี)
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        return {"dsn": db_url}

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    sslmode = os.getenv("DB_SSLMODE", "")

    cfg: Dict[str, Any] = {
        "host": host,
        "port": int(port) if port.isdigit() else port,
        "dbname": name,
        "user": user,
    }
    if password:
        cfg["password"] = password
    if sslmode:
        cfg["sslmode"] = sslmode
    return cfg

# =============== Normalizers / Validators ===============
_THAI2ARABIC = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_PLATE_RE = re.compile(r"^[A-Za-zก-ฮ]{1,3}[-\s]?\d{1,6}$")
_STATUS_TO_DIRECTION = {
    "IN": "IN", "ENTER": "IN", "TRUE": "IN", "1": "IN",
    "OUT": "OUT", "EXIT": "OUT", "FALSE": "OUT", "0": "OUT",
}

def _norm_plate(s: str) -> str:
    s = "" if not s else str(s).replace(" ", "").replace("\u200b", "").translate(_THAI2ARABIC).upper().strip()
    return s

def _plausible_plate(s: str) -> bool:
    return bool(_PLATE_RE.match(_norm_plate(s)))

def _limit(s: Optional[str], n: int) -> str:
    return ("" if s is None else str(s))[:n]

def _norm_direction(s: Optional[str]) -> str:
    s = (s or "").strip().upper()
    return _STATUS_TO_DIRECTION.get(s, "IN")

def _l2norm(x: np.ndarray) -> np.ndarray:
    """L2 normalize; ป้องกัน NaN/Inf ด้วย eps"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n <= 0.0:
        logger.warning("[_l2norm] invalid norm, returning zeros")
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32)

# mapping hint -> slot
VIEW_TO_SLOT = {
    "center": 1, "front": 1, "straight": 1,
    "left":   2,
    "right":  3,
}


class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None
        try:
            db_cfg = _get_db_config_from_env()
            if "dsn" in db_cfg:
                self.conn = psycopg2.connect(db_cfg["dsn"])
            else:
                self.conn = psycopg2.connect(**db_cfg)

            self.cursor = self.conn.cursor()
            logger.info("[DB] Connected to PostgreSQL database")
            print("[DB] Connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error(f"[DB ERROR] Connection failed: {e}")
            print(f"[DB ERROR] Connection failed: {e}")
            self.conn = None
            self.cursor = None

    # ---------------- Auth ----------------
    def check_user(self, username: str, password: str) -> Tuple[bool, Optional[str], List[str], bool]:
        if not self.conn:
            return False, None, [], False
        try:
            with self.conn:
                
                # --- ⭐️ [แก้ไข 1/2] ⭐️ ---
                # (เพิ่ม 'is_admin' เข้าไปใน SELECT)
                query = """
                    SELECT user_name, pass_user, department, access, COALESCE(is_admin, FALSE) 
                    FROM login 
                    WHERE user_name = %s
                """
                self.cursor.execute(query, (username,))
                user = self.cursor.fetchone()
                
                if user and bcrypt.checkpw(password.encode("utf-8"), str(user[1]).encode("utf-8")):
                    # (user[0]=user_name, 1=pass_user, 2=department, 3=access, 4=is_admin)
                    
                    access = user[3] if user[3] is not None else []
                    if isinstance(access, str):
                        access = [x.strip() for x in access.strip("{}").split(",") if x.strip()]
                        
                    # --- ⭐️ [แก้ไข 2/2] ⭐️ ---
                    # (อ่านค่า 'is_admin' (user[4]) จาก DB โดยตรง)
                    is_admin = bool(user[4]) 
                    
                    return True, user[2], access, is_admin
                    
                return False, None, [], False
                
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("check_user", str(e), None)
            return False, None, [], False

    # ---------------- Cameras ----------------
    def get_cameras(self, allowed_departments: Optional[List[str]] = None):
        if not self.conn:
            return []

        try:
            if allowed_departments is None:
                # Admin
                self.cursor.execute("""
                    SELECT camera_name, url, zone, comp, url2
                    FROM cameras
                    ORDER BY camera_name
                """)
            else:
                # Non-admin: ใช้ LIKE ANY
                patterns = [f"%{d.strip().upper()}%" for d in allowed_departments if d.strip()]
                if not patterns:
                    return []
                self.cursor.execute("""
                    SELECT camera_name, url, zone, comp, url2
                    FROM cameras
                    WHERE UPPER(comp) LIKE ANY(%s)
                    ORDER BY camera_name
                """, (patterns,))

            rows = self.cursor.fetchall()
            return [dict(zip(["camera_name", "url", "zone", "comp", "url2"], r)) for r in rows]

        except Exception as e:
            self.log_error("get_cameras", str(e), None)
            return []
    def add_camera(self, camera_name: str, url: str, url2: Optional[str] = None, zone: str = "face", comp: Optional[str] = None) -> Optional[int]:
        try:
            # ใช้ url เป็นค่าเริ่มต้นสำหรับ url2 หาก url2 เป็น None
            url2 = url if url2 is None else url2
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cameras (camera_name, url, url2, zone, comp, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (camera_name) DO NOTHING
                    RETURNING id
                    """,
                    (camera_name, url, url2, zone, comp)  # ✅ ลบ enter ออก
                )
                result = cur.fetchone()
                if result:
                    self.conn.commit()
                    return result[0]
                return None
        except Exception as e:
            logger.error(f"[DB] Failed to add camera {camera_name}: {e}")
            self.conn.rollback()
            raise

    def update_camera(self, camera_name: str, patch: dict) -> bool:
        if not self.conn:
            logger.error(f"[DB] No database connection for update_camera: {camera_name}")
            return False
        try:
            self.cursor.execute("SELECT COUNT(*) FROM cameras WHERE camera_name=%s", (camera_name,))
            if (self.cursor.fetchone() or (0,))[0] == 0:
                logger.warning(f"[DB] Camera {camera_name} not found")
                return False
            set_clause = ", ".join(f"{k}=%s" for k in patch)
            values = list(patch.values()) + [camera_name]
            self.cursor.execute(
                f"UPDATE cameras SET {set_clause} WHERE camera_name=%s",
                values
            )
            self.conn.commit()
            logger.info(f"[DB] Updated camera {camera_name} with patch: {patch}")
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("update_camera", str(e), camera_name)
            logger.error(f"[DB] Failed to update camera {camera_name}: {e}")
            raise

    def delete_camera(self, camera_name: str) -> bool:
        if not self.conn:
            return False
        try:
            with self.conn:
                self.cursor.execute("DELETE FROM cameras WHERE camera_name = %s", (camera_name,))
                self.conn.commit()
                return self.cursor.rowcount > 0
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("delete_camera", str(e), camera_name)
            return False

    # ---------------- Faces ----------------
    def load_known_faces(self):
        """
        (โหมดเดิม) ดึง embedding ทั้ง 5 ช่อง แล้ว 'เฉลี่ยเฉพาะช่องที่มีข้อมูลจริง'
        """
        if not self.conn:
            return []
        try:
            with self.conn:
                self.cursor.execute("""
                    SELECT emp_id, embedding, embedding2, embedding3, embedding4, embedding5, name, department
                    FROM face_embeddings
                """)
                results = self.cursor.fetchall()
                out = []
                for emp_id, e1, e2, e3, e4, e5, name, dept in results:
                    vecs = []
                    # ⭐️ วนลูป 5 ช่อง
                    for e in (e1, e2, e3, e4, e5):
                        if e and len(e) > 0:
                            arr = np.frombuffer(e, dtype=np.float32)
                            if arr.size == 512: vecs.append(arr)
                    if not vecs:
                        continue
                    mean_emb = np.mean(np.stack([_l2norm(v) for v in vecs], axis=0), axis=0)
                    emb = _l2norm(mean_emb)
                    out.append((emp_id, emb, name, dept))
                logger.info(f"[load_known_faces] Loaded {len(out)} persons (averaged from 5 slots)")
                return out
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("load_known_faces", str(e), None)
            return []

    def load_known_faces_per_slot(self) -> List[tuple]:
        """
        (โหมดใหม่สำหรับ per-image/per-slot matching)
        คืนรายการแบบ 'แตกช่อง' โดยไม่รวมเฉลี่ย:
            List[(emp_id: str, name: str, department: str, slot: int, emb_norm: np.ndarray(512,))]
        """
        if not self.conn:
            return []
        try:
            with self.conn:
                self.cursor.execute("""
                    SELECT emp_id, name, department, embedding, embedding2, embedding3, embedding4, embedding5
                    FROM face_embeddings
                """)
                results = self.cursor.fetchall()

                out: List[tuple] = []

                def _to_np_ok(b: Optional[bytes]) -> Optional[np.ndarray]:
                    if not b or len(b) == 0:
                        return None
                    arr = np.frombuffer(b, dtype=np.float32)
                    if arr.size != 512 or not np.isfinite(arr).all():
                        return None
                    return _l2norm(arr)

                for emp_id, name, dept, e1, e2, e3 in results:
                    name = name or emp_id
                    dept = dept or "Unknown"
                    for slot, eb in ((1, _to_np_ok(e1)), (2, _to_np_ok(e2)), (3, _to_np_ok(e3))):
                        if eb is not None:
                            out.append((emp_id, name, dept, slot, eb))

                persons = len({r[0] for r in out})
                logger.info(f"[load_known_faces_per_slot] Loaded {persons} persons, {len(out)} slots")
                return out

        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("load_known_faces_per_slot", str(e), None)
            return []

    def save_aligned_face(self, emp_or_name: str, aligned_jpg: Optional[bytes]) -> None:
        """ออปชันนอล: เก็บหน้า aligned ไว้ด้วย ถ้ามีตารางรองรับ"""
        if not self.conn or not aligned_jpg:
            return
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_aligned (emp_id_or_name, image_jpg, created_at)
                    VALUES (%s, %s, NOW())
                    """,
                    (emp_or_name, Binary(aligned_jpg)),
                )
        except Exception:
            pass

    def log_face_detection(
        self,
        camera_name: str,
        confidence: float,
        box_xyxy: tuple,
        full_name: str,
        department: Optional[str] = None,
        emp_id: Optional[int | str] = None,
        ts: Optional[str] = None,
        similarity: Optional[float] = None,
    ):
        """บันทึกเหตุการณ์รู้จำใบหน้า + เก็บ similarity ถ้ามีคอลัมน์"""
        if not self.conn:
            return
        try:
            with self.conn:
                self.cursor.execute("SELECT 1 FROM cameras WHERE camera_name = %s", (camera_name,))
                if self.cursor.fetchone() is None:
                    raise ValueError(f"กล้อง {camera_name} ไม่พบในตาราง cameras")

                dept = (department or "").strip() or "Unknown"
                emp_val: Optional[int | str] = None if emp_id in (None, "", "Unknown") else emp_id

                if (dept == "Unknown") or (emp_val is None):
                    if emp_val is not None:
                        self.cursor.execute("SELECT department FROM face_embeddings WHERE emp_id = %s", (emp_val,))
                        r = self.cursor.fetchone()
                        if r and dept == "Unknown":
                            dept = r[0] or "Unknown"
                    else:
                        self.cursor.execute(
                            "SELECT emp_id, department FROM face_embeddings WHERE name = %s LIMIT 1",
                            (full_name,))
                        r = self.cursor.fetchone()
                        if r:
                            emp_val = r[0]
                            if dept == "Unknown":
                                dept = r[1] or "Unknown"

                x1, y1, x2, y2 = map(int, box_xyxy)

                self.cursor.execute("""
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name='face_detection_details' AND column_name='similarity'
                """)
                has_similarity = self.cursor.fetchone() is not None

                if has_similarity:
                    sql = """
                        INSERT INTO face_detection_details
                            (camera_name, full_name, department, emp_id,
                             confidence, similarity,
                             box_x1, box_y1, box_x2, box_y2, timestamp)
                        VALUES
                            (%s, %s, %s, %s,
                             %s, %s,
                             %s, %s, %s, %s,
                             COALESCE(%s::timestamp, CURRENT_TIMESTAMP))
                    """
                    params = (
                        camera_name, full_name, dept, emp_val,
                        float(confidence), (None if similarity is None else float(similarity)),
                        x1, y1, x2, y2, None,
                    )
                else:
                    sql = """
                        INSERT INTO face_detection_details
                            (camera_name, full_name, department, emp_id,
                             confidence, box_x1, box_y1, box_x2, box_y2, timestamp)
                        VALUES
                            (%s, %s, %s, %s,
                             %s, %s, %s, %s, %s,
                             COALESCE(%s::timestamp, CURRENT_TIMESTAMP))
                    """
                    params = (
                        camera_name, full_name, dept, emp_val,
                        float(confidence), x1, y1, x2, y2, None,
                    )

                self.cursor.execute(sql, params)
                self.conn.commit()
                logger.info(f"[log_face_detection] OK: {camera_name} - {full_name} (sim={similarity})")

        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("log_face_detection", str(e), camera_name)
        except ValueError as e:
            self.conn.rollback()
            self.log_error("log_face_detection", str(e), camera_name)

    # ---------------- Errors ----------------
    def log_error(self, error_type: str, error_message: str,
                  camera_id: Optional[str] = None,
                  camera_code: Optional[str] = None,
                  zone: Optional[str] = None,
                  payload=None):
        if not self.conn:
            return
        try:
            self.cursor.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name='error_log'
            """)
            existing = {r[0] for r in self.cursor.fetchall()}
            cols, vals = ['error_type', 'error_message'], [error_type, error_message]
            if 'camera_id' in existing and camera_id is not None: cols.append('camera_id'); vals.append(camera_id)
            if 'camera_code' in existing and camera_code is not None: cols.append('camera_code'); vals.append(camera_code)
            if 'zone' in existing and zone is not None: cols.append('zone'); vals.append(zone)
            if 'payload' in existing and payload is not None:
                cols.append('payload'); vals.append(Json(payload))
            placeholders = ','.join(['%s']*len(vals))
            sql = f"INSERT INTO error_log ({', '.join(cols)}) VALUES ({placeholders})"
            self.cursor.execute(sql, vals)
            self.conn.commit()
        except psycopg2.Error as e:
            print(f"[DB ERROR] log_error: {e}")

    def check_car(self, plate_text: str):
        if not self.conn:
            return None
        try:
            with self.conn:
                query = "SELECT license, province FROM whitelist_car WHERE license = %s"
                self.cursor.execute(query, (plate_text,))
                return self.cursor.fetchone()
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("check_car", str(e), None)
            return None

    def save_record(self, plate: str, province: str, camera_name: str,
                    status: str, direction_or_status: str, ts=None) -> Optional[int]:
        """INSERT ลง car_log ให้ตรงสคีมา"""
        if not self.conn:
            return None
        try:
            if _HAS_UTILS and hasattr(utils, "is_banned_text") and utils.is_banned_text(plate, check_substring=True):
                raise ValueError(f"OCR_BANNED: '{plate}'")
        except Exception:
            pass

        plate_n = _norm_plate(plate)
        if not _plausible_plate(plate_n):
            raise ValueError(f"OCR_REJECTED_PATTERN: '{plate}' -> '{plate_n}'")

        province_n = _limit(province, 64)
        camera_n   = _limit(camera_name, 64)
        status_n   = _limit((status or "").upper(), 8)
        direction  = _norm_direction(direction_or_status)

        sql = """
            INSERT INTO car_log(plate_number, province, "timestamp", camera_name, status, direction)
            VALUES (%s, %s, COALESCE(%s, NOW()), %s, %s, %s)
            RETURNING id
        """
        with self.conn, self.conn.cursor() as cur:
            cur.execute(sql, (plate_n, province_n, ts, camera_n, status_n, direction))
            rid = cur.fetchone()[0]
            return rid

    @staticmethod
    def _pick_next_slot(row_has1: bool, row_has2: bool, row_has3: bool, 
                        row_has4: bool, row_has5: bool) -> int:
        """เลือกช่องว่างถัดไป: 1 -> 2 -> 3 -> 4 -> 5; ถ้าเต็มแล้ว return 0"""
        if not row_has1: return 1
        if not row_has2: return 2
        if not row_has3: return 3
        if not row_has4: return 4
        if not row_has5: return 5
        return 0
    def add_employee(self, emp_id: str, name: str, department: Optional[str],
                       image_data: bytes, embedding: bytes | np.ndarray,
                       view_hint: Optional[str] = None,
                       aligned_image_data: Optional[bytes] = None) -> bool:
        """
        [อัปเกรด 5 ช่อง] บันทึกพนักงาน/มุมใบหน้า ลง 'แถวเดียว'
        """
        if not self.conn:
            return False
        try:
            with self.conn:
                emb = (np.frombuffer(embedding, dtype=np.float32)
                       if isinstance(embedding, (bytes, bytearray))
                       else np.asarray(embedding, dtype=np.float32))
                expected_size = 512
                if emb.size != expected_size or not np.isfinite(emb).all():
                    logger.error(f"[add_employee] Invalid embedding size or values: {emb.size}")
                    return False

                emb = emb / (np.linalg.norm(emb) + 1e-9)
                if not np.any(emb):
                    logger.error("[add_employee] Invalid embedding after normalization (zeros)")
                    return False

                img_array = np.frombuffer(image_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    logger.error("[add_employee] Failed to decode image")
                    return False
                resized_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LANCZOS4)
                ok, buf = cv2.imencode(".jpg", resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    logger.error("[add_employee] Failed to encode resized image")
                    return False
                resized_image_blob = buf.tobytes()

                slot_from_hint = 0
                if view_hint:
                    slot_from_hint = VIEW_TO_SLOT.get(view_hint.strip().lower(), 0)

                self.cursor.execute("""
                    SELECT
                        (embedding  IS NOT NULL AND length(embedding) > 0) AS has1,
                        (embedding2 IS NOT NULL AND length(embedding2) > 0) AS has2,
                        (embedding3 IS NOT NULL AND length(embedding3) > 0) AS has3,
                        (embedding4 IS NOT NULL AND length(embedding4) > 0) AS has4,
                        (embedding5 IS NOT NULL AND length(embedding5) > 0) AS has5
                    FROM face_embeddings
                    WHERE emp_id = %s
                    FOR UPDATE
                """, (emp_id,))
                row = self.cursor.fetchone()

                emb_bin = Binary(emb.tobytes())
                aligned_bin = Binary(aligned_image_data) if aligned_image_data else None

                if row is None:
                    # (INSERT Case - พนักงานใหม่)
                    slot = slot_from_hint if slot_from_hint in (1, 2, 3, 4, 5) else 1
                    img1 = resized_image_blob if slot == 1 else b''
                    img2 = resized_image_blob if slot == 2 else b''
                    img3 = resized_image_blob if slot == 3 else b''
                    img4 = resized_image_blob if slot == 4 else b''
                    img5 = resized_image_blob if slot == 5 else b''
                    emb1 = emb_bin if slot == 1 else Binary(b'')
                    emb2 = emb_bin if slot == 2 else Binary(b'')
                    emb3 = emb_bin if slot == 3 else Binary(b'')
                    emb4 = emb_bin if slot == 4 else Binary(b'')
                    emb5 = emb_bin if slot == 5 else Binary(b'')
                    
                    self.cursor.execute("""
                        INSERT INTO face_embeddings
                          (emp_id, name, department,
                           image_data, image_data2, image_data3, image_data4, image_data5,
                           embedding,  embedding2,  embedding3,  embedding4,  embedding5,
                           aligned_image_data)
                        VALUES
                          (%s, %s, %s,
                           %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s,
                           %s)
                    """, (emp_id, name, department,
                          Binary(img1), Binary(img2), Binary(img3), Binary(img4), Binary(img5),
                          emb1, emb2, emb3, emb4, emb5,
                          aligned_bin))
                    self.conn.commit()
                    logger.info(f"[add_employee] INSERT {emp_id} -> slot{slot}")
                    return True

                # (UPDATE Case - เพิ่มรูปให้พนักงานเดิม)
                has1, has2, has3, has4, has5 = row
                slot = slot_from_hint if slot_from_hint in (1, 2, 3, 4, 5) else self._pick_next_slot(has1, has2, has3, has4, has5)
                
                if slot == 0:
                    logger.warning(f"[add_employee] Employee {emp_id} already has 5 slots filled")
                    # (Optional: อัปเดต aligned_image ถ้าส่งมา)
                    if aligned_bin is not None:
                        self.cursor.execute("UPDATE face_embeddings SET aligned_image_data = %s WHERE emp_id = %s", (aligned_bin, emp_id))
                        self.conn.commit()
                        logger.info(f"[add_employee] Slots full; updated aligned_image_data for {emp_id}")
                        return True
                    return False # ⭐️ คืนค่า False ถ้าเต็ม

                # (โค้ด UPDATE สำหรับ 5 ช่อง)
                common_sql = """
                    UPDATE face_embeddings
                    SET image_data{N} = %s,
                        embedding{N}  = %s,
                        name = COALESCE(name, %s),
                        department = COALESCE(department, %s),
                        aligned_image_data = COALESCE(%s, aligned_image_data)
                    WHERE emp_id = %s
                """
                params = (Binary(resized_image_blob), emb_bin, name, department, aligned_bin, emp_id)
                
                if slot == 1 and not has1:
                    self.cursor.execute(common_sql.format(N=""), params)
                elif slot == 2 and not has2:
                    self.cursor.execute(common_sql.format(N="2"), params)
                elif slot == 3 and not has3:
                    self.cursor.execute(common_sql.format(N="3"), params)
                elif slot == 4 and not has4:
                    self.cursor.execute(common_sql.format(N="4"), params)
                elif slot == 5 and not has5:
                    self.cursor.execute(common_sql.format(N="5"), params)
                else:
                    logger.warning(f"[add_employee] Requested slot{slot} for {emp_id} is already filled")
                    # (Optional: อัปเดต aligned_image ถ้าส่งมา แม้ว่าช่องจะเต็ม)
                    if aligned_bin is not None:
                         self.cursor.execute("UPDATE face_embeddings SET aligned_image_data = %s WHERE emp_id = %s", (aligned_bin, emp_id))
                         self.conn.commit()
                         logger.info(f"[add_employee] Slot occupied; updated aligned_image_data for {emp_id}")
                         return True
                    return False # ⭐️ คืนค่า False ถ้าช่องไม่ว่าง

                self.conn.commit()
                logger.info(f"[add_employee] UPDATE {emp_id} -> slot{slot}")
                return True

        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"[add_employee] error: {e}")
            return False

    def list_employees(self) -> List[dict]:
        """
        [ฟังก์ชันใหม่ - แก้ไขแล้ว 5 ช่อง] ดึงรายชื่อพนักงานทั้งหมด
        """
        sql = """
        SELECT 
            emp_id,
            name, 
            department,
            (CASE WHEN embedding  IS NOT NULL AND length(embedding) > 0 THEN 1 ELSE 0 END) + 
            (CASE WHEN embedding2 IS NOT NULL AND length(embedding2) > 0 THEN 1 ELSE 0 END) + 
            (CASE WHEN embedding3 IS NOT NULL AND length(embedding3) > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN embedding4 IS NOT NULL AND length(embedding4) > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN embedding5 IS NOT NULL AND length(embedding5) > 0 THEN 1 ELSE 0 END) as image_count
        FROM 
            face_embeddings 
        ORDER BY 
            name;
        """
        try:
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            
            return [
                {
                    "emp_id": r[0], 
                    "full_name": r[1],
                    "department": r[2], 
                    "image_count": r[3]
                } for r in rows
            ]
        except Exception as e:
            logger.error(f"Error listing employees: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return []
    def log_camera_event(self, camera_name: str, status: str, event_time: dt.datetime):
        """
        บันทึกเหตุการณ์สถานะกล้อง (OK หรือ DOWN)
        """
        if not self.conn:
            return
        
        sql = """
        INSERT INTO camera_status_events (event_time, camera_name, status)
        VALUES (%s, %s, %s);
        """
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute(sql, (event_time, camera_name, status))
        except Exception as e:
            logger.error(f"[log_camera_event] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()

    # --- ⭐️ [เพิ่มใหม่ 2/3] สำหรับดึง Event Log มาแสดง ⭐️ ---
    def get_camera_events(self, start_date: str, end_date: str) -> List[dict]:
        """
        ดึง Event Log ย้อนหลังตามช่วงวันที่
        """
        if not self.conn:
            return []
        
        # (เราจะแปลงเวลา +07:00 กลับเป็น String สวยๆ ใน SQL เลย)
        sql = """
        SELECT 
            TO_CHAR(event_time AT TIME ZONE 'Asia/Bangkok', 'YYYY-MM-DD HH24:MI:SS') as time_str,
            camera_name, 
            status
        FROM camera_status_events
        WHERE event_time BETWEEN %s::timestamptz AND %s::timestamptz
        ORDER BY event_time DESC;
        """
        try:
            with self.conn, self.conn.cursor() as cur:
                # (บวกเวลา 23:59:59 ให้ end_date)
                cur.execute(sql, (f"{start_date} 00:00:00+07:00", f"{end_date} 23:59:59+07:00"))
                rows = cur.fetchall()
                cols = ["time_str", "camera_name", "status"]
                return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.error(f"[get_camera_events] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return []

    # --- ⭐️ [เพิ่มใหม่ 3/3] สำหรับดึงสถานะล่าสุด (ใช้ตอน Server เปิด) ⭐️ ---
    def get_all_last_camera_statuses(self) -> Dict[str, str]:
        """
        ดึงสถานะ "ล่าสุด" ของกล้อง "ทุกตัว"
        (เพื่อป้องกันการ log ซ้ำตอน Server เปิดใหม่)
        """
        if not self.conn:
            return {}
        
        # (SQL นี้จะดึงเฉพาะแถวที่ "ล่าสุด" ของ "แต่ละ" camera_name)
        sql = """
        SELECT DISTINCT ON (camera_name)
            camera_name, status
        FROM camera_status_events
        ORDER BY camera_name, event_time DESC;
        """
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute(sql)
                # คืนค่าเป็น {"GATE4": "OK", "Office2": "DOWN"}
                return {row[0]: row[1] for row in cur.fetchall()}
        except Exception as e:
            logger.error(f"[get_all_last_camera_statuses] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return {}
    def get_timestamps_for_person(
        self, 
        start_dt: dt.datetime, 
        end_dt: dt.datetime, 
        camera_name: str, 
        person_name: str
    ) -> List[dt.datetime]:
        """
        ค้นหา "เวลา" ทั้งหมดที่คนๆ นี้ (จากชื่อ) ถูกตรวจพบบนกล้องนี้
        """
        if not self.conn:
            return []
            
        name_query = f"%{person_name.strip()}%"
        
        sql = """
        SELECT timestamp 
        FROM face_detection_details
        WHERE 
            camera_name = %s
            AND (full_name ILIKE %s OR emp_id ILIKE %s)
            AND timestamp BETWEEN %s AND %s;
        """
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute(sql, (camera_name, name_query, name_query, start_dt, end_dt))
                
                # --- ⭐️ [แก้ไข] ⭐️ ---
                # (บังคับให้ผลลัพธ์จาก DB มี Timezone +07:00 (Asia/Bangkok))
                bangkok_tz = pytz.timezone('Asia/Bangkok')
                return [bangkok_tz.localize(row[0]) for row in cur.fetchall()]
                # --- ⭐️ [สิ้นสุด] ⭐️ ---

        except Exception as e:
            logger.error(f"[get_timestamps_for_person] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return []
        
    def employee_exists(self, emp_id: str) -> bool:
        if not self.conn:
            return False
        try:
            with self.conn:
                self.cursor.execute("SELECT 1 FROM face_embeddings WHERE emp_id = %s", (emp_id,))
                return self.cursor.fetchone() is not None
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("employee_exists", str(e), None)
            return False

    def delete_employee(self, emp_id: str) -> int:
        """
        ลบข้อมูลพนักงานโดย emp_id (ลบแถวเดียวตามโมเดลใหม่ 1 คน = 1 แถว)
        """
        if not self.conn:
            return 0
        try:
            with self.conn:
                self.cursor.execute("DELETE FROM face_embeddings WHERE emp_id = %s", (emp_id,))
                self.conn.commit()
                return self.cursor.rowcount or 0
        except psycopg2.Error as e:
            self.conn.rollback()
            self.log_error("delete_employee", str(e), None)
            return 0

    def register_user(self, username: str, password: str, department: str,
                      access: list | str, is_admin: bool) -> bool:
        """
        สร้างผู้ใช้ใหม่ในตาราง login
        """
        if not self.conn:
            return False
        try:
            pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

            acc = access
            if isinstance(access, str):
                acc = [x.strip() for x in access.strip("{}").split(",") if x.strip()]

            with self.conn, self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO login (user_name, pass_user, department, access, is_admin,
                                       temp_password_hash, temp_expire_at, must_change_password)
                    VALUES (%s, %s, %s, %s, %s, NULL, NULL, FALSE)
                    ON CONFLICT (user_name) DO NOTHING
                """, (username, pw_hash, department, acc, bool(is_admin)))
                return cur.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            self.log_error("register_user", str(e), None)
            return False

    def get_user_profile(self, username: str) -> dict | None:
        """ดึง department / access / is_admin สำหรับออก JWT หลังตั้งรหัสผ่านใหม่"""
        if not self.conn:
            return None
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute("""
                    SELECT department, access, COALESCE(is_admin, FALSE)
                      FROM login
                     WHERE user_name = %s
                """, (username,))
                row = cur.fetchone()
                if not row:
                    return None
                dept, access, is_admin = row
                if isinstance(access, str):
                    access = [x.strip() for x in access.strip("{}").split(",") if x.strip()]
                return {"department": dept or "", "access": access or [], "is_admin": bool(is_admin)}
        except Exception as e:
            self.conn.rollback()
            self.log_error("get_user_profile", str(e), None)
            return None

    def set_temp_password(self, username: str, temp_password_plain: str, expire_minutes: int) -> bool:
        """
        แอดมินตั้งรหัสชั่วคราว
        """
        if not self.conn:
            return False
        try:
            hpw = bcrypt.hashpw(temp_password_plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            with self.conn, self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE login
                       SET temp_password_hash = %s,
                           temp_expire_at     = (NOW() AT TIME ZONE 'UTC') + (%s || ' minutes')::interval,
                           must_change_password = TRUE
                     WHERE user_name = %s
                """, (hpw, int(expire_minutes), username))
                return cur.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            self.log_error("set_temp_password", str(e), None)
            return False

    def verify_temp_password(self, username: str, temp_password_plain: str) -> tuple[bool, str]:
        """
        ตรวจสอบรหัสชั่วคราว (คืนค่า: (valid, reason))
        """
        if not self.conn:
            return False, "db_not_ready"
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute("""
                    SELECT temp_password_hash,
                           temp_expire_at,
                           COALESCE(must_change_password, FALSE)
                      FROM login
                     WHERE user_name = %s
                """, (username,))
                row = cur.fetchone()
                if not row:
                    return False, "user_not_found"
                temp_hash, exp_at, must_change = row
                if not temp_hash:
                    return False, "no_temp_password"
                cur.execute("SELECT (NOW() AT TIME ZONE 'UTC')")
                now_utc = cur.fetchone()[0]
                if exp_at is not None and now_utc > exp_at:
                    return False, "temp_expired"
                if not bcrypt.checkpw(temp_password_plain.encode("utf-8"), str(temp_hash).encode("utf-8")):
                    return False, "temp_invalid"
                if not must_change:
                    return False, "not_in_must_change_state"
                return True, "ok"
        except Exception as e:
            self.conn.rollback()
            self.log_error("verify_temp_password", str(e), None)
            return False, "error"

    def consume_temp_password(self, username: str, new_password_plain: str) -> bool:
        """
        เมื่อผู้ใช้ตั้งรหัสใหม่สำเร็จ
        """
        if not self.conn:
            return False
        try:
            new_hash = bcrypt.hashpw(new_password_plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            with self.conn, self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE login
                       SET pass_user = %s,
                           temp_password_hash = NULL,
                           temp_expire_at = NULL,
                           must_change_password = FALSE
                     WHERE user_name = %s
                """, (new_hash, username))
                return cur.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            self.log_error("consume_temp_password", str(e), None)
            return False

    def load_employee_embeddings(self, emp_id: str) -> list[bytes]:
        """
        คืนรายการ embedding (bytea) ที่มีทั้งหมดของ emp_id เพื่อกันซ้ำตอน enroll
        """
        if not self.conn:
            return []
        try:
            with self.conn, self.conn.cursor() as cur:
                cur.execute("""
                    SELECT embedding, embedding2, embedding3
                      FROM face_embeddings
                     WHERE emp_id = %s
                """, (emp_id,))
                row = cur.fetchone()
                if not row:
                    return []
                e1, e2, e3 = row
                out = []
                for e in (e1, e2, e3):
                    if e and len(e) > 0:
                        out.append(bytes(e))
                return out
        except Exception as e:
            self.conn.rollback()
            self.log_error("load_employee_embeddings", str(e), None)
            return []
        
    def get_employee_details(self, emp_id: str) -> dict | None:
        """
        [อัปเกรด 5 ช่อง] ดึงข้อมูลพนักงาน 1 คน พร้อมสถานะของ 5 ช่อง
        """
        if not self.conn:
            return None
        try:
            with self.conn, self.conn.cursor() as cur:
                # ⭐️ ดึง 5 ช่อง
                cur.execute("""
                    SELECT name, department, 
                           image_data, image_data2, image_data3, image_data4, image_data5
                    FROM face_embeddings
                    WHERE emp_id = %s
                """, (emp_id,))
                row = cur.fetchone()
                if not row:
                    return None
                
                name, dept, img1, img2, img3, img4, img5 = row
                
                def encode_image(img_data):
                    if img_data is not None and len(img_data) > 100:
                        try:
                            import base64
                            return base64.b64encode(img_data).decode('utf-8')
                        except Exception:
                            return None
                    return None

                return {
                    "emp_id": emp_id,
                    "name": name,
                    "department": dept,
                    "slots": {
                        "1": encode_image(img1),
                        "2": encode_image(img2),
                        "3": encode_image(img3),
                        "4": encode_image(img4),
                        "5": encode_image(img5)
                    }
                }
        except Exception as e:
            logger.error(f"[get_employee_details] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return None

    # --- ⭐️ [เพิ่มใหม่ 2/2] ฟังก์ชันสำหรับลบช่อง ⭐️ ---
    def clear_employee_slot(self, emp_id: str, slot_num: int) -> bool:
        """
        [อัปเกรด 5 ช่อง] ล้างข้อมูล image และ embedding ในช่อง (slot) ที่กำหนด
        """
        if not self.conn:
            return False
        # ⭐️ แก้ไขเป็น (1, 2, 3, 4, 5)
        if slot_num not in (1, 2, 3, 4, 5):
            return False
        
        cols_to_clear = ""
        if slot_num == 1:
            cols_to_clear = "image_data = '', embedding = ''"
        elif slot_num == 2:
            cols_to_clear = "image_data2 = '', embedding2 = ''"
        elif slot_num == 3:
            cols_to_clear = "image_data3 = '', embedding3 = ''"
        elif slot_num == 4:
            cols_to_clear = "image_data4 = '', embedding4 = ''"
        elif slot_num == 5:
            cols_to_clear = "image_data5 = '', embedding5 = ''"

        try:
            with self.conn, self.conn.cursor() as cur:
                sql = f"""
                    UPDATE face_embeddings
                    SET {cols_to_clear}
                    WHERE emp_id = %s
                """
                cur.execute(sql, (emp_id,))
                self.conn.commit()
                return cur.rowcount > 0 
        except Exception as e:
            logger.error(f"[clear_employee_slot] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return False
    def update_employee_info(self, emp_id: str, name: Optional[str], department: Optional[str]) -> bool:
        """
        อัปเดตเฉพาะ name และ department (ถ้าค่าที่ส่งมาไม่เป็น None)
        """
        if not self.conn:
            return False
        
        updates = []
        params = []
        
        # (เราจะอัปเดตเฉพาะ field ที่มีการส่งค่ามาจริงๆ)
        if name is not None and name.strip():
            updates.append("name = %s")
            params.append(name.strip())
        if department is not None:
            # (อนุญาตให้ตั้งค่าเป็น "ว่าง" ได้)
            updates.append("department = %s")
            params.append(department.strip() or None) 

        if not updates:
            # ถ้าไม่ได้ส่งอะไรมาเลย (เช่น กด Save โดยไม่แก้)
            return True 
            
        params.append(emp_id)
        
        try:
            with self.conn, self.conn.cursor() as cur:
                sql = f"""
                    UPDATE face_embeddings
                    SET {", ".join(updates)}
                    WHERE emp_id = %s
                """
                cur.execute(sql, tuple(params))
                self.conn.commit()
                logger.info(f"[update_employee_info] Updated {emp_id} with {updates}")
                return cur.rowcount > 0 # (คืนค่า True ถ้าอัปเดตสำเร็จ)
        except Exception as e:
            logger.error(f"[update_employee_info] error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return False