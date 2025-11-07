# ============================================================
# Niñera Virtual + Autenticación (Login/Registro) + SQLite
# Arquitectura por Capas y Patrones:
#  - Entrada: Adapter, Factory Method
#  - Procesamiento: Strategy, Facade, Observer
#  - Presentación: Observer, Mediator
#  - Almacenamiento: Singleton (DB), Repository (Usuarios)
# Además:
#  - Detección de tijeras (scissors) vía COCO (yolov8s.pt)
#  - Zonas: crear/cargar/guardar/eliminar por nombre o todas
#  - Autenticación: Registro / Login en SQLite con hashing seguro
# ============================================================

import cv2
import numpy as np
import pathlib
import sys
import os
import io
import json
import csv
import time
import logging
import sqlite3
import hashlib
import binascii
from datetime import datetime, timedelta
from threading import Thread, Semaphore
from queue import Queue, Empty, Full

# GUI (opcional en servidores headless)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
except Exception:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    simpledialog = None

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# Modelo
from ultralytics import YOLO

# Torch safe globals (PyTorch >=2.6 weights_only=True)
try:
    import torch
except Exception:
    torch = None

try:
    from torch.serialization import add_safe_globals
except Exception:
    add_safe_globals = None

try:
    from torch.nn import Sequential as TorchSequential
except Exception:
    TorchSequential = None

try:
    from torch.nn import Conv2d as TorchConv2d
except Exception:
    TorchConv2d = None

try:
    from ultralytics.nn.tasks import DetectionModel
except Exception:
    DetectionModel = None

try:
    from ultralytics.nn.modules.conv import Conv as UltralyticsConv
except Exception:
    UltralyticsConv = None

if add_safe_globals:
    _SAFE_CLASSES = [cls for cls in (DetectionModel, TorchSequential, TorchConv2d, UltralyticsConv) if cls is not None]
    if _SAFE_CLASSES:
        try:
            add_safe_globals(_SAFE_CLASSES)
        except Exception:
            pass

# Envío de alertas
import requests

# .env opcional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# Parche Windows (ultralytics rutas)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

APP_NAME = "Niñera Virtual"
APP_VERSION = "4.0 + Auth (Login/Registro) + Patrones"

# ==========================
# CONFIGURACIÓN / CONSTANTES
# ==========================
class Config:
    # Modelos
    MODEL_DIR = os.path.abspath(os.getenv('MODEL_DIR', '.'))
    YOLO_MODEL_FILE = os.getenv('YOLO_MODEL_FILE', 'NiñeraV.pt')
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH') or os.path.join(MODEL_DIR, YOLO_MODEL_FILE)
    COCO_MODEL_FILE = os.getenv('COCO_MODEL_FILE', 'yolov8s.pt')
    COCO_MODEL_PATH = os.getenv('COCO_MODEL_PATH') or os.path.join(MODEL_DIR, COCO_MODEL_FILE)
    USE_COCO_MODEL  = _env_flag('USE_COCO_MODEL', '1')


    # Map COCO -> etiquetas del sistema
    COCO_CLASS_MAP = {
        'knife': 'cuchillo',
        'oven': 'horno',
        'chair': 'silla',
        'dining table': 'mesa',
        'table': 'mesa',
        'person': 'nino',
        'scissors': 'tijeras',   # Tijeras
    }

    # Umbrales
    YOLO_CONF_DEFAULT = 0.25
    YOLO_IOU          = 0.45
    CLASS_THRESHOLDS = {
        'cuchillo': 0.35, 'knife': 0.35,
        'cocina': 0.35, 'kitchen': 0.35, 'cooker': 0.35,
        'olla': 0.35, 'pot': 0.35, 'pan': 0.35,
        'horno': 0.35, 'oven': 0.35,
        'escaleras': 0.30, 'stairs': 0.30,
        'nino': 0.40, 'child': 0.40,
        'handrail': 0.35, 'baranda': 0.35,
        'chair': 0.35, 'silla': 0.35,
        'bar': 0.35, 'barra': 0.35,
        'table': 0.35, 'mesa': 0.35,
        'stool': 0.35, 'taburete': 0.35,
        'counter': 0.35, 'mostrador': 0.35,
        'shelf': 0.35, 'estante': 0.35,
        'tijeras': 0.35, 'scissors': 0.35,
    }
    HIGH_SURFACE_LABELS = [
        'chair', 'silla', 'bar', 'barra', 'table', 'mesa',
        'stool', 'taburete', 'counter', 'mostrador', 'shelf', 'estante',
        'handrail', 'baranda'
    ]

    # Alertas / Cooldowns
    PROXIMITY_PX = 120.0
    CD_GENERAL   = 5
    CD_HANDRAIL  = 1
    CD_HEIGHT    = 2

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7657028357:AAHV3c1mpfHrFFUK_HciH6NNQ30pxtC6dfQ")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "1626038555")
    SEND_TELEGRAM      = _env_flag('SEND_TELEGRAM_ALERTS', '1')
    TELEGRAM_IMG_MAX_W = 640
    TELEGRAM_JPEG_QLTY = 80
    TELEGRAM_CONC      = 2

    # GUI
    UPDATE_MS    = 25
    SAVE_IMG_DIR = "alertas_img"
    MAX_ALERT_IMGS = 500
    SOUND = True

    # Pre-proc
    GRAYSCALE = False
    CLAHE     = False

    # DB
    DB_PATH = "ninera_virtual.db"
    PASSWORD_ITERATIONS = 120_000  # PBKDF2 iteraciones


# =========================================
# CAPA ALMACENAMIENTO: Singleton + Repository
# =========================================
class DatabaseConnection:
    """Singleton: una sola conexión SQLite para toda la app."""
    _instance = None
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DatabaseConnection()
        return cls._instance

    def _init_schema(self):
        cur = self.conn.cursor()
        # Tabla usuarios
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def get_conn(self):
        return self.conn


class UserRepository:
    """Repository: acceso y operaciones sobre usuarios."""
    def __init__(self, db: DatabaseConnection):
        self.conn = db.get_conn()

    # Hash seguro con PBKDF2-HMAC(SHA256)
    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        dk = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            Config.PASSWORD_ITERATIONS
        )
        return binascii.hexlify(dk).decode('utf-8')

    def create_user(self, name: str, email: str, password: str):
        salt = os.urandom(16)
        phash = self._hash_password(password, salt)
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO users (name, email, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (name.strip(), email.strip().lower(), phash, binascii.hexlify(salt).decode('utf-8'),
              datetime.now().isoformat(timespec='seconds')))
        self.conn.commit()

    def find_by_email(self, email: str):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, email, password_hash, salt FROM users WHERE email = ?",
                    (email.strip().lower(),))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "password_hash": row[3],
            "salt": row[4],
        }

    def verify_credentials(self, email: str, password: str) -> dict | None:
        user = self.find_by_email(email)
        if not user:
            return None
        salt = binascii.unhexlify(user["salt"].encode('utf-8'))
        phash = self._hash_password(password, salt)
        if phash == user["password_hash"]:
            return user
        return None


# ======================================
# CAPA ENTRADA: Adapter + Factory Method
# ======================================
class ICameraAdapter:
    def open(self) -> bool: ...
    def read(self): ...
    def release(self): ...
    def is_opened(self) -> bool: ...

class OpenCVCaptureAdapter(ICameraAdapter):
    """Adapter simple sobre cv2.VideoCapture (webcam, archivo, RTSP/HTTP)."""
    def __init__(self, source):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

class VideoSourceFactory:
    """Factory Method: decide qué adapter construir (aquí OpenCV para todas)."""
    @staticmethod
    def create(kind: str, target):
        # Si mañana cambia la lib de cámara IP, se cambia aquí el adapter.
        return OpenCVCaptureAdapter(target)


# =======================================
# CAPA PROCESAMIENTO: Strategy + Facade +
#                     Observer (eventos)
# =======================================
class Detection:
    def __init__(self, label, box, confidence, src):
        self.label = label
        self.box = box  # [x1,y1,x2,y2]
        self.confidence = float(confidence)
        self.src = src  # 'custom' | 'coco'

class IDetectionStrategy:
    def detect(self, frame_bgr) -> list['Detection']: ...

class YOLOCustomStrategy(IDetectionStrategy):
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame_bgr):
        out=[]
        res=self.model.predict(source=frame_bgr, conf=min(Config.YOLO_CONF_DEFAULT,0.25),
                               iou=Config.YOLO_IOU, verbose=False)
        if res and res[0].boxes is not None:
            names=self.model.names
            for b in res[0].boxes:
                xyxy=b.xyxy.cpu().numpy().astype(int)[0]
                conf=float(b.conf.item()); cls=int(b.cls.item())
                label=names[cls].lower()
                out.append(Detection(label, xyxy, conf, "custom"))
        return out

class YOLOCocoStrategy(IDetectionStrategy):
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame_bgr):
        out=[]
        res=self.model.predict(source=frame_bgr, conf=0.25, iou=0.5, verbose=False)
        if res and res[0].boxes is not None:
            names=self.model.names
            for b in res[0].boxes:
                xyxy=b.xyxy.cpu().numpy().astype(int)[0]
                conf=float(b.conf.item()); cls=int(b.cls.item())
                raw=names[cls]
                if isinstance(raw,(bytes,bytearray)): raw=raw.decode('utf-8',errors='ignore')
                mapped=Config.COCO_CLASS_MAP.get(str(raw).lower())
                if mapped:
                    out.append(Detection(mapped.lower(), xyxy, conf, "coco"))
        return out

class FusionDetectionStrategy(IDetectionStrategy):
    """Combina dos estrategias y aplica NMS por clase."""
    def __init__(self, strat_a: IDetectionStrategy, strat_b: IDetectionStrategy|None):
        self.a=strat_a; self.b=strat_b

    def detect(self, frame_bgr):
        dets=[]
        try: dets+=self.a.detect(frame_bgr)
        except Exception as e: logging.error(f"Custom detect error: {e}")
        if self.b:
            try: dets+=self.b.detect(frame_bgr)
            except Exception as e: logging.error(f"COCO detect error: {e}")
        if not dets: return []

        def iou(a,b):
            x1=max(a[0],b[0]); y1=max(a[1],b[1])
            x2=min(a[2],b[2]); y2=min(a[3],b[3])
            inter=max(0,x2-x1)*max(0,y2-y1)
            A=(a[2]-a[0])*(a[3]-a[1]); B=(b[2]-b[0])*(b[3]-b[1])
            return inter/max(A+B-inter+1e-9,1e-9)

        res=[]
        used=[False]*len(dets)
        for i,di in enumerate(dets):
            if used[i]: continue
            best=di
            for j in range(i+1,len(dets)):
                if used[j]: continue
                dj=dets[j]
                if di.label==dj.label and iou(di.box,dj.box)>0.5:
                    if dj.confidence>best.confidence: best=dj
                    used[j]=True
            used[i]=True
            res.append(best)
        return res


# ---- Observer infra ----
class IRiskObserver:
    def on_alert(self, event): ...

class RiskEvent:
    def __init__(self, camera_name, messages, frame_bgr):
        self.camera_name=camera_name
        self.messages=messages
        self.frame_bgr=frame_bgr
        self.ts=datetime.now()


# ---- Facade detección + evaluación ----
class RiskAnalysisFacade:
    """Expone detect_and_evaluate(). Internamente usa Strategy y reglas."""
    def __init__(self, detector: IDetectionStrategy):
        self.detector=detector
        self.observers: list[IRiskObserver]=[]
        self.cooldowns={}
        self.polygons_per_cam={}
        self.high_surfaces=set(Config.HIGH_SURFACE_LABELS)

    def subscribe(self, obs: IRiskObserver): self.observers.append(obs)
    def set_polygons(self, cam_id, zones_dict): self.polygons_per_cam[cam_id]=zones_dict

    # --- utilidades ---
    @staticmethod
    def _center(box): return int((box[0]+box[2])/2), int((box[1]+box[3])/2)
    @staticmethod
    def _child_key(box, grid=25):
        cx,cy=RiskAnalysisFacade._center(box); return (int(cx//grid), int(cy//grid))
    @staticmethod
    def _proximity(b1,b2,thr):
        x1a,y1a,x2a,y2a=b1; x1b,y1b,x2b,y2b=b2
        dx=max(x1a-x2b, x1b-x2a, 0)
        dy=max(y1a-y2b, y1b-y2a, 0)
        if dx==0 and dy==0:
            return True
        return float(np.hypot(dx,dy))<thr
    @staticmethod
    def _point_in_polygon(x,y,poly):
        inside=False; n=len(poly)
        for i in range(n):
            x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
            if ((y1>y)!=(y2>y)) and (x<(x2-x1)*(y-y1)/(y2-y1+1e-9)+x1): inside=not inside
        return inside
    @staticmethod
    def _polygon_probe_points(box):
        x1,y1,x2,y2=box
        cx=int((x1+x2)/2); cy=int((y1+y2)/2)
        return [
            (cx, cy),
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
    @staticmethod
    def _child_on_high_surface(c, s, label):
        c_x1,c_y1,c_x2,c_y2=c; s_x1,s_y1,s_x2,s_y2=s
        cw,ch=c_x2-c_x1,c_y2-c_y1; sw,sh=s_x2-s_x1,s_y2-s_y1
        if cw<=0 or ch<=0 or sw<=0 or sh<=0: return False
        cx=(c_x1+c_x2)/2; feet=c_y2; head=c_y1
        overlap_x=max(0, min(c_x2,s_x2)-max(c_x1,s_x1))
        min_width=max(min(cw,sw),1)
        center_inside=s_x1<=cx<=s_x2
        if overlap_x/min_width<0.35 and not center_inside: return False
        surface_top=min(s_y1,s_y2); surface_bottom=max(s_y1,s_y2)
        sh=surface_bottom-surface_top
        slender={'bar','barra','handrail','baranda','shelf','estante'}
        seating={'chair','silla','stool','taburete'}
        broad={'table','mesa','counter','mostrador'}
        if label in seating:
            top=surface_top+sh*0.15
            bottom=surface_top+sh*0.75
            return (top<feet<bottom) and (head<bottom)
        contact_band=max(0.18*ch, 0.35*sh, 18)
        upper_tol=max(0.20*ch, 0.25*sh, 12)
        if label in slender:
            contact_band=max(0.25*ch, 0.55*sh, 22)
            upper_tol=max(0.35*ch, 0.50*sh, 18)
        elif label in broad:
            contact_band=max(0.22*ch, 0.45*sh, 20)
            upper_tol=max(0.25*ch, 0.35*sh, 15)
        feet_delta=feet-surface_top
        if feet_delta<-upper_tol or feet_delta>contact_band: return False
        head_delta=head-surface_top
        if head_delta>upper_tol: return False
        return True

    def _can(self, cam, typ, ck=None):
        last=self.cooldowns.get((cam,typ,ck))
        cd=Config.CD_GENERAL
        if typ=="CHILD_NEAR_RAILING": cd=Config.CD_HANDRAIL
        elif typ=="CHILD_ON_HIGH_SURFACE": cd=Config.CD_HEIGHT
        return (last is None) or (datetime.now()>=last+timedelta(seconds=cd))
    def _mark(self, cam, typ, ck=None): self.cooldowns[(cam,typ,ck)]=datetime.now()
    def unsubscribe(self, obs: IRiskObserver):
        try:
            self.observers.remove(obs)
        except ValueError:
            pass

    # --- API principal ---
    def detect_and_evaluate(self, frame_bgr, camera_id, camera_name, *, return_alerts=False):
        # 1) detectar
        dets=self.detector.detect(frame_bgr)

        # 2) filtrar por thresholds
        filtered=[]
        for d in dets:
            thr=Config.CLASS_THRESHOLDS.get(d.label, Config.YOLO_CONF_DEFAULT)
            if d.confidence>=thr: filtered.append(d)

        # 3) reglas
        msgs=[]
        children=[d for d in filtered if d.label in ['nino','child']]
        prox_cfg={
            "CHILD_NEAR_KNIFE": ("NIÑO CERCA DE CUCHILLO!", {'knife','cuchillo'}),
            "CHILD_NEAR_STAIRS": ("NIÑO CERCA DE ESCALERAS!", {'stairs','escaleras'}),
            "CHILD_NEAR_STOVE": ("NIÑO CERCA DE ESTUFA/COCINA!", {'cooker','kitchen','cocina'}),
            "CHILD_NEAR_POT": ("NIÑO CERCA DE OLLA/SARTÉN!", {'pot','pan','olla'}),
            "CHILD_NEAR_OVEN": ("NIÑO CERCA DE HORNO!", {'oven','horno'}),
            "CHILD_NEAR_RAILING": ("NIÑO CERCA DE BARANDA!", {'handrail','baranda'}),
            "CHILD_NEAR_SCISSORS": ("NIÑO CERCA DE TIJERAS!", {'scissors','tijeras'}),
        }
        if children:
            # proximidad
            for ch in children:
                ck=self._child_key(ch.box)
                for key,(m,labels) in prox_cfg.items():
                    boxes=[o.box for o in filtered if o.label in labels]
                    for bx in boxes:
                        if self._proximity(ch.box,bx,Config.PROXIMITY_PX):
                            if self._can(camera_id,key,ck):
                                msgs.append(m); self._mark(camera_id,key,ck)
                            break
            # altura
            high=[o for o in filtered if o.label in self.high_surfaces]
            for ch in children:
                if (ch.box[2]-ch.box[0])*(ch.box[3]-ch.box[1]) < 40*40: continue
                for s in high:
                    if self._child_on_high_surface(ch.box, s.box, s.label):
                        k="CHILD_ON_HIGH_SURFACE"; ck=self._child_key(ch.box)
                        if self._can(camera_id,k,ck):
                            msgs.append(f"¡ALERTA! NIÑO SOBRE {s.label.upper()}!"); self._mark(camera_id,k,ck)
                        break
            # zonas poligonales
            zones=self.polygons_per_cam.get(camera_id,{})
            if zones:
                for ch in children:
                    ck=self._child_key(ch.box)
                    probes=self._polygon_probe_points(ch.box)
                    for name,polys in zones.items():
                        for poly in polys:
                            if any(self._point_in_polygon(px,py,poly) for px,py in probes):
                                k=f"CHILD_IN_ZONE_{name}"
                                if self._can(camera_id,k,ck):
                                    msgs.append(f"NIÑO EN ZONA: {name.upper()}!")
                                    self._mark(camera_id,k,ck)
                                break

        # 4) notificar observers si hay
        if msgs:
            ev=RiskEvent(camera_name, msgs, frame_bgr)
            for obs in self.observers:
                try: obs.on_alert(ev)
                except Exception as e: logging.error(f"Observer error: {e}")

        if return_alerts:
            return filtered, list(msgs)
        return filtered  # para dibujar en UI


# ======================================
# CAPA PRESENTACIÓN: Mediator + Observer
# ======================================
class INotificationService:
    def send_text(self, text: str): ...
    def send_image_with_caption(self, frame_bgr, caption: str): ...

class TelegramService(INotificationService):
    def __init__(self, token, chat_id, img_max_w, jpeg_quality, semaphore: Semaphore):
        self.token=token; self.chat_id=chat_id
        self.img_max_w=img_max_w; self.jpeg_quality=jpeg_quality
        self.sem=semaphore

    def _resize(self, frame):
        h,w=frame.shape[:2]
        if w>self.img_max_w:
            r=self.img_max_w/w
            return cv2.resize(frame,(self.img_max_w,int(h*r)),interpolation=cv2.INTER_AREA)
        return frame

    def send_text(self, text):
        def t():
            self.sem.acquire()
            try:
                url=f"https://api.telegram.org/bot{self.token}/sendMessage"
                r=requests.get(url, params={"chat_id": self.chat_id, "text": text}, timeout=10)
                r.raise_for_status()
            except Exception as e: logging.error(f"Telegram text: {e}")
            finally: self.sem.release()
        Thread(target=t,daemon=True).start()

    def send_image_with_caption(self, frame_bgr, caption):
        def t():
            self.sem.acquire()
            try:
                fr=self._resize(frame_bgr)
                ok,buf=cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY,self.jpeg_quality])
                if not ok: return
                url=f"https://api.telegram.org/bot{self.token}/sendPhoto"
                files={'photo': ('alert.jpg', io.BytesIO(buf), 'image/jpeg')}
                data={'chat_id': self.chat_id, 'caption': caption}
                r=requests.post(url, files=files, data=data, timeout=20)
                r.raise_for_status()
            except Exception as e: logging.error(f"Telegram photo: {e}")
            finally: self.sem.release()
        Thread(target=t,daemon=True).start()

class NotificationMediator:
    """Coordina los servicios externos (por ahora solo Telegram)."""
    def __init__(self):
        self.services=[]
        if Config.SEND_TELEGRAM and Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            self.services.append(TelegramService(
                Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID,
                Config.TELEGRAM_IMG_MAX_W, Config.TELEGRAM_JPEG_QLTY,
                Semaphore(Config.TELEGRAM_CONC)
            ))

    def notify(self, text, frame_bgr=None):
        for s in self.services:
            try:
                if frame_bgr is not None:
                    s.send_image_with_caption(frame_bgr, text)
                else:
                    s.send_text(text)
            except Exception as e:
                logging.error(f"Mediator notify error: {e}")


# ======================================
# UI AUTENTICACIÓN (Login / Registro)
# ======================================

class AuthWindow:
    """Pantalla de autenticación con login y registro."""

    PALETTE = {
        "bg": "#0b1120",
        "hero_bg": "#111b2c",
        "hero_text": "#f1f5f9",
        "hero_accent": "#38bdf8",
        "hero_muted": "#94a3b8",
        "card_bg": "#0f172a",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "outline": "#1f2937",
        "input_bg": "#172554",
        "input_border": "#1d4ed8",
        "success": "#22c55e",
        "error": "#f87171",
        "info": "#38bdf8"
    }

    def __init__(self, root):
        self.root = root
        self.db = DatabaseConnection.get_instance()
        self.users = UserRepository(self.db)

        self.status_var = tk.StringVar(value="")
        self.mode = tk.StringVar(value="login")

        self.login_email_var = tk.StringVar()
        self.login_pass_var = tk.StringVar()

        self.register_name_var = tk.StringVar()
        self.register_email_var = tk.StringVar()
        self.register_pass_var = tk.StringVar()
        self.register_confirm_var = tk.StringVar()
        self._pending_registration = None
        self.is_fullscreen = False

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Return>", self._submit_current)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._end_fullscreen)

    # ---------- UI ----------
    def _configure_style(self):
        c = self.PALETTE
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Auth.TFrame", background=c["bg"])
        style.configure("AuthHero.TFrame", background=c["hero_bg"])
        style.configure("AuthCard.TFrame", background=c["card_bg"])

        style.configure("HeroTitle.TLabel", background=c["hero_bg"], foreground=c["hero_text"],
                        font=("Segoe UI", 24, "bold"))
        style.configure("HeroSubtitle.TLabel", background=c["hero_bg"], foreground=c["hero_muted"],
                        font=("Segoe UI", 11))
        style.configure("HeroBadge.TFrame", background="#1e293b")
        style.configure("HeroBadgeTitle.TLabel", background="#1e293b", foreground=c["hero_text"],
                        font=("Segoe UI Semibold", 12))
        style.configure("HeroBadgeText.TLabel", background="#1e293b", foreground=c["hero_muted"],
                        font=("Segoe UI", 9))
        style.configure("HeroFeatureIcon.TLabel", background=c["hero_bg"], foreground=c["hero_accent"],
                        font=("Segoe UI", 16))
        style.configure("HeroFeatureTitle.TLabel", background=c["hero_bg"], foreground=c["hero_text"],
                        font=("Segoe UI Semibold", 12))
        style.configure("HeroFeatureText.TLabel", background=c["hero_bg"], foreground=c["hero_muted"],
                        font=("Segoe UI", 10))

        style.configure("CardTitle.TLabel", background=c["card_bg"], foreground=c["text"],
                        font=("Segoe UI", 21, "bold"))
        style.configure("CardSubtitle.TLabel", background=c["card_bg"], foreground=c["muted"],
                        font=("Segoe UI", 10))
        style.configure("FieldLabel.TLabel", background=c["card_bg"], foreground=c["text"],
                        font=("Segoe UI Semibold", 10))
        style.configure("FieldHint.TLabel", background=c["card_bg"], foreground=c["muted"],
                        font=("Segoe UI", 9))
        style.configure("AuthStatus.TLabel", background=c["card_bg"], foreground=c["error"],
                        font=("Segoe UI", 9))

        style.configure("Auth.TEntry",
                        fieldbackground=c["input_bg"],
                        foreground=c["text"],
                        bordercolor=c["outline"],
                        lightcolor=c["input_border"],
                        darkcolor=c["outline"],
                        insertcolor=c["text"],
                        padding=6)
        style.map("Auth.TEntry",
                  bordercolor=[("focus", c["input_border"])],
                  lightcolor=[("focus", c["input_border"])],
                  darkcolor=[("focus", c["input_border"])],
                  fieldbackground=[("focus", "#1e1b4b")])

        style.configure("AuthAccent.TButton",
                        background=c["accent"],
                        foreground="#ffffff",
                        font=("Segoe UI Semibold", 11),
                        padding=(18, 10),
                        borderwidth=0)
        style.map("AuthAccent.TButton",
                  background=[("active", c["accent_hover"]), ("disabled", c["outline"])],
                  foreground=[("disabled", c["muted"])])

        style.configure("AuthSecondary.TButton",
                        background=c["card_bg"],
                        foreground=c["muted"],
                        font=("Segoe UI", 10),
                        padding=(0, 8),
                        borderwidth=0)
        style.map("AuthSecondary.TButton",
                  foreground=[("active", c["text"])])

        style.configure("AuthToggle.TFrame", background=c["card_bg"])
        style.configure("AuthToggle.TButton",
                        background=c["card_bg"],
                        foreground=c["muted"],
                        font=("Segoe UI Semibold", 11),
                        padding=(14, 8),
                        borderwidth=0)
        style.map("AuthToggle.TButton",
                  foreground=[("active", c["text"])])

        style.configure("AuthToggleActive.TButton",
                        background=c["accent"],
                        foreground="#ffffff",
                        font=("Segoe UI Semibold", 11),
                        padding=(14, 8),
                        borderwidth=0)
        style.map("AuthToggleActive.TButton",
                  background=[("active", c["accent_hover"])])

    def _build_ui(self):
        c = self.PALETTE
        self.root.title("Ninera Virtual - Acceso seguro")
        self.root.geometry("1120x720")
        self.root.minsize(960, 620)
        self.root.resizable(True, True)
        self.root.configure(background=c["bg"])
        if sys.platform.startswith("win"):
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass

        outer = ttk.Frame(self.root, style="Auth.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        main = ttk.Frame(outer, style="Auth.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=32, pady=28)
        main.columnconfigure(0, weight=7)
        main.columnconfigure(1, weight=6)
        main.rowconfigure(0, weight=1)

        hero = ttk.Frame(main, style="AuthHero.TFrame", padding=(36, 40))
        hero.grid(row=0, column=0, sticky="nsew", padx=(0, 28))

        ttk.Label(hero, text="👶 Niñera Virtual", style="HeroTitle.TLabel").pack(anchor="w")
        ttk.Label(hero,
                  text="Monitoreo inteligente de entornos infantiles con visión artificial, detección de riesgos y alertas instantáneas.",
                  style="HeroSubtitle.TLabel",
                  wraplength=360).pack(anchor="w", pady=(12, 0))

        badge = ttk.Frame(hero, style="HeroBadge.TFrame", padding=(18, 16))
        badge.pack(anchor="w", pady=(20, 22), fill=tk.X)
        ttk.Label(badge, text="Versión 4.0 + Patrones + Seguridad", style="HeroBadgeTitle.TLabel").pack(anchor="w")
        ttk.Label(badge,
                  text="Incluye detección de tijeras, administración avanzada de zonas y autenticación con hashing PBKDF2.",
                  style="HeroBadgeText.TLabel",
                  wraplength=280).pack(anchor="w", pady=(6, 0))

        feats = ttk.Frame(hero, style="AuthHero.TFrame")
        feats.pack(fill=tk.BOTH, expand=True)
        items = [
            ("🛡️", "Seguridad proactiva", "Analizamos cámaras en tiempo real y alertamos ante riesgos potenciales."),
            ("⚡", "Reacción inmediata", "Recibe notificaciones con evidencia visual y registro histórico."),
            ("📊", "Control total", "Panel intuitivo, métricas y zonas configurables por cámara.")
        ]
        for icon, title, text in items:
            row = ttk.Frame(feats, style="AuthHero.TFrame")
            row.pack(anchor="w", fill=tk.X, pady=10)
            ttk.Label(row, text=icon, style="HeroFeatureIcon.TLabel").pack(side=tk.LEFT, padx=(0, 14))
            txt = ttk.Frame(row, style="AuthHero.TFrame")
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(txt, text=title, style="HeroFeatureTitle.TLabel").pack(anchor="w")
            ttk.Label(txt, text=text, style="HeroFeatureText.TLabel", wraplength=280).pack(anchor="w")

        ttk.Label(hero,
                  text="Activa tu cuenta en minutos y comienza a monitorear al instante.",
                  style="HeroSubtitle.TLabel").pack(anchor="w", pady=(24, 0))

        card = ttk.Frame(main, style="AuthCard.TFrame", padding=(34, 38))
        card.grid(row=0, column=1, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(3, weight=1)

        ttk.Label(card, text="Bienvenido de vuelta", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card,
                  text="Inicia sesión con tu email o crea una cuenta para proteger a los más pequeños.",
                  style="CardSubtitle.TLabel",
                  wraplength=360).grid(row=1, column=0, sticky="w", pady=(8, 0))

        toggle = ttk.Frame(card, style="AuthToggle.TFrame")
        toggle.grid(row=2, column=0, sticky="ew", pady=(26, 18))
        toggle.columnconfigure((0, 1), weight=1)

        self.login_toggle = ttk.Button(toggle, text="Iniciar sesión",
                                       style="AuthToggleActive.TButton",
                                       command=lambda: self._show_form("login", reset_status=True))
        self.login_toggle.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.register_toggle = ttk.Button(toggle, text="Crear cuenta",
                                          style="AuthToggle.TButton",
                                          command=lambda: self._show_form("register", reset_status=True))
        self.register_toggle.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        forms = ttk.Frame(card, style="AuthCard.TFrame")
        forms.grid(row=3, column=0, sticky="nsew")
        forms.columnconfigure(0, weight=1)

        self.forms = {
            "login": self._build_login_form(forms),
            "register": self._build_register_form(forms)
        }
        self.forms["register"].grid_remove()

        self.status_label = ttk.Label(card, textvariable=self.status_var, style="AuthStatus.TLabel", anchor="w")
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(18, 0))

        self._set_status("")
        self._show_form("login", reset_status=True)

    def _build_login_form(self, parent):
        frame = ttk.Frame(parent, style="AuthCard.TFrame")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Email", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w")
        self.login_email_entry = ttk.Entry(frame, style="Auth.TEntry", font=("Segoe UI", 11),
                                           textvariable=self.login_email_var)
        self.login_email_entry.grid(row=1, column=0, sticky="ew", pady=(4, 14))

        ttk.Label(frame, text="Contrasena", style="FieldLabel.TLabel").grid(row=2, column=0, sticky="w")
        self.login_pass_entry = ttk.Entry(frame, style="Auth.TEntry", show="•", font=("Segoe UI", 11),
                                          textvariable=self.login_pass_var)
        self.login_pass_entry.grid(row=3, column=0, sticky="ew", pady=(4, 18))

        ttk.Button(frame, text="Ingresar", style="AuthAccent.TButton",
                   command=self._handle_login).grid(row=4, column=0, sticky="ew")

        ttk.Button(frame, text="¿Aún sin cuenta? Regístrate",
                   style="AuthSecondary.TButton",
                   command=lambda: self._show_form("register", reset_status=True)).grid(row=5, column=0, sticky="w", pady=(18, 0))

        return frame

    def _build_register_form(self, parent):
        frame = ttk.Frame(parent, style="AuthCard.TFrame")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Nombre completo", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w")
        self.reg_name_entry = ttk.Entry(frame, style="Auth.TEntry", font=("Segoe UI", 11),
                                        textvariable=self.register_name_var)
        self.reg_name_entry.grid(row=1, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(frame, text="Email", style="FieldLabel.TLabel").grid(row=2, column=0, sticky="w")
        self.reg_email_entry = ttk.Entry(frame, style="Auth.TEntry", font=("Segoe UI", 11),
                                         textvariable=self.register_email_var)
        self.reg_email_entry.grid(row=3, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(frame, text="Contrasena", style="FieldLabel.TLabel").grid(row=4, column=0, sticky="w")
        self.reg_pass_entry = ttk.Entry(frame, style="Auth.TEntry", show="*", font=("Segoe UI", 11),
                                        textvariable=self.register_pass_var)
        self.reg_pass_entry.grid(row=5, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(frame, text="Confirmar contrasena", style="FieldLabel.TLabel").grid(row=6, column=0, sticky="w")
        self.reg_confirm_entry = ttk.Entry(frame, style="Auth.TEntry", show="*", font=("Segoe UI", 11),
                                           textvariable=self.register_confirm_var)
        self.reg_confirm_entry.grid(row=7, column=0, sticky="ew", pady=(4, 8))

        self.register_feedback = ttk.Label(frame, text="", style="FieldHint.TLabel", wraplength=340, anchor="w")
        self.register_feedback.grid(row=8, column=0, sticky="w", pady=(0, 8))

        ttk.Label(frame,
                  text="La contrasena debe tener minimo 6 caracteres. Usa una combinacion facil de recordar y segura.",
                  style="FieldHint.TLabel",
                  wraplength=340).grid(row=9, column=0, sticky="w", pady=(0, 14))

        self.register_validate_btn = ttk.Button(frame, text="Verificar datos", style="AuthAccent.TButton",
                                                command=self._prepare_register)
        self.register_validate_btn.grid(row=10, column=0, sticky="ew")
        self.register_validate_btn.state(["disabled"])
        self.register_validate_btn.configure(state=tk.DISABLED)

        self.register_confirm_btn = ttk.Button(frame, text="Confirmar registro", style="AuthAccent.TButton",
                                               command=self._confirm_register)
        self.register_confirm_btn.grid(row=11, column=0, sticky="ew", pady=(8, 0))
        self.register_confirm_btn.state(["disabled"])
        self.register_confirm_btn.configure(state=tk.DISABLED)

        ttk.Button(frame, text="¿Ya tienes cuenta? Inicia sesión",
                   style="AuthSecondary.TButton",
                   command=lambda: self._show_form("login", reset_status=True)).grid(row=12, column=0, sticky="w", pady=(18, 0))

        for var in (self.register_name_var, self.register_email_var,
                    self.register_pass_var, self.register_confirm_var):
            var.trace_add("write", lambda *args: self._validate_register_fields())

        self._validate_register_fields()

        return frame

    # ---------- Comportamiento ----------
    def _show_form(self, mode: str, reset_status: bool = False):
        if mode not in self.forms:
            return
        self.mode.set(mode)
        for key, frame in self.forms.items():
            if key == mode:
                frame.grid()
            else:
                frame.grid_remove()
        self._update_toggle_styles()
        if reset_status:
            self._set_status("")
        if mode == "login":
            if hasattr(self, "login_email_entry"):
                self.login_email_entry.focus_set()
        else:
            if hasattr(self, "reg_name_entry"):
                self.reg_name_entry.focus_set()
        self._validate_register_fields()

    def _update_toggle_styles(self):
        if self.mode.get() == "login":
            self.login_toggle.configure(style="AuthToggleActive.TButton")
            self.register_toggle.configure(style="AuthToggle.TButton")
        else:
            self.login_toggle.configure(style="AuthToggle.TButton")
            self.register_toggle.configure(style="AuthToggleActive.TButton")

    def _set_status(self, message: str, tone: str = "error"):
        palette = self.PALETTE
        colors = {
            "error": palette["error"],
            "success": palette["success"],
            "info": palette["info"]
        }
        self.status_label.configure(foreground=colors.get(tone, palette["error"]))
        self.status_var.set(message)

    def _set_register_feedback(self, message: str, tone: str = "info"):
        if not hasattr(self, "register_feedback"):
            return
        palette = self.PALETTE
        colors = {
            "error": palette["error"],
            "success": palette["success"],
            "info": palette["muted"]
        }
        self.register_feedback.configure(text=message, foreground=colors.get(tone, palette["muted"]))

    def _validate_register_fields(self):
        if not hasattr(self, "register_validate_btn"):
            return False

        name = (self.register_name_var.get() or "").strip()
        email = (self.register_email_var.get() or "").strip()
        password = self.register_pass_var.get() or ""
        confirm = self.register_confirm_var.get() or ""

        self._pending_registration = None
        self.register_confirm_btn.state(["disabled"])
        self.register_confirm_btn.configure(state=tk.DISABLED)

        valid = False
        message = "Completa todos los campos para crear tu cuenta."
        tone = "info"

        if not name or not email or not password or not confirm:
            pass
        elif len(name) < 2:
            message = "Ingresa tu nombre completo (minimo 2 caracteres)."
            tone = "error"
        elif "@" not in email or "." not in email:
            message = "Introduce un email valido."
            tone = "error"
        elif len(password) < 6:
            message = "La contrasena debe tener al menos 6 caracteres."
            tone = "error"
        elif password != confirm:
            message = "Las contrasenas deben coincidir exactamente."
            tone = "error"
        else:
            message = "Listo para verificar los datos."
            tone = "success"
            valid = True

        if valid:
            self.register_validate_btn.state(["!disabled"])
            self.register_validate_btn.configure(state=tk.NORMAL)
            self._set_register_feedback(message, "success")
        else:
            self.register_validate_btn.state(["disabled"])
            self.register_validate_btn.configure(state=tk.DISABLED)
            self._set_register_feedback(message, tone)

        return valid

    def _handle_login(self):
        email = (self.login_email_var.get() or "").strip().lower()
        password = self.login_pass_var.get() or ""
        if not email or not password:
            self._set_status("Completa email y contrasena para ingresar.", "info")
            return

        user = self.users.verify_credentials(email, password)
        if not user:
            self._set_status("Credenciales invalidas. Revisa los datos e intenta de nuevo.", "error")
            messagebox.showerror("Login", "Credenciales invalidas.")
            return

        self._set_status("")
        self.login_pass_var.set("")
        messagebox.showinfo("Bienvenido", f"Hola {user['name']}!")
        self._open_main(user)

    def _prepare_register(self):
        if not self._validate_register_fields():
            self._set_status("Verifica los campos resaltados antes de continuar.", "info")
            return

        data = {
            "name": self.register_name_var.get().strip(),
            "email": self.register_email_var.get().strip().lower(),
            "password": self.register_pass_var.get()
        }

        self._pending_registration = data
        self.register_confirm_btn.state(["!disabled"])
        self.register_confirm_btn.configure(state=tk.NORMAL)
        self._set_register_feedback("Datos verificados. Presiona Confirmar registro para guardar la cuenta.", "success")
        self._set_status("Datos verificados. Presiona Confirmar registro para crear la cuenta.", "info")

    def _confirm_register(self):
        if not self._pending_registration:
            self._set_status("Primero verifica los datos con el boton Verificar datos.", "info")
            return

        data = self._pending_registration
        try:
            self.users.create_user(data["name"], data["email"], data["password"])
        except sqlite3.IntegrityError:
            msg = "Ese email ya esta registrado."
            self._set_register_feedback(msg, "error")
            self._set_status(msg, "error")
            self._pending_registration = None
            self._validate_register_fields()
            return
        except Exception as exc:
            msg = f"Error creando la cuenta: {exc}"
            self._set_register_feedback(msg, "error")
            self._set_status(msg, "error")
            self._pending_registration = None
            self._validate_register_fields()
            return

        self.login_email_var.set(data["email"])
        self.login_pass_var.set("")
        self.register_pass_var.set("")
        self.register_confirm_var.set("")
        self._pending_registration = None
        self.register_confirm_btn.state(["disabled"])
        self.register_confirm_btn.configure(state=tk.DISABLED)
        self._show_form("login", reset_status=True)
        self._set_register_feedback("", "info")
        self._validate_register_fields()
        self._set_status("Cuenta creada correctamente. Inicia sesion con tus datos.", "success")
        messagebox.showinfo("Registro", "Cuenta creada. Ya puedes iniciar sesion.")

    def _submit_current(self, _event):
        if self.mode.get() == "login":
            self._handle_login()
        else:
            self._prepare_register()

    def _toggle_fullscreen(self, _event=None):
        self.is_fullscreen = not self.is_fullscreen
        try:
            self.root.attributes("-fullscreen", self.is_fullscreen)
        except tk.TclError:
            self.is_fullscreen = False

    def _end_fullscreen(self, _event=None):
        if self.is_fullscreen:
            self.is_fullscreen = False
            try:
                self.root.attributes("-fullscreen", False)
            except tk.TclError:
                pass

    # ---------- Navegación ----------
    def _open_main(self, user_dict):
        self.root.withdraw()
        win = tk.Toplevel()
        app = CCTVMonitoringSystem(win, session_user=user_dict, on_logout=self._on_logout)

        def on_main_close():
            try:
                win.destroy()
            except Exception:
                pass
            try:
                self.root.destroy()
            except Exception:
                pass
        win.protocol("WM_DELETE_WINDOW", on_main_close)

    def _on_logout(self):
        self.root.deiconify()

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


# ======================================
# UI (Observer de RiskAnalysisFacade)
# ======================================
class CCTVMonitoringSystem(IRiskObserver):
    # Paletas UI
    DARK = {
        "bg": "#0f172a", "bg2": "#111827", "panel": "#0b1220",
        "card": "#111827", "text": "#e5e7eb", "muted": "#9ca3af",
        "primary": "#2563eb", "primary_hover": "#1d4ed8",
        "accent": "#22c55e", "danger": "#ef4444", "warning": "#f59e0b",
        "outline": "#1f2937", "chip": "#1f2937"
    }
    LIGHT = {
        "bg": "#f3f4f6", "bg2": "#ffffff", "panel": "#ffffff",
        "card": "#ffffff", "text": "#111827", "muted": "#4b5563",
        "primary": "#2563eb", "primary_hover": "#1d4ed8",
        "accent": "#16a34a", "danger": "#dc2626", "warning": "#d97706",
        "outline": "#e5e7eb", "chip": "#f3f4f6"
    }

    def __init__(self, root, session_user: dict, on_logout=None):
        # Logging + dirs
        os.makedirs(Config.SAVE_IMG_DIR, exist_ok=True)
        logging.basicConfig(filename='nany.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        self.root=root
        self.root.title(f"{APP_NAME} - {APP_VERSION}")
        self.root.minsize(1080,680)
        self.theme="dark"; self.colors=self.DARK
        self.session_user = session_user
        self.on_logout_cb = on_logout
        self.is_fullscreen=False
        self._set_fullscreen(True)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._exit_fullscreen)

        # Estados
        self.running=True
        self.current_camera_id=None
        self.cameras={}          # camera_id -> dict(adapter, thread, active, name, frame_duration)
        self.frame_queues={}
        self.infer_queues={}
        self.display_queues={}
        self.infer_threads={}
        self.alert_history=[]
        self.per_camera_polygons={}
        self.metrics_vars=None

        # Estrategias / Facade / Mediator
        self._build_detector_and_facade()
        self.mediator=NotificationMediator()
        self.facade.subscribe(self)  # Observer

        # Colores anotación
        self.object_colors={
            'nino': (72,187,120), 'child': (72,187,120),
            'knife': (239,68,68), 'cuchillo': (239,68,68),
            'stairs': (245,158,11), 'escaleras': (245,158,11),
            'cooker': (37,99,235), 'kitchen': (37,99,235), 'cocina': (37,99,235),
            'oven': (99,102,241), 'horno': (99,102,241),
            'pot': (236,72,153), 'pan': (236,72,153), 'olla': (236,72,153),
            'handrail': (168,85,247), 'baranda': (168,85,247),
            'chair': (34,211,238), 'silla': (34,211,238),
            'bar': (59,130,246), 'barra': (59,130,246),
            'table': (16,185,129), 'mesa': (16,185,129),
            'stool': (250,204,21), 'taburete': (250,204,21),
            'counter': (251,146,60), 'mostrador': (251,146,60),
            'shelf': (52,211,153), 'estante': (52,211,153),
            'tijeras': (255,140,0), 'scissors': (255,140,0),
            'default': (203,213,225)
        }

        # UI
        self._configure_style()
        self._build_menu()
        self._build_layout()

        # Timers
        self.root.after(Config.UPDATE_MS, self._update_gui_frame)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- Observer: UI reacciona a eventos de riesgo ----
    def on_alert(self, event: RiskEvent):
        text=" | ".join(sorted(set(event.messages)))
        self._append_feed(f"[{event.camera_name}] {text}")
        self._update_banner(text, danger=True)
        self._save_alert_image(event.frame_bgr, event.camera_name)
        self._bump_metrics(text)
        self._beep()
        # Mediator (Telegram, etc)
        if Config.SEND_TELEGRAM:
            self.mediator.notify(f"🚨 ALERTA ({event.camera_name}): {text}", frame_bgr=event.frame_bgr)

    # ---- Detector + Facade ----
    def _build_detector_and_facade(self):
        # Strategy(s)
        try:
            custom = YOLOCustomStrategy(Config.YOLO_MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Modelo", f"No pudo cargarse el modelo custom: {e}")
            raise
        coco = None
        if Config.USE_COCO_MODEL:
            try:
                coco = YOLOCocoStrategy(Config.COCO_MODEL_PATH)
            except Exception as e:
                logging.error(f"No se cargó COCO: {e}")
        fusion = FusionDetectionStrategy(custom, coco)
        # Facade
        self.facade = RiskAnalysisFacade(fusion)

    # ==================== UI ====================
    def _configure_style(self):
        s=ttk.Style(); s.theme_use('clam'); c=self.colors
        s.configure('TFrame', background=c["bg"])
        s.configure('Panel.TFrame', background=c["panel"], relief='flat')
        s.configure('Card.TFrame', background=c["card"], relief='flat')
        s.configure('TLabel', background=c["bg"], foreground=c["text"], font=("Segoe UI",10))
        s.configure('Title.TLabel', background=c["bg"], foreground=c["text"], font=("Segoe UI",16,"bold"))
        s.configure('Muted.TLabel', background=c["bg"], foreground=c["muted"], font=("Segoe UI",9))
        s.configure('TButton', background=c["primary"], foreground="#fff", borderwidth=0, padding=(12,7),
                    font=("Segoe UI Semibold",10))
        s.map('TButton', background=[('active', c["primary_hover"])])
        s.configure('Ghost.TButton', background=c["chip"], foreground=c["text"], padding=(10,6), borderwidth=0)
        s.map('Ghost.TButton', background=[('active', c["outline"])])
        s.configure('TLabelframe', background=c["panel"], foreground=c["text"], bordercolor=c["outline"])
        s.configure('TLabelframe.Label', background=c["panel"], foreground=c["muted"], font=("Segoe UI Semibold",10))
        s.configure('Alert.TLabel', background=c["accent"], foreground="#fff", font=("Segoe UI Semibold",10), padding=6, anchor='center')
        s.configure('Danger.Alert.TLabel', background=c["danger"], foreground="#fff", font=("Segoe UI Semibold",10), padding=6, anchor='center')

    def _build_menu(self):
        m=tk.Menu(self.root, tearoff=0)
        filem=tk.Menu(m, tearoff=0)
        filem.add_command(label="Exportar historial CSV", command=self.export_history_csv)
        filem.add_separator(); filem.add_command(label="Salir", command=self.on_close)
        m.add_cascade(label="Archivo", menu=filem)

        zm=tk.Menu(m, tearoff=0)
        zm.add_command(label="Definir Zonas (polígonos)", command=self.define_zones_for_current)
        zm.add_command(label="Cargar Zonas (JSON)", command=self.load_zones_json)
        zm.add_command(label="Guardar Zonas (JSON)", command=self.save_zones_json)
        zm.add_separator()
        # ELIMINAR ZONAS
        zm.add_command(label="🗑️ Eliminar zona por nombre", command=self.delete_zone_by_name)
        zm.add_command(label="🗑️ Eliminar TODAS las zonas (cámara actual)", command=self.delete_all_zones_current)
        m.add_cascade(label="Zonas", menu=zm)

        settings=tk.Menu(m, tearoff=0)
        settings.add_command(label="Tema: Oscuro/Claro", command=self.toggle_theme)
        m.add_cascade(label="Ajustes", menu=settings)

        account=tk.Menu(m, tearoff=0)
        account.add_command(label=f"Cerrar sesión ({self.session_user.get('name','')})", command=self._logout)
        m.add_cascade(label="Cuenta", menu=account)

        helpm=tk.Menu(m, tearoff=0)
        helpm.add_command(label="Acerca de", command=lambda: messagebox.showinfo("Acerca de", f"{APP_NAME} {APP_VERSION}\nOpenCV + YOLO + Tkinter"))
        m.add_cascade(label="Ayuda", menu=helpm)

        self.root.config(menu=m)

    def _build_layout(self):
        c=self.colors
        top=ttk.Frame(self.root, style='Panel.TFrame', padding=(16,10)); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text=f"👶 {APP_NAME}", style='Title.TLabel').pack(side=tk.LEFT)
        usertxt = f"Sesión: {self.session_user.get('name','')} • {self.session_user.get('email','')}"
        ttk.Label(top, text=usertxt, style='Muted.TLabel').pack(side=tk.LEFT, padx=(10,0))
        ttk.Button(top, text="☀/🌙 Tema", style='Ghost.TButton', command=self.toggle_theme).pack(side=tk.RIGHT, padx=6)

        body=ttk.Frame(self.root, style='TFrame'); body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Izquierda: Fuentes
        left=ttk.Frame(body, style='Panel.TFrame', padding=12); left.pack(side=tk.LEFT, fill=tk.Y)
        self._build_left_panel(left)

        # Centro: Video
        center=ttk.Frame(body, style='Panel.TFrame', padding=12); center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_center_panel(center)

        # Derecha: Alertas/Métricas
        right=ttk.Frame(body, style='Panel.TFrame', padding=12); right.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_right_panel(right)

        status=ttk.Frame(self.root, style='Panel.TFrame'); status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label=ttk.Label(status, text="Listo.", style='Muted.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=6)

    def _build_left_panel(self, parent):
        c=self.colors
        lf=ttk.LabelFrame(parent, text="Fuentes de Video", padding=10); lf.pack(fill=tk.X)
        btns=ttk.Frame(lf, style='Card.TFrame'); btns.pack(fill=tk.X)
        ttk.Button(btns, text="📷  Agregar Cámara", command=self.add_camera_source).pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="🎞️  Agregar Video", command=self.add_video_source).pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="🗑️  Quitar Fuente", command=self.remove_camera_source).pack(fill=tk.X, pady=4)

        cf=ttk.LabelFrame(parent, text="Cámaras Activas", padding=10); cf.pack(fill=tk.BOTH, expand=True, pady=(12,0))
        listc=ttk.Frame(cf, style='Card.TFrame'); listc.pack(fill=tk.BOTH, expand=True)
        self.camera_list_box=tk.Listbox(listc, bg=c["card"], fg=c["text"], selectbackground=c["primary"], selectforeground="#fff",
                                        highlightthickness=0, bd=0, relief='flat', font=("Segoe UI",10))
        self.camera_list_box.pack(fill=tk.BOTH, expand=True)
        self.camera_list_box.bind('<<ListboxSelect>>', self.on_camera_select)

        zf=ttk.LabelFrame(parent, text="Zonas de Riesgo", padding=10); zf.pack(fill=tk.X, pady=(12,0))
        ttk.Button(zf, text="✏️  Definir Zonas", command=self.define_zones_for_current, style='Ghost.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(zf, text="📂  Cargar Zonas (JSON)", command=self.load_zones_json, style='Ghost.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(zf, text="💾  Guardar Zonas (JSON)", command=self.save_zones_json, style='Ghost.TButton').pack(fill=tk.X, pady=3)

    def _build_center_panel(self, parent):
        card=ttk.Frame(parent, style='Card.TFrame', padding=10); card.pack(fill=tk.BOTH, expand=True)
        header=ttk.Frame(card, style='Card.TFrame'); header.pack(fill=tk.X)
        ttk.Label(header, text="Vista en tiempo real", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header, text="Selecciona una fuente para iniciar el monitoreo.", style='Muted.TLabel').pack(side=tk.LEFT, padx=(10,0))
        self.video_label=ttk.Label(card, text=f"{APP_NAME}\n\nSelecciona o agrega una fuente.", anchor='center', style='TLabel', font=("Segoe UI Semibold",14))
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=(8,4))
        self.alert_banner=ttk.Label(card, text="Sistema listo.", style='Alert.TLabel')
        self.alert_banner.pack(fill=tk.X, pady=(6,0))

    def _build_right_panel(self, parent):
        af=ttk.LabelFrame(parent, text="Alertas Recientes", padding=10); af.pack(fill=tk.BOTH, expand=True)
        feedc=ttk.Frame(af, style='Card.TFrame'); feedc.pack(fill=tk.BOTH, expand=True)
        self.feed=tk.Text(feedc, height=14, wrap='word', bg=self.colors["card"], fg=self.colors["text"],
                          insertbackground=self.colors["text"], highlightthickness=0, bd=0, relief='flat', font=("Segoe UI",10))
        self.feed.pack(fill=tk.BOTH, expand=True); self.feed.config(state='disabled')

        mf=ttk.LabelFrame(parent, text="Métricas", padding=10); mf.pack(fill=tk.X, pady=(12,0))
        self.metrics={"total":tk.StringVar(value="0"),"knife":tk.StringVar(value="0"),
                      "stairs":tk.StringVar(value="0"),"stove":tk.StringVar(value="0"),
                      "pot":tk.StringVar(value="0"),"zone":tk.StringVar(value="0"),
                      "high":tk.StringVar(value="0"),"scissors":tk.StringVar(value="0")}

        grid=ttk.Frame(mf, style='Card.TFrame'); grid.pack(fill=tk.X)
        self._metric_chip(grid,"🔢 Total",self.metrics["total"],0,0)
        self._metric_chip(grid,"🔪 Cuchillo",self.metrics["knife"],0,1)
        self._metric_chip(grid,"🪜 Escaleras",self.metrics["stairs"],1,0)
        self._metric_chip(grid,"🔥 Estufa",self.metrics["stove"],1,1)
        self._metric_chip(grid,"🍲 Olla",self.metrics["pot"],2,0)
        self._metric_chip(grid,"📍 Zonas",self.metrics["zone"],2,1)
        self._metric_chip(grid,"⬆️ Altura",self.metrics["high"],3,0)
        self._metric_chip(grid,"✂️ Tijeras",self.metrics["scissors"],3,1)

        actions=ttk.Frame(parent, style='Card.TFrame'); actions.pack(fill=tk.X, pady=(12,0))
        ttk.Button(actions, text="⬇️  Exportar historial CSV", command=self.export_history_csv).pack(fill=tk.X, pady=4)

    def _metric_chip(self,parent,label,var,row,col):
        card=ttk.Frame(parent, style='Card.TFrame', padding=8); card.grid(row=row,column=col,padx=6,pady=6,sticky="nsew")
        ttk.Label(card,text=label,style='Muted.TLabel').pack(anchor='w')
        ttk.Label(card,textvariable=var,style='Title.TLabel').pack(anchor='w')
        parent.grid_columnconfigure(col,weight=1)

    # ==================== FUENTES (Adapter + Factory) ====================
    def _add_source(self, target, kind):
        adapter=VideoSourceFactory.create(kind, target)
        if not adapter.open():
            messagebox.showerror("Fuente", f"No se pudo abrir {kind}: {target}")
            return

        camera_id=f"{kind}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.frame_queues[camera_id]=Queue(maxsize=2)
        self.infer_queues[camera_id]=Queue(maxsize=2)
        self.display_queues[camera_id]=Queue(maxsize=2)

        # nombre visible
        display=os.path.basename(str(target)) if isinstance(target,str) else f"Cámara {target}"

        # FPS
        fps=25.0
        try:
            if hasattr(adapter,'cap'):
                val=adapter.cap.get(cv2.CV_CAP_PROP_FPS if hasattr(cv2,'CV_CAP_PROP_FPS') else cv2.CAP_PROP_FPS)
                if val and val>0: fps=float(val)
        except Exception: pass
        frame_duration=1.0/max(fps,1.0)

        # registra cámara
        self.cameras[camera_id]={'adapter':adapter,'type':kind,'source_name':display,
                                 'thread':None,'active':True,'frame_duration':frame_duration}

        # hilos
        t=Thread(target=self._capture_loop, args=(camera_id,), daemon=True); t.start()
        self.cameras[camera_id]['thread']=t
        it=Thread(target=self._inference_loop, args=(camera_id,), daemon=True); it.start()
        self.infer_threads[camera_id]=it

        # UI
        self.camera_list_box.insert(tk.END, display)
        if self.current_camera_id is None:
            self.camera_list_box.selection_set(0); self.on_camera_select(None)
        self._set_status(f"Fuente agregada: {display}")

    def add_camera_source(self):
        idx=0
        names={d['source_name'] for d in self.cameras.values()}
        while f"Cámara {idx}" in names: idx+=1
        self._add_source(idx, "live")

    def add_video_source(self):
        path=filedialog.askopenfilename(title="Seleccionar Video",
                                        filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"), ("Todos","*.*")])
        if path: self._add_source(path, "video")

    def remove_camera_source(self):
        sel=self.camera_list_box.curselection()
        if not sel:
            messagebox.showinfo("Fuentes","Seleccione una fuente para quitar."); return
        name=self.camera_list_box.get(sel[0])
        cam_id=next((cid for cid,d in self.cameras.items() if d['source_name']==name),None)
        if not cam_id: return

        self.cameras[cam_id]['active']=False
        if self.cameras[cam_id]['thread'].is_alive(): self.cameras[cam_id]['thread'].join(timeout=1.0)
        self.cameras[cam_id]['adapter'].release()
        del self.cameras[cam_id]

        for m in (self.frame_queues,self.infer_queues,self.display_queues):
            q=m.pop(cam_id,None)
            if q:
                while not q.empty():
                    try: q.get_nowait()
                    except Empty: break

        self.camera_list_box.delete(sel[0])
        if self.current_camera_id==cam_id:
            self.current_camera_id=None
            self.video_label.config(image='', text=f"{APP_NAME}\n\nSelecciona o agrega una fuente.")
            if hasattr(self.video_label,'imgtk'): self.video_label.imgtk=None

        self._set_status(f"Fuente eliminada: {name}")

    def _capture_loop(self, camera_id):
        cam=self.cameras[camera_id]; adapter=cam['adapter']; is_video = isinstance(adapter.source,str)
        fdur=cam['frame_duration']
        while self.running and cam.get('active',False) and adapter.is_opened():
            t0=time.perf_counter()
            ret,frame=adapter.read()
            if not ret:
                if is_video and cam.get('active',False):
                    try: adapter.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception: break
                    continue
                else:
                    break
            # preview
            rq=self.frame_queues.get(camera_id)
            if rq:
                if rq.full():
                    try: rq.get_nowait()
                    except Empty: pass
                try: rq.put_nowait(frame)
                except Full: pass
            # infer
            iq=self.infer_queues.get(camera_id)
            if iq:
                if iq.full():
                    try: iq.get_nowait()
                    except Empty: pass
                try: iq.put_nowait(frame)
                except Full: pass
            if is_video:
                st=fdur-(time.perf_counter()-t0)
                if st>0: time.sleep(st)
        adapter.release()

    def on_camera_select(self, _):
        sel=self.camera_list_box.curselection()
        if not sel:
            if self.camera_list_box.size()>0:
                self.camera_list_box.selection_set(0); sel=(0,)
            else:
                self.current_camera_id=None
                self.video_label.config(image='', text=f"{APP_NAME}\n\nSelecciona o agrega una fuente.")
                if hasattr(self.video_label,'imgtk'): self.video_label.imgtk=None
                return
        name=self.camera_list_box.get(sel[0])
        cam_id=next((cid for cid,d in self.cameras.items() if d['source_name']==name),None)
        if cam_id and cam_id!=self.current_camera_id:
            self.current_camera_id=cam_id
            dq=self.display_queues.get(self.current_camera_id)
            if dq:
                while not dq.empty():
                    try: dq.get_nowait()
                    except Empty: break
            self.video_label.config(image='', text="")
            if hasattr(self.video_label,'imgtk'): self.video_label.imgtk=None
            self._set_status(f"Visualizando: {name}")

    # ==================== INFERENCIA (usa Facade) ====================
    def _inference_loop(self, camera_id):
        name=self.cameras[camera_id]['source_name']
        while self.running and self.cameras.get(camera_id,{}).get('active',False):
            try:
                frame=self.infer_queues[camera_id].get(timeout=0.2)
            except Empty:
                continue
            try:
                # Pre-procesado
                frame_proc=self._preprocess(frame)
                # Facade: detecta y evalúa + notifica (Observer)
                detections=self.facade.detect_and_evaluate(frame_proc, camera_id, name)
                # Dibuja
                annotated=self._draw(detections, frame)
                dq=self.display_queues.get(camera_id)
                if dq:
                    if dq.full():
                        try: dq.get_nowait()
                        except Empty: pass
                    try: dq.put_nowait(annotated)
                    except Full: pass
            except Exception as e:
                logging.error(f"Inferencia {camera_id}: {e}")

    def _preprocess(self, frame):
        processed=frame.copy()
        try:
            if Config.GRAYSCALE:
                gray=cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                if Config.CLAHE:
                    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    gray=clahe.apply(gray)
                processed=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif Config.CLAHE:
                lab=cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l,a,b=cv2.split(lab)
                clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                cl=clahe.apply(l); limg=cv2.merge((cl,a,b))
                processed=cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logging.warning(f"Preprocesado: {e}")
        return processed

    def _draw(self, detections, frame):
        out=frame.copy()
        for d in detections:
            x1,y1,x2,y2=d.box
            color=self.object_colors.get(d.label, self.object_colors['default'])
            cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
            cv2.putText(out, f"{d.label} {d.confidence:.2f} ({d.src})", (x1,max(14,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out

    # ==================== ZONAS ====================
    def define_zones_for_current(self):
        if not self.current_camera_id or self.current_camera_id not in self.cameras:
            messagebox.showinfo("Zonas","Selecciona una camara activa.")
            return

        frame = None
        frame_queue = self.frame_queues.get(self.current_camera_id)
        if frame_queue is not None:
            try:
                frame = frame_queue.get_nowait()
            except Empty:
                frame = None
        if frame is None:
            display_queue = self.display_queues.get(self.current_camera_id)
            if display_queue is not None:
                try:
                    frame = display_queue.get_nowait()
                except Empty:
                    frame = None
        if frame is None:
            messagebox.showinfo("Zonas","No hay imagen disponible en este momento. Intenta de nuevo.")
            return

        height, width = frame.shape[:2]
        max_w, max_h = 1280, 720
        scale = min(max_w / float(width), max_h / float(height), 1.0)
        if scale < 1.0:
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            display = cv2.resize(frame, (new_w, new_h))
        else:
            display = frame.copy()
            scale = 1.0
        disp_h, disp_w = display.shape[:2]

        image_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image=image_pil)

        cam_name = self.cameras[self.current_camera_id]['source_name']
        win = tk.Toplevel(self.root)
        win.title(f"Definir zonas - {cam_name}")
        win.configure(background=self.colors.get('bg', '#0f172a'))
        win.transient(self.root)
        win.resizable(False, False)
        win.grab_set()
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        container = ttk.Frame(win, style='Panel.TFrame', padding=(20, 16))
        container.grid(row=0, column=0, sticky='nsew')
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)
        container.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(container, style='Card.TFrame', padding=12)
        canvas_frame.grid(row=0, column=0, sticky='nsew')
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        canvas = tk.Canvas(canvas_frame, width=disp_w, height=disp_h, highlightthickness=0, bg='#000000')
        canvas.grid(row=0, column=0, sticky='nsew')
        canvas.create_image(0, 0, image=photo, anchor='nw')
        canvas.image = photo

        existing = self.per_camera_polygons.get(self.current_camera_id, {})
        existing_count = sum(len(polys) for polys in existing.values())

        controls = ttk.Frame(container, style='Card.TFrame', padding=(16, 12))
        controls.grid(row=0, column=1, sticky='ns', padx=(16, 0))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(7, weight=1)

        ttk.Label(controls, text='Definir zonas', style='Title.TLabel').grid(row=0, column=0, sticky='w')
        counter_var = tk.StringVar(value=f'Zonas registradas: {existing_count}')
        ttk.Label(controls, textvariable=counter_var, style='Muted.TLabel').grid(row=1, column=0, sticky='w', pady=(0, 12))

        instructions = (
            '- Haz clic sobre la imagen para trazar los puntos.',
            '- Deshacer punto elimina el ultimo trazo.',
            '- Guardar zona cierra el poligono y lo agrega a la lista.',
            '- Solo las zonas nuevas se pueden eliminar aqui antes de aplicar.'
        )
        ttk.Label(controls, text='\n'.join(instructions), style='Muted.TLabel', justify='left', wraplength=260).grid(row=2, column=0, sticky='w')

        ttk.Label(controls, text='Nombre de la zona', style='TLabel').grid(row=3, column=0, sticky='w', pady=(12, 0))
        zone_name_var = tk.StringVar()
        name_entry = ttk.Entry(controls, textvariable=zone_name_var)
        name_entry.grid(row=4, column=0, sticky='ew', pady=(2, 0))

        status_var = tk.StringVar(value='Puntos actuales: 0')
        ttk.Label(controls, textvariable=status_var, style='Muted.TLabel').grid(row=5, column=0, sticky='w', pady=(6, 6))

        actions = ttk.Frame(controls, style='Card.TFrame')
        actions.grid(row=6, column=0, sticky='ew')
        save_btn = ttk.Button(actions, text='Guardar zona')
        save_btn.pack(fill=tk.X, pady=(0, 6))
        undo_btn = ttk.Button(actions, text='Deshacer punto', style='Ghost.TButton')
        undo_btn.pack(fill=tk.X)
        reset_btn = ttk.Button(actions, text='Reiniciar poligono', style='Ghost.TButton')
        reset_btn.pack(fill=tk.X, pady=(6, 0))

        tree_frame = ttk.Frame(controls, style='Card.TFrame')
        tree_frame.grid(row=7, column=0, sticky='nsew', pady=(12, 6))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=('zona', 'estado', 'puntos'), show='headings', selectmode='browse', height=7)
        tree.heading('zona', text='Zona')
        tree.heading('estado', text='Estado')
        tree.heading('puntos', text='Puntos')
        tree.column('zona', width=120, anchor='w')
        tree.column('estado', width=80, anchor='center')
        tree.column('puntos', width=70, anchor='center')
        tree.grid(row=0, column=0, sticky='nsew')
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=vsb.set)
        tree.tag_configure('existing', foreground=self.colors.get('muted', '#9ca3af'))
        tree.tag_configure('new', foreground=self.colors.get('accent', '#22c55e'))

        ttk.Label(controls, text='Zonas nuevas agregadas aqui se guardaran al aplicar cambios.', style='Muted.TLabel', wraplength=260).grid(row=8, column=0, sticky='w')
        remove_btn = ttk.Button(controls, text='Eliminar seleccionada', style='Ghost.TButton')
        remove_btn.grid(row=9, column=0, sticky='ew', pady=(4, 0))

        footer = ttk.Frame(controls, style='Card.TFrame')
        footer.grid(row=10, column=0, sticky='ew', pady=(16, 0))
        apply_btn = ttk.Button(footer, text='Aplicar zonas')
        apply_btn.pack(side=tk.RIGHT, fill=tk.X)
        cancel_btn = ttk.Button(footer, text='Cancelar', style='Ghost.TButton')
        cancel_btn.pack(side=tk.RIGHT, padx=8)

        name_entry.focus_set()

        current_points = []
        point_handles = []
        line_handles = []
        new_polygons = []
        polygon_shapes = {}
        item_to_polygon = {}

        palette = self.colors
        accent_color = palette.get('accent', '#22c55e')
        muted_color = palette.get('muted', '#9ca3af')

        def to_display(pt):
            x, y = pt
            return int(round(x * scale)), int(round(y * scale))

        for label, polys in existing.items():
            for poly in polys:
                if not poly:
                    continue
                coords = []
                for px, py in poly:
                    dx, dy = to_display((px, py))
                    coords.extend((dx, dy))
                if coords:
                    canvas.create_polygon(coords, outline=muted_color, width=2, fill='')
                    tree.insert('', tk.END, values=(label, 'Actual', len(poly)), tags=('existing',))

        def update_status(extra=None):
            text = f'Puntos actuales: {len(current_points)}'
            if extra:
                text = f"{text} | {extra}"
            status_var.set(text)

        def update_counter():
            base = sum(len(polys) for polys in existing.values())
            counter_var.set(f'Zonas registradas: {base + len(new_polygons)}')

        def on_canvas_click(event):
            x = max(0, min(disp_w - 1, int(event.x)))
            y = max(0, min(disp_h - 1, int(event.y)))
            point_handles.append(canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=accent_color, outline=''))
            if current_points:
                lx, ly = current_points[-1]
                line_handles.append(canvas.create_line(lx, ly, x, y, fill=accent_color, width=2))
            current_points.append((x, y))
            update_status()

        def undo_point():
            if not current_points:
                return
            canvas.delete(point_handles.pop())
            if line_handles:
                canvas.delete(line_handles.pop())
            current_points.pop()
            update_status()

        def reset_polygon():
            while point_handles:
                canvas.delete(point_handles.pop())
            while line_handles:
                canvas.delete(line_handles.pop())
            current_points.clear()
            update_status('Poligono reiniciado.')

        def save_zone():
            name = (zone_name_var.get() or '').strip().lower()
            if not name:
                messagebox.showinfo('Zonas', 'Ingresa un nombre para la zona.')
                return
            if len(current_points) < 3:
                messagebox.showinfo('Zonas', 'Necesitas al menos 3 puntos para crear el poligono.')
                return
            original = []
            coords = []
            for x, y in current_points:
                ox = min(width - 1, max(0, int(round(x / scale))))
                oy = min(height - 1, max(0, int(round(y / scale))))
                original.append((ox, oy))
                coords.extend((x, y))
            poly_id = f"new_{int(time.time() * 1000)}_{len(new_polygons)}"
            new_polygons.append({'id': poly_id, 'name': name, 'points': original})
            polygon_shapes[poly_id] = canvas.create_polygon(coords, outline=accent_color, width=2, fill='')
            item_id = tree.insert('', tk.END, values=(name, 'Nuevo', len(original)), tags=('new',))
            item_to_polygon[item_id] = poly_id
            zone_name_var.set('')
            reset_polygon()
            update_counter()
            update_status('Zona registrada, aplica cambios para guardar.')

        def remove_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo('Zonas', 'Selecciona una zona nueva para eliminar.')
                return
            item = sel[0]
            poly_id = item_to_polygon.get(item)
            if not poly_id:
                messagebox.showinfo('Zonas', 'Solo se pueden eliminar las zonas nuevas creadas en esta ventana.')
                return
            tree.delete(item)
            item_to_polygon.pop(item, None)
            for idx, data in enumerate(list(new_polygons)):
                if data['id'] == poly_id:
                    new_polygons.pop(idx)
                    break
            handle = polygon_shapes.pop(poly_id, None)
            if handle is not None:
                canvas.delete(handle)
            update_counter()
            update_status('Zona eliminada.')

        def apply_zones():
            if not new_polygons:
                messagebox.showinfo('Zonas', 'No hay zonas nuevas para guardar.')
                return
            cam_polys = self.per_camera_polygons.get(self.current_camera_id, {}).copy()
            for data in new_polygons:
                cam_polys.setdefault(data['name'], []).append(data['points'])
            self.per_camera_polygons[self.current_camera_id] = cam_polys
            self.facade.set_polygons(self.current_camera_id, cam_polys)
            self._set_status(f"{len(new_polygons)} zona(s) nuevas registradas en {cam_name}.", 'info')
            messagebox.showinfo('Zonas', f"Se guardaron {len(new_polygons)} zona(s).")
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        def cancel():
            if new_polygons:
                if not messagebox.askyesno('Zonas', 'Descartar las zonas nuevas sin guardar?'):
                    return
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        save_btn.configure(command=save_zone)
        undo_btn.configure(command=undo_point)
        reset_btn.configure(command=reset_polygon)
        remove_btn.configure(command=remove_selected)
        apply_btn.configure(command=apply_zones)
        cancel_btn.configure(command=cancel)

        canvas.bind('<Button-1>', on_canvas_click)
        win.bind('<Return>', lambda _: save_zone())
        win.bind('<Escape>', lambda _: cancel())
        win.protocol('WM_DELETE_WINDOW', cancel)
        win.lift()
        update_status()
        update_counter()

    def load_zones_json(self):
        path=filedialog.askopenfilename(title="Cargar Zonas (JSON)", filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as f: data=json.load(f)
            out={}
            for cam_id,zones in data.items():
                cmap={}
                for label, polys in zones.items():
                    clean=[[(int(x),int(y)) for (x,y) in poly] for poly in polys]
                    cmap[label]=clean
                out[cam_id]=cmap
            self.per_camera_polygons=out
            # sincroniza con Facade
            for cam_id,z in out.items():
                self.facade.set_polygons(cam_id, z)
            messagebox.showinfo("Zonas","Zonas cargadas.")
        except Exception as e:
            messagebox.showerror("Zonas", f"Error: {e}")

    def save_zones_json(self):
        if not self.per_camera_polygons:
            messagebox.showinfo("Zonas","No hay zonas definidas."); return
        path=filedialog.asksaveasfilename(title="Guardar Zonas (JSON)", defaultextension=".json",
                                          filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path,"w",encoding="utf-8") as f:
                json.dump(self.per_camera_polygons, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Zonas","Zonas guardadas.")
        except Exception as e:
            messagebox.showerror("Zonas", f"Error: {e}")

    # ---- Eliminar TODAS las zonas de la cámara actual ----
    def delete_all_zones_current(self):
        if not self.current_camera_id:
            messagebox.showinfo("Zonas", "No hay cámara seleccionada.")
            return
        if self.current_camera_id not in self.per_camera_polygons:
            messagebox.showinfo("Zonas", "Esta cámara no tiene zonas.")
            return
        if not messagebox.askyesno("Zonas", "¿Eliminar TODAS las zonas de esta cámara?"):
            return
        del self.per_camera_polygons[self.current_camera_id]
        self.facade.set_polygons(self.current_camera_id, {})
        messagebox.showinfo("Zonas", "Zonas eliminadas para la cámara actual.")

    # ---- Eliminar una zona por nombre ----
    def delete_zone_by_name(self):
        if not self.current_camera_id:
            messagebox.showinfo("Zonas", "No hay cámara seleccionada.")
            return
        zones = self.per_camera_polygons.get(self.current_camera_id, {})
        if not zones:
            messagebox.showinfo("Zonas", "Esta cámara no tiene zonas.")
            return

        win = tk.Toplevel(self.root)
        win.title("Eliminar zona")
        win.transient(self.root)
        ttk.Label(win, text="Selecciona la zona a eliminar:", style='TLabel', padding=10).pack(anchor='w')
        lb = tk.Listbox(win, height=min(8, len(zones)))
        for name in zones.keys():
            lb.insert(tk.END, name)
        lb.pack(fill=tk.X, padx=10)

        def do_del():
            sel = lb.curselection()
            if not sel:
                messagebox.showinfo("Zonas", "Elige una zona.")
                return
            zname = lb.get(sel[0])
            if messagebox.askyesno("Zonas", f"¿Eliminar la zona '{zname}'?"):
                zones.pop(zname, None)
                if zones:
                    self.per_camera_polygons[self.current_camera_id] = zones
                else:
                    self.per_camera_polygons.pop(self.current_camera_id, None)
                self.facade.set_polygons(self.current_camera_id, zones)
                messagebox.showinfo("Zonas", f"Zona '{zname}' eliminada.")
                win.destroy()

        btns = ttk.Frame(win, padding=10)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Eliminar", command=do_del).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancelar", style='Ghost.TButton', command=win.destroy).pack(side=tk.RIGHT, padx=8)

    # ==================== RENDER ====================
    def _update_gui_frame(self):
        if self.current_camera_id and self.current_camera_id in self.display_queues:
            try:
                frame=self.display_queues[self.current_camera_id].get_nowait()
                img_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil=Image.fromarray(img_rgb)
                self.video_label.imgtk=ImageTk.PhotoImage(image=img_pil)
                self.video_label.configure(image=self.video_label.imgtk)
            except Empty:
                pass
            except Exception as e:
                logging.error(f"Update frame: {e}")
        self.root.after(Config.UPDATE_MS, self._update_gui_frame)

    # ==================== UTILIDADES UI ====================
    def _set_fullscreen(self, enabled: bool):
        try:
            self.root.attributes("-fullscreen", enabled)
        except tk.TclError:
            if enabled:
                try:
                    self.root.state('zoomed')
                except tk.TclError:
                    pass
            else:
                try:
                    self.root.state('normal')
                except tk.TclError:
                    pass
        self.is_fullscreen=enabled

    def _toggle_fullscreen(self, _event=None):
        self._set_fullscreen(not self.is_fullscreen)

    def _exit_fullscreen(self, _event=None):
        if self.is_fullscreen:
            self._set_fullscreen(False)

    def toggle_theme(self):
        self.theme="light" if self.theme=="dark" else "dark"
        self.colors=self.LIGHT if self.theme=="light" else self.DARK
        self._configure_style()
        self._set_status(f"Tema cambiado a {'claro' if self.theme=='light' else 'oscuro'}.")

    def export_history_csv(self):
        if not self.alert_history:
            messagebox.showinfo("Exportar","No hay alertas aún."); return
        path=filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")],
                                          title="Guardar historial de alertas")
        if not path: return
        try:
            with open(path,"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["timestamp","alerta","fuente"])
                for row in self.alert_history: w.writerow(row)
            messagebox.showinfo("Exportar","Historial exportado.")
        except Exception as e:
            messagebox.showerror("Exportar", f"Error: {e}")

    def _append_feed(self, line):
        self.feed.config(state='normal')
        ts=datetime.now().strftime("%H:%M:%S")
        self.feed.insert('1.0', f"🛎️  {ts}  {line}\n")
        self.feed.config(state='disabled')
        # guarda para exportar
        cam=line.split('] ')[0].split('[')[-1]
        text=line.split('] ')[-1]
        self.alert_history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, cam))

    def _update_banner(self, text, danger=False):
        self.alert_banner.config(text=text, style='Danger.Alert.TLabel' if danger else 'Alert.TLabel')

    def _bump_metrics(self, text):
        def inc(k):
            self.metrics[k].set(str(int(self.metrics[k].get())+1))
            self.metrics["total"].set(str(int(self.metrics["total"].get())+1))
        low=text.lower()
        if "cuchillo" in low: inc("knife")
        if "escalera" in low: inc("stairs")
        if "estufa" in low or "cocina" in low: inc("stove")
        if "olla" in low or "sartén" in low: inc("pot")
        if "zona" in low: inc("zone")
        if "sobre" in low: inc("high")
        if "tijeras" in low: inc("scissors")

    def _beep(self):
        if not Config.SOUND: return
        def t():
            try:
                if sys.platform=="win32":
                    import winsound; winsound.Beep(1300, 400)
                else:
                    print("\007", end=""); sys.stdout.flush()
            except Exception: pass
        Thread(target=t,daemon=True).start()

    def _save_alert_image(self, frame, camera_name):
        clean="".join(ch if ch.isalnum() else "_" for ch in camera_name)
        tsf=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path=os.path.join(Config.SAVE_IMG_DIR, f"alerta_{clean}_{tsf}.jpg")
        try:
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, Config.TELEGRAM_JPEG_QLTY])
        except Exception as e:
            logging.error(f"Guardar IMG: {e}")
        # rotación
        try:
            files=[os.path.join(Config.SAVE_IMG_DIR,f) for f in os.listdir(Config.SAVE_IMG_DIR)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]
            files.sort(key=os.path.getmtime)
            while len(files)>Config.MAX_ALERT_IMGS:
                os.remove(files[0]); files.pop(0)
        except Exception:
            pass

    def _set_status(self, text, tone='info'):
        palette=self.colors
        color_map={
            'info': palette.get('muted', '#9ca3af'),
            'success': palette.get('accent', '#22c55e'),
            'warning': palette.get('warning', '#f59e0b'),
            'danger': palette.get('danger', '#dc2626')
        }
        self.status_label.config(text=text, foreground=color_map.get(tone, palette.get('muted', '#9ca3af')))

    def _logout(self):
        if messagebox.askyesno("Cuenta", "¿Cerrar sesión y volver a la pantalla de inicio?"):
            try:
                self.on_close()
            except Exception:
                pass
            if callable(self.on_logout_cb):
                self.on_logout_cb()

    # ==================== CIERRE ====================
    def on_close(self):
        self.running=False
        for d in list(self.cameras.values()):
            d['active']=False
        for d in list(self.cameras.values()):
            if d.get('thread') and d['thread'].is_alive(): d['thread'].join(timeout=0.8)
            d['adapter'].release()
        for m in (self.frame_queues,self.infer_queues,self.display_queues):
            for q in list(m.values()):
                while not q.empty():
                    try: q.get_nowait()
                    except Empty: break
        try: self.root.destroy()
        except Exception: pass


# ==================== BOOTSTRAP ====================
if __name__ == "__main__":
    # Crea/abre DB (Singleton) y asegura schema
    DatabaseConnection.get_instance()

    # Arranca la UI de autenticación; tras login abre la app principal
    if tk is None:
        print("El entorno actual no soporta interfaz grafica (tkinter). Ejecuta la aplicacion GUI en un equipo con soporte de escritorio.")
    else:
        root = tk.Tk()
        AuthWindow(root)
        root.mainloop()
