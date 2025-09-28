# ============================================================
# Niñera Virtual - Código funcional + Arquitectura por Capas
# Patrones: Adapter, Factory Method, Strategy, Facade, Observer, Mediator
# Incluye: borrar todas las zonas y borrar una zona por nombre
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
from datetime import datetime, timedelta
from threading import Thread, Semaphore
from queue import Queue, Empty, Full

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# Modelo
from ultralytics import YOLO

# Envío de alertas
import requests

# .env opcional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Parche Windows (ultralytics rutas)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

APP_NAME = "Niñera Virtual"
APP_VERSION = "3.2 Capas+Patrones (Zonas delete)"

# ==========================
# CONFIGURACIÓN / CONSTANTES
# ==========================
class Config:
    # Modelos
    YOLO_MODEL_PATH = 'NiñeraV.pt'     # tu modelo personalizado
    COCO_MODEL_PATH = 'yolov8s.pt'     # Ultralytics lo descarga si no está
    USE_COCO_MODEL  = True

    # Map COCO -> etiquetas del sistema
    COCO_CLASS_MAP = {
        'knife': 'cuchillo',
        'oven': 'horno',
        'chair': 'silla',
        'dining table': 'mesa',
        'table': 'mesa',
        'person': 'nino'
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
        'shelf': 0.35, 'estante': 0.35
    }
    HIGH_SURFACE_LABELS = [
        'chair', 'silla', 'bar', 'barra', 'table', 'mesa',
        'stool', 'taburete', 'counter', 'mostrador', 'shelf', 'estante'
    ]

    # Alertas / Cooldowns
    PROXIMITY_PX = 120.0
    CD_GENERAL   = 5
    CD_HANDRAIL  = 1
    CD_HEIGHT    = 2

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7657028357:AAHV3c1mpfHrFFUK_HciH6NNQ30pxtC6dfQ")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "1626038555")
    SEND_TELEGRAM      = bool(int(os.getenv("SEND_TELEGRAM_ALERTS", "1")))
    TELEGRAM_IMG_MAX_W = 640
    TELEGRAM_JPEG_QLTY = 80
    TELEGRAM_CONC      = 2

    # GUI
    UPDATE_MS    = 25
    CHECK_MS     = 30
    SAVE_IMG_DIR = "alertas_img"
    MAX_ALERT_IMGS = 500
    SOUND = True

    # Pre-proc
    GRAYSCALE = False
    CLAHE     = False


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
        x1,y1=RiskAnalysisFacade._center(b1); x2,y2=RiskAnalysisFacade._center(b2)
        return np.hypot(x1-x2,y1-y2)<thr
    @staticmethod
    def _point_in_polygon(x,y,poly):
        inside=False; n=len(poly)
        for i in range(n):
            x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
            if ((y1>y)!=(y2>y)) and (x<(x2-x1)*(y-y1)/(y2-y1+1e-9)+x1): inside=not inside
        return inside
    @staticmethod
    def _child_on_high_surface(c, s, label):
        c_x1,c_y1,c_x2,c_y2=c; s_x1,s_y1,s_x2,s_y2=s
        cw,ch=c_x2-c_x1,c_y2-c_y1; sw,sh=s_x2-s_x1,s_y2-s_y1
        if cw<=0 or ch<=0 or sw<=0 or sh<=0: return False
        cx=(c_x1+c_x2)/2; feet=c_y2
        ox=min(c_x2,s_x2)-max(c_x1,s_x1)
        if not (ox>cw*0.35 or (s_x1<cx<s_x2)): return False
        eff=max(sh,10)
        if label in ['bar','barra','table','mesa','counter','mostrador','shelf','estante']:
            tol=ch*0.12; cmin=s_y1-tol; cmax=s_y1+eff*0.30
            head_ok=(c_y1 < s_y1+eff*0.10)
            return (cmin<feet<cmax) and head_ok
        if label in ['chair','silla','stool','taburete']:
            top=s_y1+sh*0.15; bottom=s_y1+sh*0.75
            return (top<feet<bottom) and (c_y1<bottom)
        return False

    def _can(self, cam, typ, ck=None):
        last=self.cooldowns.get((cam,typ,ck))
        cd=Config.CD_GENERAL
        if typ=="CHILD_NEAR_RAILING": cd=Config.CD_HANDRAIL
        elif typ=="CHILD_ON_HIGH_SURFACE": cd=Config.CD_HEIGHT
        return (last is None) or (datetime.now()>=last+timedelta(seconds=cd))
    def _mark(self, cam, typ, ck=None): self.cooldowns[(cam,typ,ck)]=datetime.now()

    # --- API principal ---
    def detect_and_evaluate(self, frame_bgr, camera_id, camera_name):
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
            "CHILD_NEAR_RAILING": ("NIÑO CERCA DE BARANDA!", {'handrail','baranda'})
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
            high=[o for o in filtered if o.label in Config.HIGH_SURFACE_LABELS]
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
                    cx,cy=self._center(ch.box); ck=self._child_key(ch.box)
                    for name,polys in zones.items():
                        for poly in polys:
                            if self._point_in_polygon(cx,cy,poly):
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

    def __init__(self, root):
        # Logging + dirs
        os.makedirs(Config.SAVE_IMG_DIR, exist_ok=True)
        logging.basicConfig(filename='nany.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        self.root=root
        self.root.title(f"{APP_NAME} — {APP_VERSION}")
        self.root.minsize(1080,680)
        self.theme="dark"; self.colors=self.DARK

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
        # NUEVAS OPCIONES DE ELIMINAR ZONAS
        zm.add_command(label="🗑️ Eliminar zona por nombre", command=self.delete_zone_by_name)
        zm.add_command(label="🗑️ Eliminar TODAS las zonas (cámara actual)", command=self.delete_all_zones_current)
        m.add_cascade(label="Zonas", menu=zm)

        settings=tk.Menu(m, tearoff=0)
        settings.add_command(label="Tema: Oscuro/Claro", command=self.toggle_theme)
        m.add_cascade(label="Ajustes", menu=settings)

        helpm=tk.Menu(m, tearoff=0)
        helpm.add_command(label="Acerca de", command=lambda: messagebox.showinfo("Acerca de", f"{APP_NAME} {APP_VERSION}\nOpenCV + YOLO + Tkinter"))
        m.add_cascade(label="Ayuda", menu=helpm)

        self.root.config(menu=m)

    def _build_layout(self):
        c=self.colors
        top=ttk.Frame(self.root, style='Panel.TFrame', padding=(16,10)); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text=f"👶 {APP_NAME}", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(top, text="Arquitectura por Capas • Strategy+Facade+Observer+Mediator", style='Muted.TLabel').pack(side=tk.LEFT, padx=(10,0))
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
                      "high":tk.StringVar(value="0")}
        grid=ttk.Frame(mf, style='Card.TFrame'); grid.pack(fill=tk.X)
        self._metric_chip(grid,"🔢 Total",self.metrics["total"],0,0)
        self._metric_chip(grid,"🔪 Cuchillo",self.metrics["knife"],0,1)
        self._metric_chip(grid,"🪜 Escaleras",self.metrics["stairs"],1,0)
        self._metric_chip(grid,"🔥 Estufa",self.metrics["stove"],1,1)
        self._metric_chip(grid,"🍲 Olla",self.metrics["pot"],2,0)
        self._metric_chip(grid,"📍 Zonas",self.metrics["zone"],2,1)
        self._metric_chip(grid,"⬆️ Altura",self.metrics["high"],3,0)

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
                val=adapter.cap.get(cv2.CAP_PROP_FPS)
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
        if not self.current_camera_id or self.current_camera_id not in self.frame_queues:
            messagebox.showinfo("Zonas","Selecciona una cámara activa."); return
        try:
            frame=self.frame_queues[self.current_camera_id].get_nowait()
        except Empty:
            messagebox.showinfo("Zonas","No hay frame ahora. Intenta de nuevo."); return

        clone=frame.copy(); draw=frame.copy()
        win=f"Definir Zonas - {self.cameras[self.current_camera_id]['source_name']}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1000, 700)

        current_pts=[]; polygons=[]

        def cb(event,x,y,flags,param):
            nonlocal draw,current_pts
            if event==cv2.EVENT_LBUTTONDOWN:
                current_pts.append((x,y))
                cv2.circle(draw,(x,y),3,(0,255,0),-1)
                if len(current_pts)>1:
                    cv2.line(draw,current_pts[-2], current_pts[-1], (0,255,0),2)
        cv2.setMouseCallback(win, cb)

        info=["Instrucciones:","• Clicks para puntos del polígono.","• 'c' cierra polígono y pide etiqueta.",
              "• 'r' reinicia puntos.","• 'q' finaliza y guarda."]
        while True:
            disp=draw.copy(); y0=24
            for t in info:
                cv2.putText(disp,t,(12,y0),cv2.FONT_HERSHEY_SIMPLEX,0.7,(60,220,60),2); y0+=26
            cv2.imshow(win, disp)
            k=cv2.waitKey(30)&0xFF
            if k==ord('q'): break
            elif k==ord('r'): draw=clone.copy(); current_pts=[]
            elif k==ord('c'):
                if len(current_pts)>=3:
                    label=simpledialog.askstring("Etiqueta de zona","Nombre (ej: cocina, escaleras, balcon):", parent=self.root)
                    if label:
                        polygons.append((label.lower(), current_pts.copy()))
                        cv2.polylines(draw,[np.array(current_pts,dtype=np.int32)],True,(0,255,255),2)
                        current_pts=[]
                else:
                    messagebox.showinfo("Zonas","Mínimo 3 puntos.")
        cv2.destroyWindow(win)

        if polygons:
            cam_polys=self.per_camera_polygons.get(self.current_camera_id,{})
            for label,pts in polygons:
                cam_polys.setdefault(label,[]).append(pts)
            self.per_camera_polygons[self.current_camera_id]=cam_polys
            # actualizar en Facade (para que la capa de procesamiento las use)
            self.facade.set_polygons(self.current_camera_id, cam_polys)
            messagebox.showinfo("Zonas", f"Se registraron {len(polygons)} polígonos.")

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

    # ---- NUEVO: eliminar TODAS las zonas de la cámara actual ----
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

    # ---- NUEVO: eliminar una zona por nombre ----
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
        def inc(k): self.metrics[k].set(str(int(self.metrics[k].get())+1)); self.metrics["total"].set(str(int(self.metrics["total"].get())+1))
        low=text.lower()
        if "cuchillo" in low: inc("knife")
        if "escalera" in low: inc("stairs")
        if "estufa" in low or "cocina" in low: inc("stove")
        if "olla" in low or "sartén" in low: inc("pot")
        if "zona" in low: inc("zone")
        if "sobre" in low: inc("high")

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

    def _set_status(self, text): self.status_label.config(text=text)

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
    root=tk.Tk()
    app=CCTVMonitoringSystem(root)
    root.mainloop()
