import logging
import os
from threading import Lock
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from entrenamiento_niñeravirtual2.proyecto import (
    Config,
    Detection,
    FusionDetectionStrategy,
    RiskAnalysisFacade,
    YOLOCustomStrategy,
    YOLOCocoStrategy,
)


logger = logging.getLogger("ninera_virtual.api")


def _resolve_file(path: str, label: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Archivo de modelo '{label}' no encontrado en {resolved}")
    return resolved


def _build_facade() -> RiskAnalysisFacade:
    custom_path = _resolve_file(Config.YOLO_MODEL_PATH, "YOLO_MODEL_PATH")
    custom_strategy = YOLOCustomStrategy(custom_path)
    coco_strategy = None
    if Config.USE_COCO_MODEL:
        try:
            coco_path = _resolve_file(Config.COCO_MODEL_PATH, "COCO_MODEL_PATH")
            coco_strategy = YOLOCocoStrategy(coco_path)
        except FileNotFoundError as exc:
            logger.warning("No se encontró el modelo COCO (%s). Continuando solo con el modelo personalizado.", exc)
            coco_strategy = None
    fusion = FusionDetectionStrategy(custom_strategy, coco_strategy)
    return RiskAnalysisFacade(fusion)


_facade = _build_facade()
_lock = Lock()


def _serialize_detection(det: Detection) -> Dict[str, Any]:
    x1, y1, x2, y2 = [int(v) for v in det.box]
    return {
        "label": det.label,
        "confidence": round(float(det.confidence), 4),
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "source": det.src,
    }


app = FastAPI(
    title="Niñera Virtual - API",
    description="Servicio HTTP para detección de riesgos usando el modelo de Niñera Virtual.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    camera_id: str = Form("webcam"),
    camera_name: str = Form("Web Upload"),
) -> JSONResponse:
    if file.content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Formato no soportado. Usa JPEG o PNG.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío.")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

    with _lock:
        try:
            detections, alerts = _facade.detect_and_evaluate(
                frame, camera_id=camera_id, camera_name=camera_name, return_alerts=True
            )
        except FileNotFoundError as exc:
            logger.error("Modelo faltante: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error ejecutando inferencia: %s", exc)
            raise HTTPException(status_code=500, detail="Error interno durante la inferencia.") from exc

    payload: Dict[str, Any] = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "detections": [_serialize_detection(det) for det in detections],
        "alerts": alerts,
    }
    return JSONResponse(payload)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Servicio Niñera Virtual listo.",
        "health_url": "/health",
        "analyze_url": "/analyze",
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web_service.app:app", host=host, port=port, reload=False)
