# Niñera Virtual - Servicio Web

Este directorio contiene un backend HTTP/REST ligero basado en FastAPI para ejecutar la lógica
de detección y evaluación de riesgos de **Niñera Virtual** en un entorno de servidor.

## Requisitos del host

- Python 3.10+ (recomendado 3.11) con acceso a las dependencias del proyecto (OpenCV,
  Ultralytics, NumPy, etc.).
- Capacidad de ejecutar procesos de larga duración. Un hosting que solo acepte archivos
  estáticos vía FTP **no es suficiente**; necesitas un VPS, instancia cloud o contenedor
  donde puedas instalar Python y arrancar servicios.
- Acceso al modelo `NiñeraV.pt` (copiado al mismo directorio que el proyecto o con la ruta
  configurada mediante variables de entorno).

## Instalación

1. Copia el repositorio (por FTP/SFTP o Git) al servidor. Asegúrate de subir también los
   pesos `NiñeraV.pt` y, si se usa, `yolov8s.pt`.
2. Crea y activa un entorno virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. Instala las dependencias base del proyecto (las mismas que usas localmente) y después las
   específicas de la API web:

   ```bash
   pip install -r web_service/requirements.txt
   ```

4. Configura variables de entorno si necesitas rutas personalizadas para los modelos o tokens
   de Telegram (opcional).

## Ejecución del servicio

Ejecuta el servidor con Uvicorn:

```bash
uvicorn web_service.app:app --host 0.0.0.0 --port 8000
```

Opcionalmente establece `HOST` y `PORT` en variables de entorno y usa:

```bash
python web_service/app.py
```

## Uso de la API

- **GET `/health`**: comprueba que el backend está vivo.
- **POST `/analyze`**: recibe una imagen (JPEG/PNG) y devuelve las detecciones y alertas.

Ejemplo con `curl`:

```bash
curl -X POST "http://<tu-servidor>:8000/analyze" \
     -F "camera_id=cam1" \
     -F "camera_name=Sala" \
     -F "file=@frame.jpg"
```

La respuesta incluye las detecciones (cajas, etiqueta y confianza) y un arreglo de alertas
generadas por la lógica de riesgo.

## Integración con un sitio web

- Desde el front-end (HTML/JS), sube frames o capturas periódicas de la cámara infantil al
  endpoint `/analyze` mediante `fetch` o WebSocket (puedes añadir un endpoint streaming si lo
  necesitas).
- Para mejorar rendimiento, considera enviar frames ya redimensionados (ej. 640x480) y limitar
  la tasa de envío.
- Si quieres enviar alertas push o WebSocket al navegador, añade un canal adicional en FastAPI
  o usa un servicio de mensajería (MQTT, Redis, etc.).

## Despliegue típico con FTP

1. Contrata un VPS o servicio cloud que permita procesos Python siempre activos.
2. Usa FTP/SFTP solo para transferir el código y los modelos.
3. Conéctate por SSH para instalar dependencias, crear servicios (`systemd`, `pm2`, `supervisor`)
   y mantener el backend corriendo.
4. Configura un proxy inverso (Nginx/Apache) si quieres exponer el API en el puerto 80/443.

Recuerda que un hosting compartido “solo FTP” no podrá ejecutar la IA en tiempo real. Necesitas
control total del servidor para iniciar el proceso de FastAPI + Uvicorn.

## Despliegue en Render

1. Versiona el proyecto completo (incluyendo los modelos `.pt`) en un repositorio Git accesible
   para Render. Asegúrate de agregar el `requirements.txt` de la raíz y `render.yaml`.
2. En el panel de Render crea un **Web Service** y selecciona ese repositorio.
3. Render detectará el `render.yaml` y usará:
   - `pip install -r requirements.txt` como comando de build.
   - `uvicorn web_service.app:app --host 0.0.0.0 --port $PORT` como comando de arranque.
4. Configura variables de entorno según tu despliegue:
   - `MODEL_DIR`, `YOLO_MODEL_FILE`, `YOLO_MODEL_PATH` si los modelos viven fuera de la raíz.
   - `USE_COCO_MODEL=0` si prefieres omitir el modelo COCO.
   - `SEND_TELEGRAM_ALERTS=0/1`, tokens de Telegram, etc.
5. Despliega. Cuando el build finalice revisa los logs para confirmar que los modelos se cargaron
   correctamente. La API quedará disponible en `https://<tu-servicio>.onrender.com`.

Si necesitas recursos adicionales (GPU o más memoria) cambia el plan del servicio en Render antes
de desplegar.
