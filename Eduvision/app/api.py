from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np

from app import config as C
from app.db import DB
from app.vision import Hub

app = FastAPI(title="JPPN Security API")

app.mount("/static", StaticFiles(directory=str(C.STATIC_DIR)), name="static")

db = DB(C.DB_PATH)
hub = Hub()

@app.on_event("startup")
def startup_event():
    hub.start()

@app.on_event("shutdown")
def shutdown_event():
    hub.stop()

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = C.STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>JPPN Security Core Ativo</h1>"

@app.get("/api/v1/classroom/status")
def classroom_status():
    return hub.get_status()

@app.get("/video/{cam_id}")
def video_feed(cam_id: str):
    def gen():
        while True:
            frame = hub.get_frame(cam_id)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            import time
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

class StudentCreate(BaseModel):
    name: str
    turma: str = "1ECA"

@app.get("/api/v1/students")
def list_students():
    students = db.get_students()
    res = []
    for s in students:
        res.append({
            "id": s['id'],
            "name": s['name'],
            "turma": s.get('turma', '1ECA'),
            "samples": 3 if s.get('has_emb') else 0
        })
    return res

@app.post("/api/v1/students")
def create_student(data: StudentCreate):
    try:
        student_id = db.add_student(data.name, data.turma)
        return {"id": student_id, "status": "criado"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/v1/students/{student_id}")
def remove_student(student_id: int):
    db.delete_student(student_id)
    return {"status": "removido"}

@app.post("/api/v1/students/{student_id}/sample")
def capture_sample(student_id: int, camera: str = "CAM_01"):
    # Tenta obter um frame válido da câmara com várias tentativas rápidas
    import time
    frame = None
    for _ in range(10):
        frame = hub.get_frame(camera)
        if frame is not None:
            break
        time.sleep(0.2)

    if frame is None:
        raise HTTPException(status_code=400, detail="Câmara indisponível ou sem sinal de vídeo.")
    
    engine = hub.engines.get(camera)
    if not engine or not engine.det or not engine.rec:
        raise HTTPException(status_code=500, detail="Modelos de IA (YuNet/SFace) não inicializados.")

    h, w = frame.shape[:2]
    engine.det.setInputSize((w, h))
    faces = engine.det.detect(frame)
    
    faces_list = faces[1] if faces is not None and len(faces) > 0 and faces[1] is not None else []
    if len(faces_list) == 0:
        raise HTTPException(status_code=400, detail="Nenhum rosto detetado. Olhe bem diretamente para a câmara.")

    face = faces_list[0]
    try:
        aligned_face = engine.rec.alignCrop(frame, face)
        feature = engine.rec.feature(aligned_face)
        
        if feature is None or len(feature) == 0:
            raise HTTPException(status_code=400, detail="Falha ao extrair características faciais.")

        db.update_embedding(student_id, feature.astype(np.float32).tobytes())
        return {"status": "sucesso", "samples": 3}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Erro no processamento facial: {str(ex)}")

@app.get("/api/v1/events")
def get_events(limit: int = 10):
    return db.get_pending_events()

class EventUpdate(BaseModel):
    status: str

@app.patch("/api/v1/events/{event_id}")
def update_event(event_id: int, data: EventUpdate):
    db.update_event_status(event_id, data.status)
    return {"status": "atualizado"}

@app.get("/api/v1/attendance")
def get_attendance():
    return db.get_today_attendance()

@app.post("/api/v1/session/reset")
def reset_session():
    db.reset_session()
    return {"status": "sessao reiniciada"}