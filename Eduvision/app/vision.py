import cv2
import numpy as np
import threading
import time
import os
from pathlib import Path
from app import config as C
from app.db import DB

# Força o transporte RTSP via TCP para evitar corrompimento de pacotes e erros de autenticação nas câmaras Intelbras
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class VisionEngine:
    def __init__(self, camera_id, source):
        self.camera_id = camera_id
        self.source = source
        self.cap = None
        self.running = False
        self.frame = None
        self.current_intruders_count = 0
        self.lock = threading.Lock()
        
        self.db = DB(C.DB_PATH)
        self.det = None
        self.rec = None
        self._init_models()

    def _init_models(self):
        try:
            self.det = cv2.FaceDetectorYN.create(
                str(C.MODEL_DET), "", (320, 320), 0.8, 0.3, 5000
            )
            self.rec = cv2.FaceRecognizerSF.create(
                str(C.MODEL_REC), ""
            )
        except Exception as e:
            print(f"[{self.camera_id}] Erro ao carregar modelos de IA: {e}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                src = self.source
                
                if isinstance(src, str) and src.startswith("rtsp://"):
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                else:
                    if isinstance(src, str) and src.isdigit():
                        src = int(src)
                    self.cap = cv2.VideoCapture(src)

                if not self.cap.isOpened():
                    time.sleep(2.0)
                    continue
                
                time.sleep(0.5)

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                self.cap = None
                time.sleep(1.0)
                continue

            processed = self._process_frame(frame)
            with self.lock:
                self.frame = processed

    def _process_frame(self, frame):
        if self.det is None or self.rec is None:
            return frame

        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        faces = self.det.detect(frame)
        
        faces_list = faces[1] if faces is not None and len(faces) > 0 and faces[1] is not None else []
        students = self.db.get_students_with_samples()

        intruders_in_frame = 0

        for face in faces_list:
            box = face[:4].astype(int)
            confidence_det = face[-1]

            label = "Desconhecido"
            color = (0, 140, 255)  # Laranja para desconhecido
            matched = False

            if students and self.rec:
                try:
                    aligned = self.rec.alignCrop(frame, face)
                    feature = self.rec.feature(aligned)
                    feature = feature / np.linalg.norm(feature)
                    
                    highest_sim = -1.0
                    best_student = None
                    
                    for s in students:
                        if s['embedding']:
                            db_feat = np.frombuffer(s['embedding'], dtype=np.float32)
                            db_feat = db_feat / np.linalg.norm(db_feat)
                            
                            sim = self.rec.match(feature, db_feat, cv2.FaceRecognizerSF_FR_COSINE)
                            if sim > highest_sim:
                                highest_sim = sim
                                best_student = s

                    if highest_sim > 0.30 and best_student:
                        student_name = best_student['name']
                        student_rm = '565126'
                        student_turma = best_student.get('turma', '1ECA')

                        label = f"{student_name} ({int(highest_sim * 100)}%)"
                        color = (0, 255, 120)  # Verde para reconhecido
                        matched = True
                        
                        self.db.log_attendance(
                            rm=student_rm, 
                            name=student_name, 
                            turma=student_turma, 
                            camera_id=self.camera_id, 
                            confidence=float(highest_sim)
                        )
                except Exception as e:
                    pass

            if not matched:
                if confidence_det > 0.85:
                    intruders_in_frame += 1
                    self.db.log_event(self.camera_id, "Intruso / Pessoa Não Reconhecida")

            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.current_intruders_count = intruders_in_frame
        return frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_intruders_count(self):
        with self.lock:
            return self.current_intruders_count

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class Hub:
    def __init__(self):
        self.engines = {}
        if isinstance(C.CAMERAS, dict):
            cameras_iter = C.CAMERAS.items()
        else:
            cameras_iter = [(f"CAM_{i+1}", src) for i, src in enumerate(C.CAMERAS)]

        for cam_id, src in cameras_iter:
            self.engines[cam_id] = VisionEngine(cam_id, src)

    def start(self):
        for engine in self.engines.values():
            engine.start()

    def stop(self):
        for engine in self.engines.values():
            engine.stop()

    def get_frame(self, cam_id):
        engine = self.engines.get(cam_id)
        if engine:
            return engine.get_frame()
        return None

    def get_status(self):
        db = DB(C.DB_PATH)
        att = db.get_today_attendance()
        total_live_intruders = sum(engine.get_intruders_count() for engine in self.engines.values())

        return {
            "room_id": "JPPN_LAB_01",
            "people_count": len(att),
            "authorized_count": len(att),
            "current_intruders_count": total_live_intruders
        }