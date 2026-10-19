import os

# Precisa ser definido ANTES de importar o cv2.
# TCP para estabilidade nas câmaras Intelbras + timeout de 4s para não travar quando o AIBOX está desligado.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;4000000"

import sys
import base64
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from app.db import DB
from app import config as C

BASE_DIR = Path(__file__).resolve().parent.parent
db = DB(C.DB_PATH)

IS_WINDOWS = sys.platform.startswith("win")

# deixa CPU livre para a thread da câmera (a IA do OpenCV usaria todos os núcleos)
cv2.setNumThreads(max(2, (os.cpu_count() or 4) // 2))

# ---------------------------------------------------------------------------
# Parâmetros de reconhecimento
# ---------------------------------------------------------------------------
LIMIAR_SIMILARIDADE = getattr(C, "LIMIAR_SIMILARIDADE", 0.363)  # recomendado p/ SFace (cosseno); ajustável no config.py
DEBUG_IA = False              # mostra a similaridade no vídeo/console para diagnóstico (ponha False depois)
LIMIAR_INTRUSO = 0.85         # confiança mínima do detector para contar como intruso
LARGURA_DETECCAO = 640        # a detecção roda numa imagem reduzida (mais rápido)
PROCESSAR_A_CADA = 3          # (não usado: a IA roda em thread própria)
IA_INTERVALO_MIN = 0.12       # s entre análises da IA (~8/s). Maior = mais FPS de vídeo, menos reatividade
LIMIAR_MESMO_DESCONHECIDO = 0.40  # similaridade p/ considerar que dois rostos desconhecidos são a MESMA pessoa
CONFIRMAR_APOS = 3            # detecções seguidas para confirmar um desconhecido (evita falso alarme de 1 frame)
JANELA_CONFIRMACAO = 4        # segundos máx. entre essas detecções
MIN_LADO_ROSTO = 40           # rostos menores que isso (px) não são rastreados
COOLDOWN_PRESENCA = 60        # segundos entre registros de presença do mesmo aluno
COOLDOWN_INTRUSO = 10         # segundos entre registros de evento de intruso
CACHE_ALUNOS_SEG = 3          # recarrega embeddings do banco a cada N segundos

COR_RECONHECIDO = (0, 255, 120)
COR_INTRUSO = (0, 140, 255)

# ---------------------------------------------------------------------------
# Modelos de IA (YuNet & SFace)
# ---------------------------------------------------------------------------
det = None
rec = None

model_det_path = str(C.MODELS_DIR / "face_detection_yunet_2023mar.onnx")
model_rec_path = str(C.MODELS_DIR / "face_recognition_sface_2021dec.onnx")

try:
    if not os.path.exists(model_det_path):
        print(f"[ERRO] Modelo de detecção não encontrado: {model_det_path}")
    elif not os.path.exists(model_rec_path):
        print(f"[ERRO] Modelo de reconhecimento não encontrado: {model_rec_path}")
    else:
        det = cv2.FaceDetectorYN.create(model_det_path, "", (320, 320), 0.8, 0.3, 5000)
        rec = cv2.FaceRecognizerSF.create(model_rec_path, "")
        print("Modelos de IA carregados com sucesso!")
except Exception as e:
    det = None
    rec = None
    print(f"[ERRO] Falha ao carregar modelos de IA: {e}")

if det is None or rec is None:
    print("[AVISO] Reconhecimento facial DESATIVADO. Baixe os modelos do OpenCV Zoo "
          "(face_detection_yunet_2023mar.onnx e face_recognition_sface_2021dec.onnx) "
          f"e coloque em: {C.MODELS_DIR}")

app = FastAPI(title="EduVision // JPPN Command Center", version="7.4")

# ---------------------------------------------------------------------------
# Estado compartilhado
# ---------------------------------------------------------------------------
frame_atual_1 = None      # frame com caixas desenhadas (stream)
frame_limpo_1 = None      # frame sem anotações (usado no cadastro da biometria)
frame_atual_2 = None
frame_limpo_2 = None
camera_1_online = False
camera_2_online = False
camera_1_fonte = "NENHUMA"   # "AIBOX", "WEBCAM" ou "NENHUMA"
fps_1 = 0
fps_2 = 0
current_intruders_count = 0

lock = threading.Lock()          # protege os frames
ia_lock = threading.Lock()       # protege det/rec (não são thread-safe)
cache_lock = threading.Lock()

_cache_alunos = {"t": 0.0, "dados": []}
_ultimo_registro = {}            # rm -> timestamp do último log de presença
_ultimo_intruso = 0.0
_erros_vistos = set()

frame_id = {1: 0, 2: 0}          # contador de frames novos (o stream só envia frames novos)
ia_estado = {"resultados": []}   # últimos rostos detetados pela thread de IA
reconhecidos = {}                # rm -> dados do aluno reconhecido hoje (vai para o dashboard)
reg_lock = threading.Lock()
ia_info = {"alunos_carregados": 0, "rostos": 0, "melhor_nome": None, "melhor_sim": None,
           "ultimo_erro": None, "limiar": LIMIAR_SIMILARIDADE}
_ultimo_print_ia = 0.0
desconhecidos = []              # pessoas desconhecidas já vistas (galeria de embeddings, 1ª detecção, miniatura)
_contador_desconhecidos = 0


# ---------------------------------------------------------------------------
# Abertura de fontes de vídeo
# ---------------------------------------------------------------------------
def abrir_rtsp(url):
    """Tenta abrir o stream RTSP (AIBOX). Retorna o cap pronto ou None."""
    if not url:
        return None
    try:
        cap = cv2.VideoCapture(
            url, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000],
        )
    except Exception:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
    cap.release()
    return None


def abrir_webcam():
    """Procura uma webcam local e devolve o cap pronto ou None.
    Tenta configurações da mais ambiciosa (720p MJPG) até o padrão do driver, e mostra no console o que falhou."""
    indice_cfg = getattr(C, "WEBCAM_INDEX", None)
    indices = (indice_cfg,) if indice_cfg is not None else (0, 1, 2)
    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("AUTO", cv2.CAP_ANY)] if IS_WINDOWS else [("AUTO", cv2.CAP_ANY)]
    w = getattr(C, "WEBCAM_LARGURA", 1280)
    h = getattr(C, "WEBCAM_ALTURA", 720)
    # (largura, altura, mjpg): da melhor para a mais compatível (None = padrão do driver)
    configs = [(w, h, True), (w, h, False), (640, 480, False), (None, None, False)]
    aberta_mas_sem_imagem = False

    for idx in indices:
        for nome_be, backend in backends:
            for (cw, ch, mjpg) in configs:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    break  # esse backend não enxerga este índice; passa para o próximo
                if mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if cw:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Webcams costumam entregar frames vazios nas primeiras leituras
                for _ in range(15):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        print(f"[WEBCAM] aberta: índice {idx}, {nome_be}, {cw or 'padrão'}x{ch or ''}, "
                              f"MJPG={mjpg}, imagem {frame.shape[1]}x{frame.shape[0]}")
                        return cap
                    time.sleep(0.05)
                aberta_mas_sem_imagem = True
                print(f"[WEBCAM] índice {idx} {nome_be} {cw or 'padrão'}x{ch or ''} MJPG={mjpg}: abriu mas não entregou imagem")
                cap.release()

    if aberta_mas_sem_imagem:
        print("[WEBCAM] A câmera existe mas não entrega imagem: provavelmente está EM USO por outro programa "
              "(outra instância do servidor ainda rodando, Zoom/Teams/app Câmera) ou bloqueada nas permissões de "
              "privacidade do Windows. Feche tudo e finalize processos python.exe no Gerenciador de Tarefas.")
    else:
        print("[WEBCAM] Nenhuma webcam encontrada (índices testados: " + ", ".join(map(str, indices)) + ").")
    return None


# ---------------------------------------------------------------------------
# Reconhecimento facial
# ---------------------------------------------------------------------------
def invalidar_cache_alunos():
    with cache_lock:
        _cache_alunos["t"] = 0.0


def carregar_alunos():
    """Lista [(aluno_dict, embedding_normalizado)] com cache curto para não bater no banco a cada frame."""
    agora = time.time()
    with cache_lock:
        if agora - _cache_alunos["t"] < CACHE_ALUNOS_SEG:
            return _cache_alunos["dados"]

    dados = []
    try:
        for s in db.get_students_with_samples():
            s = dict(s)
            emb = s.get("embedding")
            if not emb:
                continue
            v = np.frombuffer(emb, dtype=np.float32)
            if v.size != 128:
                continue
            n = np.linalg.norm(v)
            if n == 0:
                continue
            dados.append((s, (v / n).astype(np.float32)))
    except Exception as e:
        _avisar_uma_vez("db", f"Erro ao ler alunos do banco: {e}")

    with cache_lock:
        _cache_alunos["dados"] = dados
        _cache_alunos["t"] = agora
    ia_info["alunos_carregados"] = len(dados)
    return dados


def _avisar_uma_vez(chave, msg):
    ia_info["ultimo_erro"] = msg
    if chave not in _erros_vistos:
        _erros_vistos.add(chave)
        print(f"[ERRO] {msg}")


def detectar_rostos(img):
    """Detecta rostos numa imagem reduzida e devolve as coordenadas já na escala original."""
    h, w = img.shape[:2]
    escala = 1.0
    pequena = img
    if w > LARGURA_DETECCAO:
        escala = w / LARGURA_DETECCAO
        pequena = cv2.resize(img, (LARGURA_DETECCAO, int(round(h / escala))))

    with ia_lock:
        det.setInputSize((pequena.shape[1], pequena.shape[0]))
        _, faces = det.detect(pequena)

    if faces is None or len(faces) == 0:
        return []

    faces = faces.copy()
    faces[:, :14] *= escala  # caixa + 5 landmarks voltam para a escala original
    return list(faces)


def extrair_embedding(img, face):
    """Embedding SFace normalizado, shape (128,), float32."""
    with ia_lock:
        alinhado = rec.alignCrop(img, face)
        feat = rec.feature(alinhado)
    feat = np.asarray(feat, dtype=np.float32).flatten()
    n = np.linalg.norm(feat)
    if n == 0:
        return None
    return feat / n


def registrar_reconhecido(rm, nome, turma, sim, agora):
    """Guarda o aluno reconhecido em memória para o dashboard exibir."""
    with reg_lock:
        r = reconhecidos.get(rm)
        if r is None:
            reconhecidos[rm] = {
                "rm": rm, "name": nome, "turma": turma,
                "confidence": float(sim),
                "first_seen": agora, "last_seen": agora,
            }
        else:
            r["confidence"] = float(sim)
            r["last_seen"] = agora


def _miniatura(frame, face):
    x, y, w, h = [int(v) for v in face[:4]]
    m = int(0.25 * max(w, h))
    crop = frame[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, (96, 96)), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return ("data:image/jpeg;base64," + base64.b64encode(buf).decode()) if ok else None


def rastrear_desconhecido(frame, face, feat, agora):
    """Diferencia pessoas desconhecidas pelo embedding. Cada pessoa gera UMA ocorrência (na 1ª detecção);
    voltar a aparecer só atualiza 'last_seen'. Retorna o nº do desconhecido (None se ainda não confirmado)."""
    global _contador_desconhecidos
    novo = None
    with reg_lock:
        desconhecidos[:] = [d for d in desconhecidos if d["confirmado"] or agora - d["last_seen"] < 30]
        d, melhor = None, -1.0
        for c in desconhecidos:
            sim = max(float(np.dot(feat, g)) for g in c["galeria"])
            if sim > melhor:
                melhor, d = sim, c
        if d is not None and melhor >= LIMIAR_MESMO_DESCONHECIDO:
            if not d["confirmado"] and agora - d["last_seen"] > JANELA_CONFIRMACAO:
                d["hits"], d["first_seen"] = 0, agora
            d["hits"] += 1
            d["last_seen"] = agora
            if melhor < 0.8 and len(d["galeria"]) < 8:
                d["galeria"].append(feat)  # guarda poses diferentes da mesma pessoa
        else:
            d = {"hits": 1, "first_seen": agora, "last_seen": agora, "galeria": [feat],
                 "confirmado": False, "numero": None, "thumb": None}
            desconhecidos.append(d)
        if not d["confirmado"] and d["hits"] >= CONFIRMAR_APOS:
            _contador_desconhecidos += 1
            d.update(confirmado=True, numero=_contador_desconhecidos, thumb=_miniatura(frame, face))
            novo = d["numero"]
        numero = d["numero"]
    if novo:
        try:
            db.log_event("CAM_01", f"Desconhecido #{novo} detectado")
        except Exception as e:
            _avisar_uma_vez("evento", f"Falha ao registrar evento: {e!r}")
    return numero


def analisar_frame(frame):
    """Roda detecção + reconhecimento. Retorna (lista de (box, label, cor), qtd_intrusos)."""
    global _ultimo_print_ia
    resultados, intrusos, agora = [], 0, time.time()
    alunos = carregar_alunos()

    for face in detectar_rostos(frame):
        box = face[:4].astype(int)
        conf_det = float(face[-1])
        label, cor, reconhecido = "Desconhecido", COR_INTRUSO, False
        melhor, melhor_sim, feat = None, -1.0, None

        try:
            feat = extrair_embedding(frame, face)
            if feat is not None:
                for aluno, emb in alunos:
                    sim = float(np.dot(feat, emb))  # ambos normalizados = cosseno
                    if sim > melhor_sim:
                        melhor_sim, melhor = sim, aluno
                if melhor is not None and melhor_sim >= LIMIAR_SIMILARIDADE:
                    nome, rm = melhor["name"], f"RM{melhor['id']:04d}"
                    turma = melhor.get("turma", "1ECA")
                    label, cor, reconhecido = f"{nome} ({int(melhor_sim * 100)}%)", COR_RECONHECIDO, True
                    registrar_reconhecido(rm, nome, turma, melhor_sim, agora)
                    if agora - _ultimo_registro.get(rm, 0) > COOLDOWN_PRESENCA:
                        _ultimo_registro[rm] = agora
                        db.log_attendance(rm=rm, name=nome, turma=turma,
                                          camera_id="CAM_01", confidence=float(melhor_sim))
        except Exception as e:
            _avisar_uma_vez("reconhecimento", f"Falha no reconhecimento: {e!r}")

        if not reconhecido and conf_det > LIMIAR_INTRUSO:
            intrusos += 1
            if feat is not None and face[2] >= MIN_LADO_ROSTO:
                try:
                    numero = rastrear_desconhecido(frame, face, feat, agora)
                    if numero:
                        label = f"Desconhecido #{numero}"
                except Exception as e:
                    _avisar_uma_vez("rastreio", f"Falha ao rastrear desconhecido: {e!r}")
            if DEBUG_IA and melhor is not None:
                label += f" (melhor: {melhor['name']} {melhor_sim:.2f})"

        resultados.append((box, label, cor))
        ia_info["melhor_nome"] = melhor["name"] if melhor is not None else None
        ia_info["melhor_sim"] = round(melhor_sim, 3) if melhor is not None else None

    ia_info["rostos"] = len(resultados)
    if DEBUG_IA and resultados and agora - _ultimo_print_ia > 2:
        _ultimo_print_ia = agora
        print(f"[IA] alunos={len(alunos)} rostos={len(resultados)} melhor={ia_info['melhor_nome']} sim={ia_info['melhor_sim']}")
    return resultados, intrusos


def desenhar(frame, resultados, fonte):
    saida = frame.copy()
    for box, label, cor in resultados:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(saida, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(saida, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    cv2.putText(saida, f"FONTE: {fonte}", (10, saida.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)
    return saida


# ---------------------------------------------------------------------------
# Threads de captura
# ---------------------------------------------------------------------------
def capturar_camera_1(url):
    """CAM_01: só LÊ e publica frames (rápido, sem lag). A IA roda em outra thread (loop_ia).
    Usa o AIBOX (RTSP) quando disponível; se não, cai para a webcam local e volta ao AIBOX sozinha."""
    global frame_atual_1, frame_limpo_1, camera_1_online, camera_1_fonte, fps_1

    print("Iniciando captura CAM_01...")
    cap = None
    fonte = "NENHUMA"
    probe = {"thread": None, "cap": None, "done": False}
    proxima_sonda = 0.0

    contador = 0
    inicio = time.time()

    def sondar_rtsp():
        probe["cap"] = abrir_rtsp(url)
        probe["done"] = True

    while True:
        # --- escolhe a fonte -------------------------------------------------
        if cap is None:
            cap = abrir_rtsp(url)
            fonte = "AIBOX"
            if cap is None:
                print("Aviso: AIBOX indisponível via RTSP. Tentando webcam local...")
                cap = abrir_webcam()
                fonte = "WEBCAM"
                proxima_sonda = time.time() + 30
            if cap is None:
                camera_1_online = False
                camera_1_fonte = "NENHUMA"
                print("Aviso: nenhuma câmera disponível (AIBOX e webcam). Nova tentativa em 3s...")
                time.sleep(3)
                continue
            camera_1_fonte = fonte
            print(f"CAM_01 usando: {fonte}")

        elif fonte == "WEBCAM" and url:
            # sonda o AIBOX em segundo plano, sem travar o vídeo da webcam
            if probe["thread"] is None and time.time() >= proxima_sonda:
                probe.update(thread=threading.Thread(target=sondar_rtsp, daemon=True), cap=None, done=False)
                probe["thread"].start()
            elif probe["thread"] is not None and probe["done"]:
                if probe["cap"] is not None:
                    print("AIBOX voltou! Trocando CAM_01 da webcam para o RTSP.")
                    cap.release()
                    cap = probe["cap"]
                    fonte = "AIBOX"
                    camera_1_fonte = fonte
                probe.update(thread=None, cap=None, done=False)
                proxima_sonda = time.time() + 30

        # --- lê o frame ------------------------------------------------------
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            camera_1_online = False
            ia_estado["resultados"] = []
            cap.release()
            cap = None
            time.sleep(0.5)
            continue

        camera_1_online = True
        contador += 1
        tempo = time.time() - inicio
        if tempo >= 1:
            fps_1 = contador / tempo
            contador = 0
            inicio = time.time()

        anotado = desenhar(frame, ia_estado["resultados"], fonte)
        with lock:
            frame_limpo_1 = frame
            frame_atual_1 = anotado
            frame_id[1] += 1


def loop_ia():
    """Thread de IA: pega sempre o frame mais recente da CAM_01, detecta/reconhece e publica o resultado.
    Como roda separada da captura, o vídeo não trava enquanto a IA trabalha."""
    global current_intruders_count
    ultimo_id = -1

    while True:
        if det is None or rec is None:
            time.sleep(1)
            continue

        with lock:
            frame = frame_limpo_1 if camera_1_online else None
            fid = frame_id[1]

        if frame is None:
            ia_estado["resultados"] = []
            current_intruders_count = 0
            time.sleep(0.1)
            continue
        if fid == ultimo_id:
            time.sleep(0.01)
            continue
        ultimo_id = fid
        t0 = time.time()

        try:
            resultados, intrusos = analisar_frame(frame)
            ia_estado["resultados"] = resultados
            current_intruders_count = intrusos
        except Exception as e:
            _avisar_uma_vez("analise", f"Falha na análise do frame: {e!r}")
            ia_estado["resultados"] = []
            current_intruders_count = 0
            time.sleep(0.2)

        # limita a IA para não competir com a leitura da câmera
        time.sleep(max(0.0, IA_INTERVALO_MIN - (time.time() - t0)))


def capturar_camera_2(url):
    global frame_atual_2, frame_limpo_2, camera_2_online, fps_2

    print("Iniciando captura CAM_02...")
    cap = None
    contador = 0
    inicio = time.time()

    while True:
        if cap is None:
            cap = abrir_rtsp(url)
            if cap is None:
                camera_2_online = False
                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(standby, "CAM_02 // AIBOX OFFLINE", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                with lock:
                    frame_atual_2 = standby
                    frame_limpo_2 = None
                    frame_id[2] += 1
                time.sleep(5)
                continue
            print("CAM_02 conectada via RTSP.")

        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            camera_2_online = False
            cap.release()
            cap = None
            time.sleep(1)
            continue

        camera_2_online = True
        contador += 1
        tempo = time.time() - inicio
        if tempo >= 1:
            fps_2 = contador / tempo
            contador = 0
            inicio = time.time()

        with lock:
            frame_atual_2 = frame
            frame_limpo_2 = frame
            frame_id[2] += 1


# Inicia as threads de forma independente
# (Se o arquivo roda como __main__, o uvicorn importa "app.main" de novo; iniciar aqui também
#  abriria a webcam duas vezes e a segunda cópia ficaria sem câmera.)
if __name__ != "__main__":
    threading.Thread(target=capturar_camera_1, args=(C.CAMERA_RTSP_1,), daemon=True).start()
    threading.Thread(target=loop_ia, daemon=True).start()
    threading.Thread(target=capturar_camera_2, args=(C.CAMERA_RTSP_2,), daemon=True).start()


def gerar_video(cam_id):
    """Stream MJPEG: envia somente frames NOVOS (antes reenviava o mesmo frame várias vezes)."""
    ultimo = -1
    while True:
        with lock:
            fid = frame_id[2] if cam_id == 2 else frame_id[1]
            src = frame_atual_2 if cam_id == 2 else frame_atual_1
        if src is None or fid == ultimo:
            time.sleep(0.01)
            continue
        ultimo = fid

        frame = src
        if frame.shape[1] > 960:  # reduz streams grandes (RTSP 1080p) para o navegador
            esc = 960 / frame.shape[1]
            frame = cv2.resize(frame, (960, int(frame.shape[0] * esc)))

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


INDEX_PATH = Path(__file__).resolve().parent / "index.html"

for _pasta in (Path(__file__).resolve().parent / "static", BASE_DIR / "static"):
    if _pasta.is_dir():
        app.mount("/static", StaticFiles(directory=str(_pasta)), name="static")  # /static/logo.jpeg
        break

@app.get("/", response_class=HTMLResponse)
def dashboard():
    if INDEX_PATH.exists():
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(f"<h3>index.html não encontrado em {INDEX_PATH}</h3>", status_code=500)

@app.get("/video/{cam_id}")
def video(cam_id: int):
    return StreamingResponse(gerar_video(cam_id), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/api/v1/classroom/status")
def status():
    att = db.get_today_attendance()
    return {
        "room_id": "JPPN_LAB_01",
        "camera_1_online": camera_1_online,
        "camera_1_source": camera_1_fonte,
        "camera_2_online": camera_2_online,
        "ai_ready": det is not None and rec is not None,
        "ai": ia_info,
        "authorized_count": len(reconhecidos),
        "current_intruders_count": current_intruders_count,
        "fps_1": round(fps_1, 2),
        "fps_2": round(fps_2, 2)
    }

@app.get("/api/v1/recognized")
def get_recognized():
    """Alunos reconhecidos hoje (para o dashboard). 'live' = visto nos últimos 3 segundos."""
    agora = time.time()
    with reg_lock:
        itens = sorted(reconhecidos.values(), key=lambda r: r["last_seen"], reverse=True)
        return [{
            "rm": r["rm"], "name": r["name"], "turma": r["turma"],
            "confidence": r["confidence"],
            "first_seen": int(r["first_seen"] * 1000),
            "last_seen": int(r["last_seen"] * 1000),
            "live": (agora - r["last_seen"]) <= 3,
        } for r in itens]

@app.get("/api/v1/intruders")
def get_intruders():
    """Desconhecidos confirmados: UM item por pessoa, com o horário da 1ª detecção."""
    agora = time.time()
    with reg_lock:
        itens = sorted((d for d in desconhecidos if d["confirmado"]), key=lambda d: d["first_seen"], reverse=True)
        return [{"id": d["numero"], "label": f"Desconhecido #{d['numero']}", "camera_id": "CAM_01",
                 "first_seen": int(d["first_seen"] * 1000), "last_seen": int(d["last_seen"] * 1000),
                 "live": agora - d["last_seen"] <= 3, "thumb": d["thumb"]} for d in itens]

@app.get("/api/v1/students")
def get_students():
    return db.get_students()

@app.post("/api/v1/students")
def add_student(data: dict):
    name = data.get("name")
    turma = data.get("turma", "1ECA")
    if not name: return JSONResponse({"detail": "Nome obrigatório"}, status_code=400)
    student_id = db.add_student(name, turma)
    return {"id": student_id, "name": name, "turma": turma}

def _frame_limpo(camera):
    with lock:
        if camera == "CAM_01":
            src = frame_limpo_1 if camera_1_online else None
        else:
            src = frame_limpo_2 if camera_2_online else None
        return src.copy() if src is not None else None

@app.post("/api/v1/students/{student_id}/sample")
def capture_sample(student_id: int, camera: str = "CAM_01"):
    if det is None or rec is None:
        return JSONResponse({"detail": "IA não inicializada (modelos .onnx não encontrados)"}, status_code=500)

    # Junta até 3 embeddings de frames diferentes (sem caixas desenhadas) e faz a média:
    # o cadastro fica bem mais robusto a pequenas variações de pose/luz.
    embeddings = []
    houve_frame = False
    for _ in range(8):
        img = _frame_limpo(camera)
        if img is not None:
            houve_frame = True
            faces = detectar_rostos(img)
            if faces:
                maior = max(faces, key=lambda f: f[2] * f[3])
                feat = extrair_embedding(img, maior)
                if feat is not None:
                    embeddings.append(feat)
                    if len(embeddings) >= 3:
                        break
        time.sleep(0.2)

    if not houve_frame:
        return JSONResponse({"detail": "Frame indisponível (câmera offline)"}, status_code=400)
    if not embeddings:
        return JSONResponse({"detail": "Nenhum rosto detetado"}, status_code=400)

    media = np.mean(embeddings, axis=0)
    media = (media / np.linalg.norm(media)).astype(np.float32)
    db.update_embedding(student_id, media.tobytes())
    invalidar_cache_alunos()
    return {"status": "success", "message": "Biometria capturada", "amostras": len(embeddings)}

@app.delete("/api/v1/students/{student_id}")
def delete_student(student_id: int):
    db.delete_student(student_id)
    invalidar_cache_alunos()
    with reg_lock:
        reconhecidos.pop(f"RM{student_id:04d}", None)
    return {"status": "success"}

@app.get("/api/v1/attendance")
def get_attendance():
    return db.get_today_attendance()

@app.get("/api/v1/events")
def get_events():
    return db.get_pending_events()

@app.post("/api/v1/session/reset")
def reset_session():
    db.reset_session()
    _ultimo_registro.clear()
    global _contador_desconhecidos
    with reg_lock:
        reconhecidos.clear()
        desconhecidos.clear()
        _contador_desconhecidos = 0
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    # reload=False: com reload o uvicorn reinicia o processo e a webcam pode ficar presa/ocupada
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)