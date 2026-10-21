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
with db.get_connection() as _c:  # amostras da biometria avançada (1 linha por amostra)
    _c.execute("CREATE TABLE IF NOT EXISTS face_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "student_id INTEGER NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL)")
    _c.commit()

IS_WINDOWS = sys.platform.startswith("win")

# deixa CPU livre para a thread da câmera (a IA do OpenCV usaria todos os núcleos)
cv2.setNumThreads(max(2, (os.cpu_count() or 4) // 2))

# ---------------------------------------------------------------------------
# Parâmetros de reconhecimento
# ---------------------------------------------------------------------------
LIMIAR_SIMILARIDADE = getattr(C, "LIMIAR_SIMILARIDADE", 0.45)  # SFace (cosseno, média das 3 melhores amostras). Ajustável no config.py
DEBUG_IA = True               # mostra a similaridade no vídeo/console para diagnóstico (ponha False depois)
LIMIAR_INTRUSO = 0.78         # confiança mínima do detector para contar como intruso (antes 0.85: rosto aparecia como desconhecido mas não contava)
LARGURA_DETECCAO = 2560       # a detecção roda numa imagem reduzida (maior = acha rostos mais distantes, mais lento)
CONF_DETECTOR = 0.75          # confiança mínima do YuNet para considerar um rosto (antes 0.8)
MIN_LADO_RECONHECER = 16      # rostos menores que isso (px) ficam "distante": nem reconhece nem acusa intruso
LADO_ROSTO_BOM = 40           # abaixo disso o rosto é "pequeno": exige similaridade maior (+BONUS_PEQUENO)
BONUS_PEQUENO = 0.08
ZOOM_LARGURA = 220            # zoom na cabeça: amplia o recorte até ~essa largura (px)
ZOOM_MAX = 4.0                # ampliação máxima do zoom digital
MARGEM_AMBIGUIDADE = 0.05     # o melhor aluno precisa vencer o 2º colocado por essa margem (evita confundir parecidos)
# --- Biometria avançada (ArcFace / InsightFace). Se o .onnx não existir, usa o SFace como antes.
ARC_MODELO = getattr(C, "ARC_MODELO", "w600k_mbf.onnx")   # ou "w600k_r50.onnx" (mais preciso, ~8x mais lento)
LIMIAR_ARC = getattr(C, "LIMIAR_ARC", 0.35)               # cosseno mínimo no ArcFace
MARGEM_ARC = 0.08                                         # folga mínima sobre o 2º colocado no ArcFace
# --- Anti-troca: pontuação acumulada por pessoa rastreada (decai a cada nova observação do rosto)
DECAIMENTO = 0.85             # peso das observações antigas
SCORE_TROCA = 1.0             # para trocar o nome de alguém já identificado, o novo precisa vencer por essa folga
REVERIFICAR_S = 1.0           # pessoa já identificada: recalcula a biometria só a cada N s (economiza CPU)
CONFIRMAR_ALUNO = 3           # reconhecimentos do mesmo aluno para confirmar presença (roteiro: não confirmar com 1 frame)
AMOSTRAS_POR_CAPTURA = 10     # embeddings coletados a cada clique em CAPTURAR
MAX_AMOSTRAS = 40             # máximo de amostras guardadas por aluno (recapturar ACUMULA poses)
MIN_LADO_CADASTRO = 50        # rosto mínimo (px) no cadastro: amostra pequena/borrada "parece com todo mundo"
CONF_CADASTRO = 0.85          # confiança mínima do detector na amostra de cadastro
# --- Rastreamento de pessoas (YOLO): reconhece o rosto UMA vez e mantém o nome na pessoa
#     mesmo de lado / olhando para baixo / de costas, enquanto ela continuar no lugar.
USAR_TRACKING = getattr(C, "USAR_TRACKING", True)
CONF_PESSOA = 0.35            # confiança mínima do YOLO para "pessoa"
IMGSZ_YOLO = 960              # resolução de entrada do YOLO (maior = acha pessoas menores, mais lento)
IOU_MESMA_PESSOA = 0.30       # sobreposição mínima entre caixas p/ ser a mesma pessoa no frame seguinte
TTL_TRILHA = 8.0              # segundos que uma pessoa some (oclusão) antes de perder o nome
FRONTAL_MIN, FRONTAL_MAX = 0.20, 0.80  # rosto "de frente" (nariz entre os olhos): só esses contam como intruso
PROCESSAR_A_CADA = 3          # (não usado: a IA roda em thread própria)
ROI_CAM_1 = getattr(C, "ROI_CAM_1", None)  # área das cadeiras (x1, y1, x2, y2) em fração da imagem; None = imagem toda
LARGURA_RAPIDA = 960          # detector rápido (só posição dos quadros) roda nessa largura a cada frame
YOLO_A_CADA = 3               # roda o YOLO (pesado) 1 a cada N ciclos de IA; entre eles reaproveita as pessoas
IA_INTERVALO_MIN = 0.12       # s entre análises da IA (~8/s). Maior = mais FPS de vídeo, menos reatividade
LIMIAR_MESMO_DESCONHECIDO = 0.45  # similaridade p/ religar um desconhecido que SAIU e VOLTOU (antes 0.40: juntava 2 pessoas)
LIMIAR_FORTE_DESCONHECIDO = 0.60  # similaridade alta: mesma pessoa mesmo se ela "pulou" de lugar
VIVO_S = 2.0                  # desconhecido visto há menos que isso está "na tela": não pode ser outro rosto em outro lugar
CONFIRMAR_APOS = 2            # detecções (de frente) mínimas para confirmar um desconhecido
ESPERA_INTRUSO_S = 4.0        # o rosto precisa continuar SEM ser reconhecido por N s antes de virar intruso
                              # (aluno que demora a ser reconhecido ao chegar não cai no histórico)
PROTECAO_ALUNO_S = 3.0        # rosto no MESMO lugar onde um aluno foi reconhecido há < N s não vira intruso (oscilação)
JANELA_CONFIRMACAO = 6        # segundos máx. entre essas detecções
MIN_LADO_ROSTO = 24           # rostos menores que isso (px) não viram intruso (antes 40: rosto distante nunca era contado)
COOLDOWN_PRESENCA = 60        # segundos entre registros de presença do mesmo aluno
COOLDOWN_INTRUSO = 10         # segundos entre registros de evento de intruso
CACHE_ALUNOS_SEG = 3          # recarrega embeddings do banco a cada N segundos

COR_RECONHECIDO = (0, 255, 120)
COR_INTRUSO = (0, 140, 255)
COR_VERIFICANDO = (0, 220, 255)   # amarelo: rosto não reconhecido, ainda confirmando

# ---------------------------------------------------------------------------
# Modelos de IA (YuNet & SFace)
# ---------------------------------------------------------------------------
det = None
rec = None
det_rapido = None   # 2º detector, só para seguir os rostos a cada frame (sem disputar lock com a IA)

model_det_path = str(C.MODELS_DIR / "face_detection_yunet_2023mar.onnx")
model_rec_path = str(C.MODELS_DIR / "face_recognition_sface_2021dec.onnx")

try:
    if not os.path.exists(model_det_path):
        print(f"[ERRO] Modelo de detecção não encontrado: {model_det_path}")
    elif not os.path.exists(model_rec_path):
        print(f"[ERRO] Modelo de reconhecimento não encontrado: {model_rec_path}")
    else:
        det = cv2.FaceDetectorYN.create(model_det_path, "", (320, 320), CONF_DETECTOR, 0.3, 5000)
        rec = cv2.FaceRecognizerSF.create(model_rec_path, "")
        det_rapido = cv2.FaceDetectorYN.create(model_det_path, "", (320, 320), CONF_DETECTOR, 0.3, 5000)
        print("Modelos de IA carregados com sucesso!")
except Exception as e:
    det = None
    rec = None
    print(f"[ERRO] Falha ao carregar modelos de IA: {e}")

arc, arc_in = None, None
try:
    _arc_path = C.MODELS_DIR / ARC_MODELO
    if _arc_path.exists():
        import onnxruntime as ort
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = max(2, (os.cpu_count() or 4) // 2)
        arc = ort.InferenceSession(str(_arc_path), _so, providers=["CPUExecutionProvider"])
        arc_in = arc.get_inputs()[0].name
        print(f"Biometria AVANÇADA ativa (ArcFace {ARC_MODELO})")
    else:
        print(f"[AVISO] Biometria avançada desligada: {ARC_MODELO} não está em {C.MODELS_DIR} (usando SFace)")
except Exception as e:
    arc = None
    print(f"[AVISO] Biometria avançada desligada ({e!r}); usando SFace")

MODELO_BIO = "arc" if arc is not None else "sface"
if arc is not None:  # limiares próprios do ArcFace
    LIMIAR_SIMILARIDADE = LIMIAR_ARC
    MARGEM_AMBIGUIDADE = MARGEM_ARC
    LIMIAR_MESMO_DESCONHECIDO = 0.38
    LIMIAR_FORTE_DESCONHECIDO = 0.55
SCORE_CONFIRMA = CONFIRMAR_ALUNO * LIMIAR_SIMILARIDADE  # ~3 boas observações para fixar o nome

if det is None or rec is None:
    print("[AVISO] Reconhecimento facial DESATIVADO. Baixe os modelos do OpenCV Zoo "
          "(face_detection_yunet_2023mar.onnx e face_recognition_sface_2021dec.onnx) "
          f"e coloque em: {C.MODELS_DIR}")

yolo = None
if USAR_TRACKING:
    try:
        from ultralytics import YOLO
        _cands = [Path(getattr(C, "YOLO_MODEL", "")) if getattr(C, "YOLO_MODEL", None) else None,
                  C.MODELS_DIR / "yolov8n.pt", BASE_DIR / "models" / "yolov8n.pt", BASE_DIR / "yolov8n.pt"]
        _yolo_path = next((p for p in _cands if p and p.exists()), None)
        if _yolo_path is None:
            print(f"[AVISO] Rastreamento DESATIVADO: coloque yolov8n.pt em {C.MODELS_DIR}")
        else:
            yolo = YOLO(str(_yolo_path))
            yolo.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)  # aquece
            print(f"Rastreamento de pessoas ATIVO ({_yolo_path.name})")
    except Exception as e:
        yolo = None
        print(f"[AVISO] Rastreamento DESATIVADO ({e!r}). Seguindo só com reconhecimento facial.")

app = FastAPI(title="JPPN Security School", version="7.5")

# ---------------------------------------------------------------------------
# Estado compartilhado
# ---------------------------------------------------------------------------
frame_atual_1 = None      # frame com caixas desenhadas (stream)
frame_limpo_1 = None      # frame sem anotações (usado no cadastro da biometria)
camera_1_online = False
camera_1_fonte = "NENHUMA"   # "AIBOX", "WEBCAM" ou "NENHUMA"
fps_1 = 0
current_intruders_count = 0

lock = threading.Lock()          # protege os frames
ia_lock = threading.Lock()       # protege det/rec (não são thread-safe)
cache_lock = threading.Lock()

_cache_alunos = {"t": 0.0, "dados": []}
_ultimo_registro = {}            # rm -> timestamp do último log de presença
_votos = {}                      # rm -> {"hits", "t0", "last"} para confirmar o aluno em vários frames
_ultimo_intruso = 0.0
_alunos_recentes = []           # (cx, cy, lado, t, rótulo, rm) de rostos reconhecidos como aluno (sem tracking)
_erros_vistos = set()

frame_id = {1: 0}                # contador de frames novos (o stream só envia frames novos)
ia_estado = {"resultados": []}   # últimos rostos detetados pela thread de IA
reconhecidos = {}                # rm -> dados do aluno reconhecido hoje (vai para o dashboard)
reg_lock = threading.Lock()
ia_info = {"alunos_carregados": 0, "rostos": 0, "melhor_nome": None, "melhor_sim": None,
           "ultimo_erro": None, "limiar": LIMIAR_SIMILARIDADE, "pessoas": 0, "tracking": False, "sem_biometria_avancada": 0}
_ciclo_yolo = 0
trilhas = {}                     # id -> pessoa rastreada (caixa, aluno identificado, votos)
_prox_trilha = 1
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
            print(f"[RTSP] conectado: {frame.shape[1]}x{frame.shape[0]}")
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
    if MODELO_BIO == "arc":
        try:
            with db.get_connection() as conn:
                rows = conn.execute("SELECT s.id, s.name, s.turma, f.vec FROM face_samples f "
                                    "JOIN students s ON s.id = f.student_id WHERE f.model = ? ORDER BY f.id",
                                    (ARC_MODELO,)).fetchall()
                total = conn.execute("SELECT COUNT(*) FROM students WHERE embedding IS NOT NULL").fetchone()[0]
            por_aluno = {}
            for r in rows:
                a = por_aluno.setdefault(r[0], ({"id": r[0], "name": r[1], "turma": r[2]}, []))
                v = np.frombuffer(r[3], dtype=np.float32)
                if v.size == 512 and np.linalg.norm(v) > 0:
                    a[1].append(v / np.linalg.norm(v))
            dados = [(a, np.asarray(g, dtype=np.float32)) for a, g in por_aluno.values() if g]
            ia_info["sem_biometria_avancada"] = max(0, total - len(dados))
        except Exception as e:
            _avisar_uma_vez("db", f"Erro ao ler amostras ArcFace: {e}")
        with cache_lock:
            _cache_alunos["dados"] = dados
            _cache_alunos["t"] = agora
        ia_info["alunos_carregados"] = len(dados)
        return dados
    try:
        for s in db.get_students_with_samples():
            s = dict(s)
            emb = s.get("embedding")
            if not emb:
                continue
            v = np.frombuffer(emb, dtype=np.float32)
            if v.size == 0 or v.size % 128 != 0:
                continue
            g = v.reshape(-1, 128)                       # 1 linha por amostra
            n = np.linalg.norm(g, axis=1, keepdims=True)
            g = g[n[:, 0] > 0] / n[n[:, 0] > 0]
            if len(g) == 0:
                continue
            dados.append((s, g.astype(np.float32)))
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


ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def extrair_arc(img, face):
    """Embedding ArcFace normalizado (512,): alinha pelos 5 pontos do YuNet no padrão 112x112 do InsightFace."""
    pts = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    alinhado = cv2.warpAffine(img, M, (112, 112), borderValue=0)
    blob = cv2.dnn.blobFromImage(alinhado, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    feat = arc.run(None, {arc_in: blob})[0].flatten().astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else None


def extrair_embedding(img, face):
    """Embedding do modelo ativo (ArcFace se disponível, senão SFace)."""
    if MODELO_BIO == "arc":
        return extrair_arc(img, face)
    return extrair_sface(img, face)


def extrair_sface(img, face):
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


def rastrear_desconhecido(frame, face, feat, agora, ocupados, tid=None, conta=True):
    """Cada desconhecido vira UMA ficha (1 evento na confirmação). A ligação rosto -> ficha é feita por:
    1) trilha do YOLO (mesma pessoa), 2) mesma posição há pouco tempo, 3) embedding (pessoa que saiu e voltou).
    'ocupados' = fichas já usadas neste frame: dois rostos no mesmo frame NUNCA caem na mesma ficha.
    Retorna o nº do desconhecido (None enquanto não confirmado)."""
    global _contador_desconhecidos
    novo = None
    x, y, w, h = [float(v) for v in face[:4]]
    cx, cy, lado = x + w / 2, y + h / 2, max(w, h, 1.0)

    def _sim(c):
        return max(float(np.dot(feat, g)) for g in c["galeria"])

    def _livre_de_trilha(c):  # ficha não está presa a OUTRA pessoa rastreada que ainda existe
        return c.get("tid") is None or c["tid"] == tid or c["tid"] not in trilhas

    with reg_lock:
        desconhecidos[:] = [d for d in desconhecidos if d["confirmado"] or agora - d["last_seen"] < 30]
        livres = [c for c in desconhecidos if id(c) not in ocupados]
        d = None
        if tid is not None:                                   # 1) mesma pessoa rastreada
            d = next((c for c in livres if c.get("tid") == tid), None)
        if d is None:                                         # 2) mesmo lugar, visto agora há pouco
            perto = [(((c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2) ** 0.5 / lado, c) for c in livres
                     if agora - c["last_seen"] < VIVO_S and _livre_de_trilha(c)]
            perto = [pc for pc in perto if pc[0] < 1.5]
            if perto:
                d = min(perto, key=lambda pc: pc[0])[1]
        if d is None:                                         # 3) embedding
            melhor, cand = -1.0, None
            for c in livres:
                if not _livre_de_trilha(c):
                    continue
                sim = _sim(c)
                vivo = agora - c["last_seen"] < VIVO_S       # está na tela em outro lugar: só com similaridade alta
                lim = LIMIAR_FORTE_DESCONHECIDO if vivo else LIMIAR_MESMO_DESCONHECIDO
                if sim >= lim and sim > melhor:
                    melhor, cand = sim, c
            d = cand

        if d is None and not conta:
            return None  # rosto de lado/fraco: só segue fichas que já existem, não cria alerta novo
        if d is not None and not conta:
            d["last_seen"] = agora
        elif d is not None:
            if not d["confirmado"] and agora - d["last_seen"] > JANELA_CONFIRMACAO:
                d["hits"], d["first_seen"] = 0, agora
            d["hits"] += 1
            d["last_seen"] = agora
            sim = _sim(d)
            if 0.25 <= sim < 0.8 and len(d["galeria"]) < 8:
                d["galeria"].append(feat)  # guarda poses diferentes da MESMA pessoa (sem misturar pessoas)
        else:
            d = {"hits": 1, "first_seen": agora, "last_seen": agora, "galeria": [feat],
                 "confirmado": False, "numero": None, "thumb": None}
            desconhecidos.append(d)
        d["cx"], d["cy"] = cx, cy
        if tid is not None:
            d["tid"] = tid
        ocupados.add(id(d))
        if not d["confirmado"] and d["hits"] >= CONFIRMAR_APOS and agora - d["first_seen"] >= ESPERA_INTRUSO_S:
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


def _descartar_desconhecido(tid):
    """A pessoa rastreada foi reconhecida como aluno: a ficha de desconhecido dela era falso alarme."""
    with reg_lock:
        desconhecidos[:] = [d for d in desconhecidos if d.get("tid") != tid]


def _descartar_pendente_perto(face, agora):
    """Aluno reconhecido aqui: apaga a ficha de desconhecido AINDA NÃO confirmada nesse mesmo lugar
    (era o próprio aluno antes de ser reconhecido)."""
    x, y, w, h = [float(v) for v in face[:4]]
    cx, cy, lado = x + w / 2, y + h / 2, max(w, h, 1.0)
    with reg_lock:
        desconhecidos[:] = [d for d in desconhecidos if d["confirmado"] or agora - d["last_seen"] > VIVO_S
                            or ((d["cx"] - cx) ** 2 + (d["cy"] - cy) ** 2) ** 0.5 > 1.5 * lado]


def _perto_de_aluno(face, inf, usados, agora):
    """Sem tracking: aluno reconhecido AQUI agora há pouco e cuja similaridade só caiu um pouco neste frame?
    Só mantém o nome se o rosto ainda PARECE com esse aluno (outra pessoa no lugar dele não herda o nome)
    e se o aluno não está em outro rosto neste frame."""
    x, y, w, h = [float(v) for v in face[:4]]
    cx, cy = x + w / 2, y + h / 2
    _alunos_recentes[:] = [a for a in _alunos_recentes if agora - a[3] < PROTECAO_ALUNO_S]
    if inf["melhor"] is None or inf["sim"] < LIMIAR_SIMILARIDADE - 0.12:
        return None
    rm = _rm(inf["melhor"])
    if rm in usados:
        return None
    for a in reversed(_alunos_recentes):
        if a[5] == rm and ((a[0] - cx) ** 2 + (a[1] - cy) ** 2) ** 0.5 < 0.8 * max(a[2], w):
            usados.add(rm)
            return a[4]
    return None


def _candidato_intruso(inf):
    """Rosto nítido, de frente e grande o bastante para virar intruso."""
    return (inf["feat"] is not None and inf["conf"] >= LIMIAR_INTRUSO and inf["frontal"]
            and min(inf["face"][2], inf["face"][3]) >= MIN_LADO_ROSTO)


def _frontal(face):
    """Rosto de frente? Usa os landmarks do YuNet: nariz entre os dois olhos."""
    rex, lex, nx = float(face[4]), float(face[6]), float(face[8])
    d = lex - rex
    if abs(d) < 1e-3:
        return False
    r = (nx - rex) / d
    return FRONTAL_MIN <= r <= FRONTAL_MAX


def _identificar(feat, alunos):
    """Compara com a galeria de cada aluno. Retorna (aluno_aceito|None, melhor_aluno, melhor_sim)."""
    melhor, melhor_sim, segundo = None, -1.0, -1.0
    for aluno, galeria in alunos:
        sims = np.sort(galeria @ feat)
        sim = float(np.mean(sims[-3:]))  # média das 3 melhores amostras (1 amostra ruim não "casa" com todo mundo)
        if sim > melhor_sim:
            segundo = melhor_sim
            melhor_sim, melhor = sim, aluno
        elif sim > segundo:
            segundo = sim
    claro = len(alunos) < 2 or (melhor_sim - segundo) >= MARGEM_AMBIGUIDADE
    ok = melhor is not None and melhor_sim >= LIMIAR_SIMILARIDADE and claro
    return (melhor if ok else None), melhor, melhor_sim


def _rm(aluno):
    return f"RM{aluno['id']:04d}"


def _presenca(aluno, sim, agora):
    nome, rm, turma = aluno["name"], _rm(aluno), aluno.get("turma", "1ECA")
    registrar_reconhecido(rm, nome, turma, sim, agora)
    if agora - _ultimo_registro.get(rm, 0) > COOLDOWN_PRESENCA:
        _ultimo_registro[rm] = agora
        db.log_attendance(rm=rm, name=nome, turma=turma, camera_id="CAM_01", confidence=float(sim))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    un = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / un if un > 0 else 0.0


def detectar_pessoas(frame, agora):
    """YOLO (classe pessoa) + rastreador simples por sobreposição. Retorna [(id, caixa xyxy)]."""
    global _prox_trilha, _ciclo_yolo
    _ciclo_yolo += 1
    if _ciclo_yolo % YOLO_A_CADA != 1 and trilhas:
        # pessoas sentadas quase não se mexem: reaproveita as caixas do último YOLO
        return [(tid, t["box"]) for tid, t in trilhas.items() if agora - t["last"] < 1.5]
    r = yolo.predict(frame, classes=[0], conf=CONF_PESSOA, imgsz=IMGSZ_YOLO, verbose=False)[0]
    caixas = [] if r.boxes is None else [tuple(int(v) for v in b) for b in r.boxes.xyxy.cpu().numpy()]

    pares = sorted(((_iou(t["box"], c), tid, k) for tid, t in trilhas.items() for k, c in enumerate(caixas)),
                   reverse=True)
    usados_t, usados_c, saida = set(), set(), []
    for iou, tid, k in pares:
        if iou < IOU_MESMA_PESSOA:
            break
        if tid in usados_t or k in usados_c:
            continue
        usados_t.add(tid); usados_c.add(k)
        trilhas[tid].update(box=caixas[k], last=agora)
        saida.append((tid, caixas[k]))
    for k, c in enumerate(caixas):
        if k not in usados_c:
            tid = _prox_trilha; _prox_trilha += 1
            trilhas[tid] = {"box": c, "last": agora, "votos": {}, "cands": {}, "aluno": None,
                            "sim": 0.0, "desc": 0, "num": None, "score": {}, "verif": 0.0}
            saida.append((tid, c))
    for tid in [t for t, v in trilhas.items() if agora - v["last"] > TTL_TRILHA]:
        del trilhas[tid]
    return saida


def _por_rosto(frame, infos, agora, ocupados=None):
    """Modo sem rastreamento (ou rosto sem pessoa associada): decide rosto a rosto."""
    resultados, intrusos = [], 0
    ocupados = set() if ocupados is None else ocupados
    usados = {_rm(i["aluno"]) for i in infos if i["aluno"] is not None}  # alunos já presentes neste frame
    for inf in infos:
        face, box = inf["face"], inf["box"]
        if inf["pequeno"] or inf["feat"] is None:
            continue  # rosto pequeno demais para decidir: não desenha
        if inf["conflito"]:
            resultados.append((box, "Desconhecido", COR_INTRUSO))  # parecido com aluno já visto em outro rosto
            continue
        aluno = inf["aluno"]
        if aluno is not None:
            rm = _rm(aluno)
            v = _votos.get(rm)
            if v is None or agora - v["last"] > JANELA_CONFIRMACAO:
                v = _votos[rm] = {"hits": 0, "t0": agora, "last": agora}
            v["hits"] += 1
            v["last"] = agora
            if v["hits"] >= CONFIRMAR_ALUNO:
                _presenca(aluno, inf["sim"], agora)  # grava presença só após confirmar
            rot = aluno["name"]
            _descartar_pendente_perto(face, agora)
            _alunos_recentes.append((face[0] + face[2] / 2, face[1] + face[3] / 2, max(face[2], face[3]), agora, rot, rm))
            resultados.append((box, rot, COR_RECONHECIDO))
            continue
        rot_aluno = _perto_de_aluno(face, inf, usados, agora)
        if rot_aluno:  # aluno cuja similaridade oscilou neste frame: mantém o nome, sem alarme
            resultados.append((box, rot_aluno, COR_RECONHECIDO))
            continue
        numero = None
        try:  # rosto de lado/oscilando continua com o mesmo nº (não fica piscando)
            conta = _candidato_intruso(inf)
            numero = rastrear_desconhecido(frame, face, inf["feat"], agora, ocupados, conta=conta)
        except Exception as e:
            _avisar_uma_vez("rastreio", f"Falha ao rastrear desconhecido: {e!r}")
        if numero:
            intrusos += 1
            resultados.append((box, f"Desconhecido #{numero}", COR_INTRUSO))
        else:
            resultados.append((box, "Desconhecido", COR_INTRUSO))
    return resultados, intrusos


def _dono(rm, tid, agora):
    """Outra pessoa rastreada (viva) que já está com esse aluno."""
    for t2id, t2 in trilhas.items():
        if t2id != tid and t2["aluno"] is not None and _rm(t2["aluno"]) == rm and agora - t2["last"] < 2.0:
            return t2
    return None


def _atualizar_identidade(tid, t, inf, agora):
    """Pontuação acumulada com decaimento + exclusividade (um aluno não pode estar em 2 lugares)."""
    for k in t["score"]:
        t["score"][k] *= DECAIMENTO
    if inf["aluno"] is not None:
        rm = _rm(inf["aluno"])
        t["score"][rm] = t["score"].get(rm, 0.0) + inf["sim"]
        t["cands"][rm] = inf["aluno"]
        if t["aluno"] is not None and _rm(t["aluno"]) == rm:
            t["sim"], t["verif"] = inf["sim"], agora

    atual = _rm(t["aluno"]) if t["aluno"] is not None else None
    for rm, sc in sorted(t["score"].items(), key=lambda kv: kv[1], reverse=True):
        if sc < SCORE_CONFIRMA:
            break
        dono = _dono(rm, tid, agora)
        if dono is not None and sc < dono["score"].get(rm, 0.0) + SCORE_TROCA:
            continue  # outra pessoa já é esse aluno: só toma o nome com folga grande
        if rm != atual and (atual is None or sc >= t["score"].get(atual, 0.0) + SCORE_TROCA):
            if dono is not None:
                dono["aluno"] = None
            t["aluno"] = t["cands"][rm]
            t["sim"], t["verif"] = (inf["sim"] if inf["aluno"] is not None else sc / 6.0), agora
        break
    # muitas observações contra: perde o nome
    if t["aluno"] is not None and t["score"].get(_rm(t["aluno"]), 0.0) < 0.5 * SCORE_CONFIRMA:
        t["aluno"] = None


def _por_pessoa(frame, infos, pessoas, agora):
    """Modo com rastreamento: o nome (ou o nº do desconhecido) fica preso na PESSOA, não no rosto."""
    resultados, intrusos = [], 0
    ocupados = set()
    por_tid = {i["tid"]: i for i in infos if i["tid"] is not None}
    for tid, (x1, y1, x2, y2) in pessoas:
        t = trilhas[tid]
        t.setdefault("score", {})
        t.setdefault("verif", 0.0)
        inf = por_tid.get(tid)

        if inf is not None and inf["feat"] is not None:
            # se o aluno sugerido já está firme em outra pessoa, esse rosto não pode ser ele
            if inf["aluno"] is not None:
                dono = _dono(_rm(inf["aluno"]), tid, agora)
                if dono is not None and dono["score"].get(_rm(inf["aluno"]), 0) >= SCORE_CONFIRMA \
                        and inf["sim"] < dono["sim"] + MARGEM_AMBIGUIDADE:
                    inf["aluno"], inf["conflito"] = None, True
            _atualizar_identidade(tid, t, inf, agora)
            if t["aluno"] is None and inf["aluno"] is None and not inf["conflito"]:
                try:
                    n = rastrear_desconhecido(frame, inf["face"], inf["feat"], agora, ocupados, tid=tid,
                                              conta=_candidato_intruso(inf))
                    if n:
                        t["num"] = n
                except Exception as e:
                    _avisar_uma_vez("rastreio", f"Falha ao rastrear desconhecido: {e!r}")

        # desenha só no ROSTO; sem rosto visível mas já identificado: quadro na região da cabeça
        if inf is not None:
            caixa = tuple(int(v) for v in inf["box"])
        else:
            lw = x2 - x1
            caixa = (int(x1 + 0.25 * lw), int(y1), int(0.5 * lw), int(0.5 * lw))

        if t["aluno"] is not None:
            _descartar_desconhecido(tid)  # era aluno, não intruso (ficha pendente ou confirmada)
            t["num"] = None
            # nome estável da pessoa (não pisca entre alunos a cada frame)
            _presenca(t["aluno"], t["sim"], agora)
            resultados.append((caixa, t["aluno"]["name"], COR_RECONHECIDO))
        elif t.get("num"):
            intrusos += 1  # continua contando mesmo se a pessoa abaixar a cabeça
            resultados.append((caixa, f"Desconhecido #{t['num']}", COR_INTRUSO))
        elif inf is not None and inf["aluno"] is not None:
            resultados.append((caixa, inf["aluno"]["name"], COR_RECONHECIDO))
        elif inf is not None and inf["feat"] is not None:
            resultados.append((caixa, "Desconhecido", COR_INTRUSO))

    r2, i2 = _por_rosto(frame, [i for i in infos if i["tid"] is None], agora, ocupados)
    return resultados + r2, intrusos + i2


def _montar_info(img, face, face_frame, alunos, tid=None, calcular=True):
    """img/face: onde o embedding é extraído (pode ser o recorte ampliado). face_frame: coordenadas no frame."""
    lado = min(face_frame[2], face_frame[3])
    inf = {"face": face_frame, "box": face_frame[:4].astype(int), "conf": float(face[-1]), "usado": tid is not None,
           "pequeno": lado < MIN_LADO_RECONHECER, "frontal": _frontal(face), "tid": tid, "conflito": False,
           "feat": None, "aluno": None, "melhor": None, "sim": -1.0}
    if not inf["pequeno"] and calcular:
        try:
            inf["feat"] = extrair_embedding(img, face)
            if inf["feat"] is not None and alunos:
                inf["aluno"], inf["melhor"], inf["sim"] = _identificar(inf["feat"], alunos)
                # rosto pequeno = embedding menos confiável: só aceita com folga maior
                if (inf["aluno"] is not None and lado < LADO_ROSTO_BOM
                        and inf["sim"] < LIMIAR_SIMILARIDADE + BONUS_PEQUENO):
                    inf["aluno"] = None
                # parece MAIS com um desconhecido já confirmado do que com o aluno: não é o aluno
                if inf["aluno"] is not None:
                    with reg_lock:
                        galerias = [d["galeria"] for d in desconhecidos if d["confirmado"]]
                    if any(max(float(np.dot(inf["feat"], g)) for g in gal) >= inf["sim"] for gal in galerias):
                        inf["aluno"] = None
        except Exception as e:
            _avisar_uma_vez("reconhecimento", f"Falha no reconhecimento: {e!r}")
    return inf


def _rosto_ampliado(frame, caixa, evitar=()):
    """Zoom digital na cabeça de uma pessoa (parte de cima da caixa do YOLO) para achar rostos distantes."""
    x1, y1, x2, y2 = [int(v) for v in caixa]
    H, W = frame.shape[:2]
    x1, y1, x2 = max(x1, 0), max(y1, 0), min(x2, W)
    y2c = min(y1 + int(0.45 * (y2 - y1)), H)
    crop = frame[y1:y2c, x1:x2]
    if crop.size == 0 or crop.shape[1] < 10 or crop.shape[0] < 10:
        return None
    esc = min(ZOOM_MAX, max(1.0, ZOOM_LARGURA / crop.shape[1]))
    up = cv2.resize(crop, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC) if esc > 1 else crop
    with ia_lock:
        det.setInputSize((up.shape[1], up.shape[0]))
        _, faces = det.detect(up)
    if faces is None or len(faces) == 0:
        return None
    # maior rosto do recorte que NÃO seja um rosto já encontrado (pessoas lado a lado)
    for f in sorted(faces, key=lambda v: v[2] * v[3], reverse=True):
        f = f.copy()
        ff = f.copy()
        ff[:14] /= esc
        ff[[0, 4, 6, 8, 10, 12]] += x1  # x da caixa e dos 5 landmarks (2/3 são largura/altura)
        ff[[1, 5, 7, 9, 11, 13]] += y1  # y
        if not _ja_visto(ff, evitar):
            return up, f, ff
    return None


def _ja_visto(ff, rostos):
    """Centro do rosto cai dentro de um rosto já detectado?"""
    cx, cy = ff[0] + ff[2] / 2, ff[1] + ff[3] / 2
    return any(o[0] <= cx <= o[0] + o[2] and o[1] <= cy <= o[1] + o[3] for o in rostos)


def _exclusivo(infos):
    """No mesmo frame, um aluno só pode ser UM rosto: fica com o de maior similaridade."""
    melhor = {}
    for inf in infos:
        if inf["aluno"] is not None:
            rm = _rm(inf["aluno"])
            if rm not in melhor or inf["sim"] > melhor[rm]["sim"]:
                melhor[rm] = inf
    for inf in infos:
        if inf["aluno"] is not None and melhor[_rm(inf["aluno"])] is not inf:
            inf["aluno"], inf["conflito"] = None, True


def analisar_frame(frame):
    """Detecção + reconhecimento (+ rastreamento de pessoas se o YOLO estiver disponível).
    Retorna (lista de (box, label, cor), qtd_intrusos)."""
    global _ultimo_print_ia
    agora = time.time()
    alunos = carregar_alunos()
    rostos = [(frame, f, f) for f in detectar_rostos(frame)]

    pessoas = None
    if yolo is not None:
        try:
            pessoas = detectar_pessoas(frame, agora)
        except Exception as e:
            _avisar_uma_vez("yolo", f"Falha no rastreamento (seguindo só com rostos): {e!r}")
            pessoas = None
    ia_info["tracking"] = pessoas is not None

    # associa cada rosto à pessoa (centro do rosto na parte de cima do corpo); sem rosto -> zoom na cabeça
    # (caixas sobrepostas: cada rosto vai para a pessoa cuja CABEÇA esperada está mais perto; 1 rosto por pessoa)
    dono_rosto = [None] * len(rostos)
    if pessoas is not None:
        cands = []
        for k, (_, f, ff) in enumerate(rostos):
            cx, cy = ff[0] + ff[2] / 2, ff[1] + ff[3] / 2
            for tid, (x1, y1, x2, y2) in pessoas:
                if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.6 * (y2 - y1):
                    hx, hy = (x1 + x2) / 2, y1 + 0.12 * (y2 - y1)
                    cands.append((((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 / max(x2 - x1, 1), k, tid))
        com_rosto = set()
        for _, k, tid in sorted(cands):
            if dono_rosto[k] is None and tid not in com_rosto:
                dono_rosto[k] = tid
                com_rosto.add(tid)
        for tid, caixa in pessoas:
            if tid in com_rosto:
                continue
            r = _rosto_ampliado(frame, caixa, [ff for (_, _, ff) in rostos])
            if r is not None:
                rostos.append(r)
                dono_rosto.append(tid)
        ia_info["pessoas"] = len(pessoas)

    infos = []
    for (img, f, ff), tid in zip(rostos, dono_rosto):
        t = trilhas.get(tid) if tid is not None else None
        pular = t is not None and t.get("aluno") is not None and agora - t.get("verif", 0.0) < REVERIFICAR_S
        infos.append(_montar_info(img, f, ff, alunos, tid=tid, calcular=not pular))
    _exclusivo(infos)

    if pessoas is not None:
        resultados, intrusos = _por_pessoa(frame, infos, pessoas, agora)
    else:
        for i in infos:
            i["tid"] = None
        resultados, intrusos = _por_rosto(frame, infos, agora)

    melhores = [i for i in infos if i["melhor"] is not None]
    if melhores:
        b = max(melhores, key=lambda i: i["sim"])
        ia_info["melhor_nome"], ia_info["melhor_sim"] = b["melhor"]["name"], round(b["sim"], 3)
    ia_info["rostos"] = len(infos)
    if DEBUG_IA and infos and agora - _ultimo_print_ia > 2:
        _ultimo_print_ia = agora
        print(f"[IA] alunos={len(alunos)} rostos={len(infos)} pessoas={ia_info['pessoas']} "
              f"limiar={LIMIAR_SIMILARIDADE:.2f} sims=" +
              ", ".join(f"{i['melhor']['name'] if i['melhor'] else '-'}:{i['sim']:.2f}{'*' if i['aluno'] else ''}"
                        for i in infos if i["feat"] is not None))
    return resultados, intrusos


def desenhar(frame, resultados, fonte):
    saida = frame.copy()
    if ROI_CAM_1:  # mostra a área analisada
        h, w = saida.shape[:2]
        cv2.rectangle(saida, (int(ROI_CAM_1[0] * w), int(ROI_CAM_1[1] * h)),
                      (int(ROI_CAM_1[2] * w), int(ROI_CAM_1[3] * h)), (248, 189, 56), 1)
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

        anotado = desenhar(frame, _para_desenho(), fonte)
        with lock:
            frame_limpo_1 = frame
            frame_atual_1 = anotado
            frame_id[1] += 1


def _para_desenho():
    """Usa as caixas reposicionadas pelo seguidor rápido (se recentes); senão, as da IA."""
    rap = ia_estado.get("rapido")
    if rap and time.time() - rap[0] < 0.5:
        return rap[1]
    return ia_estado["resultados"]


def loop_caixas():
    """Seguidor rápido: detecta só a POSIÇÃO dos rostos em todo frame e move para lá
    os rótulos (nome/desconhecido) calculados pela IA, que roda mais devagar. Tira o atraso do quadro."""
    ultimo = -1
    base_ref, anterior = None, []
    while True:
        if det_rapido is None:
            time.sleep(1)
            continue
        with lock:
            frame = frame_limpo_1 if camera_1_online else None
            fid = frame_id[1]
        if frame is None or fid == ultimo:
            time.sleep(0.005)
            continue
        ultimo = fid
        try:
            h, w = frame.shape[:2]
            esc = max(1.0, w / LARGURA_RAPIDA)
            peq = cv2.resize(frame, (int(w / esc), int(h / esc))) if esc > 1 else frame
            det_rapido.setInputSize((peq.shape[1], peq.shape[0]))
            _, faces = det_rapido.detect(peq)
            rapidas = [] if faces is None else [tuple(int(v * esc) for v in f[:4]) for f in faces]

            atual = ia_estado["resultados"]
            if atual is not base_ref:          # chegou resultado novo da IA: reancora
                base_ref, base = atual, list(atual)
            else:                              # mesmo resultado: segue a partir da última posição
                base = anterior
            pares = []
            for i, (bx, _, _) in enumerate(base):
                bcx, bcy, lim = bx[0] + bx[2] / 2, bx[1] + bx[3] / 2, 1.5 * max(bx[2], bx[3], 20)
                for k, (x, y, fw, fh) in enumerate(rapidas):
                    d = ((x + fw / 2 - bcx) ** 2 + (y + fh / 2 - bcy) ** 2) ** 0.5
                    if d <= lim:
                        pares.append((d, i, k))
            pares.sort()
            mapa, usados = {}, set()
            for d, i, k in pares:
                if i not in mapa and k not in usados:
                    mapa[i] = k
                    usados.add(k)
            saida = [(rapidas[mapa[i]], lab, cor) if i in mapa else (bx, lab, cor)
                     for i, (bx, lab, cor) in enumerate(base)]
            anterior = saida
            ia_estado["rapido"] = (time.time(), saida)
        except Exception as e:
            _avisar_uma_vez("seguidor", f"Falha no seguidor rápido: {e!r}")
            time.sleep(0.2)


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
            ox = oy = 0
            if ROI_CAM_1:
                h, w = frame.shape[:2]
                ox, oy = int(ROI_CAM_1[0] * w), int(ROI_CAM_1[1] * h)
                frame = frame[oy:int(ROI_CAM_1[3] * h), ox:int(ROI_CAM_1[2] * w)]
            resultados, intrusos = analisar_frame(frame)
            if ox or oy:  # volta as caixas para a coordenada da imagem inteira
                resultados = [((int(b[0]) + ox, int(b[1]) + oy, int(b[2]), int(b[3])), lab, cor)
                              for b, lab, cor in resultados]
            ia_estado["resultados"] = resultados
            current_intruders_count = intrusos
        except Exception as e:
            _avisar_uma_vez("analise", f"Falha na análise do frame: {e!r}")
            ia_estado["resultados"] = []
            current_intruders_count = 0
            time.sleep(0.2)

        # limita a IA para não competir com a leitura da câmera
        time.sleep(max(0.0, IA_INTERVALO_MIN - (time.time() - t0)))


# Inicia as threads de forma independente
# (Se o arquivo roda como __main__, o uvicorn importa "app.main" de novo; iniciar aqui também
#  abriria a webcam duas vezes e a segunda cópia ficaria sem câmera.)
if __name__ != "__main__":
    threading.Thread(target=capturar_camera_1, args=(C.CAMERA_RTSP_1,), daemon=True).start()
    threading.Thread(target=loop_ia, daemon=True).start()
    threading.Thread(target=loop_caixas, daemon=True).start()


def gerar_video(cam_id):
    """Stream MJPEG: envia somente frames NOVOS (antes reenviava o mesmo frame várias vezes)."""
    ultimo = -1
    while True:
        with lock:
            fid = frame_id[1]          # sistema com UMA câmera (CAM_01 / .108)
            src = frame_atual_1
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
        "ai_ready": det is not None and rec is not None,
        "ai": ia_info,
        "authorized_count": len(reconhecidos),
        "current_intruders_count": current_intruders_count,
        "fps_1": round(fps_1, 2)
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
        src = frame_limpo_1 if camera_1_online else None  # só existe a CAM_01
        return src.copy() if src is not None else None

@app.post("/api/v1/students/{student_id}/sample")
def capture_sample(student_id: int, camera: str = "CAM_01"):
    if det is None or rec is None:
        return JSONResponse({"detail": "IA não inicializada (modelos .onnx não encontrados)"}, status_code=500)

    # Junta até 3 embeddings de frames diferentes (sem caixas desenhadas) e faz a média:
    # o cadastro fica bem mais robusto a pequenas variações de pose/luz.
    # Coleta várias amostras de frames diferentes (peça para a pessoa mexer levemente a cabeça).
    # Amostras quase idênticas são descartadas para guardar poses variadas.
    embeddings = []
    arcs = []
    houve_frame = False
    pequeno = False
    for _ in range(45):  # ~7 s: tempo para a pessoa girar a cabeça devagar
        img = _frame_limpo(camera)
        if img is not None:
            houve_frame = True
            faces = detectar_rostos(img)
            if faces:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                maior = faces[0]
                if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * maior[2] * maior[3]:
                    pass  # 2 rostos de tamanho parecido: não dá pra saber quem é o aluno, ignora o frame
                elif min(maior[2], maior[3]) < MIN_LADO_CADASTRO or float(maior[-1]) < CONF_CADASTRO \
                        or not _frontal(maior):
                    pequeno = True
                else:
                    feat = extrair_sface(img, maior)
                    fa = extrair_arc(img, maior) if MODELO_BIO == "arc" else None
                    ref = fa if fa is not None else feat
                    if feat is not None and (MODELO_BIO != "arc" or fa is not None) and \
                            all(float(np.dot(ref, e)) < 0.93 for e in (arcs if MODELO_BIO == "arc" else embeddings)):
                        embeddings.append(feat)
                        if fa is not None:
                            arcs.append(fa)
                        if len(embeddings) >= AMOSTRAS_POR_CAPTURA:
                            break
        time.sleep(0.15)

    if not houve_frame:
        return JSONResponse({"detail": "Frame indisponível (câmera offline)"}, status_code=400)
    if not embeddings:
        msg = "Rosto pequeno, de lado ou borrado: aproxime-se e olhe de frente para a câmera" if pequeno else "Nenhum rosto detetado"
        return JSONResponse({"detail": msg}, status_code=400)

    # descarta amostras que não batem com as demais (outra pessoa/borrão no meio da captura)
    ref_lista = arcs if arcs else embeddings
    if len(ref_lista) >= 3:
        M = np.asarray(ref_lista)
        media = (M @ M.T).mean(axis=1)
        manter = [i for i in range(len(ref_lista)) if media[i] >= np.median(media) - 0.15]
        embeddings = [embeddings[i] for i in manter]
        arcs = [arcs[i] for i in manter] if arcs else arcs
    # SUBSTITUI as amostras antigas (amostras ruins antigas faziam qualquer um virar esse aluno)
    galeria = embeddings[-MAX_AMOSTRAS:]
    db.update_embedding(student_id, np.asarray(galeria, dtype=np.float32).tobytes())
    total = len(galeria)
    if arcs:
        with db.get_connection() as conn:
            conn.execute("DELETE FROM face_samples WHERE student_id = ? AND model = ?", (student_id, ARC_MODELO))
            conn.executemany("INSERT INTO face_samples (student_id, model, vec) VALUES (?, ?, ?)",
                             [(student_id, ARC_MODELO, a.astype(np.float32).tobytes()) for a in arcs])
            # mantém só as MAX_AMOSTRAS mais recentes
            conn.execute("DELETE FROM face_samples WHERE student_id = ? AND model = ? AND id NOT IN ("
                         "SELECT id FROM face_samples WHERE student_id = ? AND model = ? ORDER BY id DESC LIMIT ?)",
                         (student_id, ARC_MODELO, student_id, ARC_MODELO, MAX_AMOSTRAS))
            total = conn.execute("SELECT COUNT(*) FROM face_samples WHERE student_id = ? AND model = ?",
                                 (student_id, ARC_MODELO)).fetchone()[0]
            conn.commit()
    invalidar_cache_alunos()
    return {"status": "success", "message": "Biometria capturada", "amostras": total,
            "modelo": ARC_MODELO if arcs else "sface"}

@app.delete("/api/v1/students/{student_id}")
def delete_student(student_id: int):
    db.delete_student(student_id)
    with db.get_connection() as conn:
        conn.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
        conn.commit()
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
    _votos.clear()
    trilhas.clear()
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