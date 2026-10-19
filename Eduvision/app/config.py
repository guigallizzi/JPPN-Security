from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent.parent   # pasta do projeto (eduvision/)


def _achar(padrao_busca, padrao_final):
    """Procura um arquivo dentro do projeto (ignorando venv/.git). Se não achar, usa o caminho padrão."""
    try:
        for p in BASE_DIR.rglob(padrao_busca):
            if not {".venv", "venv", ".git", "__pycache__", "wheels"} & set(p.parts):
                return p
    except OSError:
        pass
    return padrao_final


# ---------------------------------------------------------------------------
# Banco de dados e modelos de IA
# (se já existe um .db / .onnx no projeto, usa o que encontrar, para não perder os alunos cadastrados)
# ---------------------------------------------------------------------------
# Usa o banco do JPPN (esquema com rm/turma). O database.db antigo tem outro esquema e quebra o log de presença.
DB_PATH = _achar("jppn_security.db", BASE_DIR / "data" / "jppn_security.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MODELS_DIR = _achar("face_detection_yunet_2023mar.onnx", BASE_DIR / "models" / "x").parent

print(f"[CONFIG] Banco de dados: {DB_PATH}")
print(f"[CONFIG] Modelos de IA:  {MODELS_DIR}")

# ---------------------------------------------------------------------------
# Câmeras (RTSP)
# ---------------------------------------------------------------------------
PORTA = 554
USUARIO = "Fiap"
SENHA_REAL = "fiap@2026"

# A câmera recusou a senha codificada (%40) com 401 -> usa a senha como está.
# Se ainda der 401, troque para True e teste de novo.
CODIFICAR_SENHA = False
SENHA = quote(SENHA_REAL, safe="") if CODIFICAR_SENHA else SENHA_REAL

IP_CAM_1 = "192.168.50.108"
IP_CAM_2 = "192.168.50.109"


def montar_rtsp(ip, canal=1, subtipo=1):
    # subtipo=1 -> substream (leve, menos lag) | subtipo=0 -> stream principal (mais detalhe)
    return (
        f"rtsp://{USUARIO}:{SENHA}@{ip}:{PORTA}"
        f"/cam/realmonitor?channel={canal}&subtype={subtipo}"
    )


CAMERA_RTSP_1 = montar_rtsp(IP_CAM_1)
CAMERA_RTSP_2 = montar_rtsp(IP_CAM_2)

# ---------------------------------------------------------------------------
# Opcionais (descomente para ajustar)
# ---------------------------------------------------------------------------
# WEBCAM_INDEX = 0              # índice da webcam local (reserva quando o AIBOX está offline)
# WEBCAM_LARGURA = 960
# WEBCAM_ALTURA = 540
# LIMIAR_SIMILARIDADE = 0.363   # sensibilidade do reconhecimento (menor = mais permissivo)