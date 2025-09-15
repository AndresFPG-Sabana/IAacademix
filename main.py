import os
import time
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# === Configuración desde variables de entorno ===
DATA_URL = os.getenv("DATA_URL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b:free")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# === FastAPI ===
app = FastAPI(title="Recomendador Backend Ligero")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {"ts": 0, "rows": []}


# Normaliza columnas esperadas
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "nombre": ["nombre", "Nombre", "NOMBRE"],
        "nivel de dificultad": ["nivel de dificultad", "nivel", "Nivel"],
        "subcategoria": ["subcategoria", "Subcategoria", "SUBCATEGORIA"],
        "descripción": ["descripción", "descripcion", "Descripcion"],
        "enlace": ["enlace", "link", "URL"],
        "tutorial": ["tutorial", "Tutorial"],
    }
    cols = {c: c for c in df.columns}
    for target, aliases in mapping.items():
        for a in aliases:
            if a in df.columns:
                cols[a] = target
                break
    df = df.rename(columns=cols)
    for c in mapping.keys():
        if c not in df.columns:
            df[c] = ""
    return df[list(mapping.keys())]


def load_data(force: bool = False):
    now = time.time()
    if not force and _cache["rows"] and now - _cache["ts"] < CACHE_TTL_SECONDS:
        return _cache["rows"]

    if not DATA_URL:
        raise RuntimeError("No se configuró DATA_URL")

    resp = requests.get(DATA_URL, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"No se pudo descargar datos ({resp.status_code})")

    if DATA_URL.lower().endswith(".csv") or "output=csv" in DATA_URL.lower():
        # read CSV from text content
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
    else:
        # read Excel from binary content
        from io import BytesIO
        df = pd.read_excel(io=BytesIO(resp.content))

    df = _normalize(df)
    rows = df.fillna("").to_dict(orient="records")

    _cache["rows"] = rows
    _cache["ts"] = now
    return rows


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datos")
def datos():
    try:
        rows = load_data()
        return {"herramientas": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConsultaIn(BaseModel):
    mensaje: str


@app.post("/consulta")
def consulta(body: ConsultaIn):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    try:
        rows = load_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando datos: {e}")

    listado = "\n".join(
        f"{r['nombre']} | {r['nivel de dificultad']} | {r['subcategoria']} | {r['descripción']} | {r['enlace']} | {r['tutorial']}"
        for r in rows
    )

    system_prompt = (
        "Eres un experto en herramientas de inteligencia artificial. "
        "Responde únicamente con una TABLA HTML que contenga: "
        "nombre, nivel de dificultad, subcategoria, descripción, enlace y tutorial. "
        "Las filas deben provenir del listado dado."
    )

    user_prompt = (
        f"Estas son las herramientas:\n{listado}\n\n"
        f"El usuario dice: \"{body.mensaje}\".\n\n"
        "Devuelve SOLO la tabla HTML (sin <html>, sin markdown)."
    )

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        html = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not html:
            raise RuntimeError("Respuesta vacía del modelo")
        return {"html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al consultar modelo: {e}")