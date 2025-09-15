import os
import time
import csv
from io import StringIO
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_URL = os.getenv("DATA_URL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b:free")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Recomendador Backend Ligero (CSV-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {"ts": 0, "rows": []}

REQUIRED_COLS = ["nombre", "nivel de dificultad", "subcategoria", "descripción", "enlace", "tutorial"]

def _normalize_row(row: dict) -> dict:
    mapping = {
        "nombre": ["nombre", "Nombre", "NOMBRE", "Nombre de la herramienta"],
        "nivel de dificultad": ["nivel de dificultad", "nivel", "Nivel", "Nivel de dificultad"],
        "subcategoria": ["subcategoria", "Subcategoria", "SUBCATEGORIA", "Subcategorías"],
        "descripción": ["descripción", "descripcion", "Descripcion", "Descripción"],
        "enlace": ["enlace", "link", "URL", "Link"],
        "tutorial": ["tutorial", "Tutorial"],
    }
    norm = {}
    for target, aliases in mapping.items():
        value = ""
        for key in aliases:
            if key in row and row[key]:
                value = str(row[key])
                break
        norm[target] = value
    return norm

def load_data(force: bool = False):
    now = time.time()
    if not force and _cache["rows"] and now - _cache["ts"] < CACHE_TTL_SECONDS:
        return _cache["rows"]

    if not DATA_URL:
        raise RuntimeError("No se configuró DATA_URL")

    resp = requests.get(DATA_URL, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"No se pudo descargar datos ({resp.status_code})")

    rows = []
    if DATA_URL.lower().endswith(".json"):
        data = resp.json()
        if isinstance(data, dict) and "herramientas" in data:
            data = data["herramientas"]
        if not isinstance(data, list):
            raise RuntimeError("JSON inválido: se esperaba lista o clave 'herramientas'")
        for r in data:
            rows.append(_normalize_row(r))
    else:
        reader = csv.DictReader(StringIO(resp.text))
        for r in reader:
            rows.append(_normalize_row(r))

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
