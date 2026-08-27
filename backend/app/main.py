from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import UPLOAD_DIR, AI_MODE
from database.db import init_db
from api.routes import router as api_router

app = FastAPI(
    title="MPOX-Assist API",
    description=(
        "API du prototype de recherche MPOX-Assist. "
        "Ceci n'est PAS un dispositif médical validé. "
        "Aucune sortie de cette API ne constitue un diagnostic confirmé."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC uniquement — restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.include_router(api_router)


@app.on_event("startup")
def on_startup():
    init_db()
    print(f"[MPOX-Assist] Démarrage en mode AI_MODE={AI_MODE}")


@app.get("/")
def root():
    return {
        "project": "MPOX-Assist",
        "status": "prototype de recherche — non validé cliniquement",
        "ai_mode": AI_MODE,
        "docs": "/docs",
    }
