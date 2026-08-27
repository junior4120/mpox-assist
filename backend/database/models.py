"""
Modèles SQLAlchemy — MPOX-Assist.

NOTE ÉTHIQUE : aucune donnée personnelle réelle de patient n'est stockée.
`patient_reference` est un identifiant anonymisé généré côté client
(ou côté serveur en mode démo), jamais un nom, prénom ou numéro
d'identité réel.
"""
import uuid
import datetime as dt

from sqlalchemy import (
    Column, String, Float, DateTime, ForeignKey, Boolean, Integer, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def gen_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    full_name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="health_agent")  # health_agent | doctor | admin
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    analyses = relationship("Analysis", back_populates="user")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=gen_uuid)
    patient_reference = Column(String, nullable=False)  # identifiant anonymisé, PAS de nom réel
    user_id = Column(String, ForeignKey("users.id"), nullable=True)

    image_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)  # {"Mpox": 0.8, ...}

    gradcam_path = Column(String, nullable=True)
    triage_message = Column(String, nullable=False)

    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    ai_mode = Column(String, nullable=False)  # "demo" | "real"

    inference_time_ms = Column(Float, nullable=True)

    synced = Column(Boolean, default=False)
    created_offline = Column(Boolean, default=False)

    user = relationship("User", back_populates="analyses")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    mode = Column(String, nullable=False)  # "demo" | "real"
    classes = Column(JSON, nullable=False)

    # Ces champs restent NULL tant qu'aucune mesure réelle n'a été faite.
    accuracy = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    model_size_mb = Column(Float, nullable=True)
    avg_inference_time_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, default=dt.datetime.utcnow)


class SyncEvent(Base):
    __tablename__ = "sync_events"

    id = Column(String, primary_key=True, default=gen_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    synced_at = Column(DateTime, default=dt.datetime.utcnow)
    status = Column(String, default="success")  # success | failed
