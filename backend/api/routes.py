import time
import uuid
import json
from pathlib import Path


from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import (
    MAX_IMAGE_SIZE_MB, ALLOWED_EXTENSIONS, UPLOAD_DIR,
    AI_MODE, MODEL_NAME, MODEL_VERSION, CLASSES,
    REAL_METRICS_PATH, REAL_MODEL_PATH,
)
from api.schemas import (
    PredictionResponse, AnalysisSummary, StatisticsResponse,
    ModelInfoResponse, SyncRequest, SyncResponse,
)
from api.auth import verify_api_key
from database.db import get_db
from database.models import Analysis, SyncEvent
from ml.preprocessing.image_preprocessing import (
    load_and_validate_image, preprocess_image, InvalidImageError,
)
from ml.inference.engine_factory import get_engine
from ml.explainability.gradcam_utils import save_gradcam_outputs
from services.triage_service import build_triage_message

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health_check():
    return {"status": "ok", "ai_mode": AI_MODE}


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(db: Session = Depends(get_db)):
    accuracy = recall = precision = f1_score = model_size_mb = avg_inference_time_ms = None

    if AI_MODE == "real":
        metrics_path = Path(REAL_METRICS_PATH)
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            accuracy = metrics.get("test_accuracy")
            recall = metrics.get("test_recall_macro")
            f1_score = metrics.get("test_f1_macro")
            precision = metrics.get("classification_report", {}).get("macro avg", {}).get("precision")

        model_file = Path(REAL_MODEL_PATH)
        if model_file.exists():
            model_size_mb = model_file.stat().st_size / (1024 * 1024)

        avg_time_db = db.query(func.avg(Analysis.inference_time_ms)).filter(Analysis.ai_mode == "real").scalar()
        avg_inference_time_ms = avg_time_db

    return ModelInfoResponse(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        classes=CLASSES,
        explainability="Grad-CAM" if AI_MODE == "real" else "Grad-CAM (simulation en mode démonstration)",
        mode=AI_MODE,
        accuracy=accuracy,
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        model_size_mb=model_size_mb,
        avg_inference_time_ms=avg_inference_time_ms,
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    patient_reference: str = Form(default=None),
    db: Session = Depends(get_db),
    _auth: bool = Depends(verify_api_key),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Extension non autorisée : {ext}. Autorisées : {ALLOWED_EXTENSIONS}")

    raw_bytes = await file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(400, f"Image trop volumineuse ({size_mb:.1f} Mo). Limite : {MAX_IMAGE_SIZE_MB} Mo.")

    try:
        image = load_and_validate_image(raw_bytes)
    except InvalidImageError as e:
        raise HTTPException(400, str(e))

    preprocessed = preprocess_image(image)

    engine = get_engine()

    start = time.perf_counter()
    predicted_class, probabilities = engine.predict(preprocessed)
    heatmap = engine.generate_gradcam(preprocessed, predicted_class)
    inference_time_ms = (time.perf_counter() - start) * 1000

    confidence = probabilities[predicted_class]
    triage_message = build_triage_message(predicted_class, confidence)

    analysis_id = str(uuid.uuid4())
    base_name = analysis_id

    original_path = UPLOAD_DIR / f"{base_name}_original.png"
    preprocessed.save(original_path)

    outputs = save_gradcam_outputs(preprocessed, heatmap, UPLOAD_DIR, base_name)

    analysis = Analysis(
        id=analysis_id,
        patient_reference=patient_reference or f"ANON-{analysis_id[:8]}",
        image_path=str(original_path),
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        gradcam_path=outputs["overlay_path"],
        triage_message=triage_message,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        ai_mode=AI_MODE,
        inference_time_ms=inference_time_ms,
        synced=True,
    )
    db.add(analysis)
    db.commit()

    return PredictionResponse(
        analysis_id=analysis_id,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        triage_message=triage_message,
        ai_mode=AI_MODE,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inference_time_ms=inference_time_ms,
        original_image_url=f"/uploads/{original_path.name}",
        heatmap_url=f"/uploads/{Path(outputs['heatmap_path']).name}",
        overlay_url=f"/uploads/{Path(outputs['overlay_path']).name}",
    )


@router.get("/results", response_model=list[AnalysisSummary])
def list_results(db: Session = Depends(get_db), limit: int = 50):
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
    return [
        AnalysisSummary(
            id=a.id,
            patient_reference=a.patient_reference,
            predicted_class=a.predicted_class,
            confidence=a.confidence,
            model_name=a.model_name,
            model_version=a.model_version,
            created_at=a.created_at.isoformat(),
            synced=a.synced,
        )
        for a in analyses
    ]


@router.get("/results/{analysis_id}")
def get_result(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(404, "Analyse introuvable.")
    return {
        "id": analysis.id,
        "patient_reference": analysis.patient_reference,
        "predicted_class": analysis.predicted_class,
        "confidence": analysis.confidence,
        "probabilities": analysis.probabilities,
        "triage_message": analysis.triage_message,
        "model_name": analysis.model_name,
        "model_version": analysis.model_version,
        "ai_mode": analysis.ai_mode,
        "created_at": analysis.created_at.isoformat(),
        "original_image_url": f"/uploads/{Path(analysis.image_path).name}",
        "overlay_url": f"/uploads/{Path(analysis.gradcam_path).name}" if analysis.gradcam_path else None,
    }


@router.post("/sync", response_model=SyncResponse)
def sync_analyses(payload: SyncRequest, db: Session = Depends(get_db), _auth: bool = Depends(verify_api_key)):
    synced, failed = 0, 0
    for analysis_id in payload.analysis_ids:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.synced = True
            db.add(SyncEvent(analysis_id=analysis_id, status="success"))
            synced += 1
        else:
            failed += 1
    db.commit()
    return SyncResponse(synced_count=synced, failed_count=failed)


@router.get("/statistics", response_model=StatisticsResponse)
def statistics(db: Session = Depends(get_db)):
    total = db.query(func.count(Analysis.id)).scalar() or 0
    suspects = db.query(func.count(Analysis.id)).filter(Analysis.predicted_class != "Peau saine").scalar() or 0

    distribution = {cls: 0 for cls in CLASSES}
    rows = db.query(Analysis.predicted_class, func.count(Analysis.id)).group_by(Analysis.predicted_class).all()
    for cls, count in rows:
        distribution[cls] = count

    avg_time = db.query(func.avg(Analysis.inference_time_ms)).scalar()

    return StatisticsResponse(
        total_analyses=total,
        suspect_cases=suspects,
        class_distribution=distribution,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        avg_inference_time_ms=avg_time,
        is_demo_data=(AI_MODE == "demo"),
    )
