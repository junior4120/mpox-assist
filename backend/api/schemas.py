from typing import Dict, List, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    analysis_id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    triage_message: str
    ai_mode: str
    model_name: str
    model_version: str
    inference_time_ms: float
    original_image_url: str
    heatmap_url: str
    overlay_url: str
    warning: str = (
        "Grad-CAM constitue une méthode d'explicabilité et ne constitue pas une preuve médicale."
    )


class AnalysisSummary(BaseModel):
    id: str
    patient_reference: str
    predicted_class: str
    confidence: float
    model_name: str
    model_version: str
    created_at: str
    synced: bool

    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    total_analyses: int
    suspect_cases: int
    class_distribution: Dict[str, int]
    model_name: str
    model_version: str
    avg_inference_time_ms: Optional[float]
    is_demo_data: bool


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    classes: List[str]
    explainability: str
    mode: str
    accuracy: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1_score: Optional[float] = None
    model_size_mb: Optional[float] = None
    avg_inference_time_ms: Optional[float] = None


class SyncRequest(BaseModel):
    analysis_ids: List[str]


class SyncResponse(BaseModel):
    synced_count: int
    failed_count: int
