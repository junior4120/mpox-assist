from ml.inference.base_engine import BaseEngine
from ml.inference.demo_engine import DemoEngine
from ml.inference.real_engine import RealEngine
from app.config import AI_MODE

_engine_instance: BaseEngine | None = None


def get_engine() -> BaseEngine:
    """Retourne une instance unique (singleton) du moteur IA actif."""
    global _engine_instance
    if _engine_instance is None:
        if AI_MODE == "real":
            _engine_instance = RealEngine()
        else:
            _engine_instance = DemoEngine()
        _engine_instance.load_model()
    return _engine_instance
