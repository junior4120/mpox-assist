from fastapi import Header, HTTPException, status
from app.config import API_KEY


async def verify_api_key(x_api_key: str = Header(...)):
    """
    Authentification simple par clé API (header X-API-Key).
    Suffisant pour un POC ; à remplacer par OAuth2/JWT en production.
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide.",
        )
    return True
