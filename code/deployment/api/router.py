import base64
import io
import typing as tp
from code.deployment.api import model as api_model_module

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

router = APIRouter()
model = api_model_module.load_model()


class ImageRequest(BaseModel):
    image: str


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: tp.Dict[str, float]


def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")


@router.get("/")
async def root() -> tp.Dict[str, str]:
    return {"service": "cats-dogs-classifier", "version": "1.0.0"}


@router.get("/status")
async def get_status() -> tp.Dict[str, tp.Union[str, bool]]:
    if model is None:
        raise HTTPException(status_code=503, detail="Service unavailable")

    return {"service": "operational", "model_loaded": True}


@router.post("/predict", response_model=PredictionResponse)
async def classify_image(request: ImageRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    image: Image.Image = decode_base64_image(request.image)
    result: tp.Dict[str, tp.Any] = api_model_module.predict_image(image)

    return PredictionResponse(
        prediction=result["prediction"], probabilities=result["probabilities"]
    )
