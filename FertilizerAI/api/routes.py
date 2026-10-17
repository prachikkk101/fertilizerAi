from fastapi import APIRouter
from src.models.predict import predict_fertilizer

router = APIRouter()


@router.post("/predict")
def predict(data: dict):
    result = predict_fertilizer(data)
    return result
