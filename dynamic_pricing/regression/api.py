# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from utils import suggest_discount, MODEL_PATH

app = FastAPI()

class Request(BaseModel):
    hour: int
    rating: float
    price: float

class Response(BaseModel):
    discount: float
    discount_pct: float
    expected_obj: float

@app.post("/recommend", response_model=Response)
def recommend(req: Request):
    d, obj = suggest_discount(req.hour, req.rating, req.price)
    return Response(
        discount=d,
        discount_pct=d*100,
        expected_obj=obj
    )
