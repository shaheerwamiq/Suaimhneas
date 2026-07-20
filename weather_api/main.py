from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(title="Weather API", version="1.0.0")


class WeatherRequest(BaseModel):
    city: str


class WeatherResponse(BaseModel):
    city: str
    temperature_c: float
    condition: str
    description: str


MOCK_WEATHER: Dict[str, Dict[str, object]] = {
    "london": {"temperature_c": 18.0, "condition": "cloudy", "description": "Light clouds and cool breeze."},
    "new york": {"temperature_c": 28.0, "condition": "sunny", "description": "Bright and warm."},
    "tokyo": {"temperature_c": 24.0, "condition": "rainy", "description": "Scattered showers."},
    "sydney": {"temperature_c": 22.0, "condition": "clear", "description": "Clear skies and mild weather."},
}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/weather/{city}", response_model=WeatherResponse)
def get_weather(city: str) -> WeatherResponse:
    normalized = city.strip().lower()
    data = MOCK_WEATHER.get(normalized)
    if not data:
        raise HTTPException(status_code=404, detail="City not found")
    return WeatherResponse(city=city.title(), **data)


@app.post("/weather", response_model=WeatherResponse)
def post_weather(payload: WeatherRequest) -> WeatherResponse:
    normalized = payload.city.strip().lower()
    data = MOCK_WEATHER.get(normalized)
    if not data:
        raise HTTPException(status_code=404, detail="City not found")
    return WeatherResponse(city=payload.city.title(), **data)
