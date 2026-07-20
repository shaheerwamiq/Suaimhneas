"""MCP-style tools exposing the weather API methods for the agent."""

import json
from typing import Any, Dict

import requests
from langchain_core.tools import tool

WEATHER_API_BASE_URL = "http://127.0.0.1:8000"


def _call_weather_api(path: str, payload: Dict[str, Any] | None = None) -> str:
    try:
        if payload is None:
            response = requests.get(f"{WEATHER_API_BASE_URL}{path}", timeout=5)
        else:
            response = requests.post(f"{WEATHER_API_BASE_URL}{path}", json=payload, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        return f"Weather API unavailable: {exc}"


@tool
def get_weather(city: str) -> str:
    """Get current weather details for a city."""
    return _call_weather_api(f"/weather/{city}")


@tool
def get_weather_by_post(city: str) -> str:
    """Get weather details for a city through the POST endpoint."""
    return _call_weather_api("/weather", {"city": city})


@tool
def health_check() -> str:
    """Check whether the weather API is available."""
    return _call_weather_api("/health")


WEATHER_TOOLS = [get_weather, get_weather_by_post, health_check]


if __name__ == "__main__":
    print(json.dumps({"health": health_check.invoke("")}, indent=2))
