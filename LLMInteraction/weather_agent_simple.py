"""Very small weather agent that consumes the weather MCP-style tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLMInteraction.weather_mcp_tools import get_weather, get_weather_by_post, health_check


def run_agent(city: str | None = None) -> str:
    if not city:
        city = input("Enter a city to check the weather: ").strip()

    print("[agent] Checking health...")
    print(health_check.invoke(""))

    print(f"[agent] Calling get_weather for {city!r}...")
    result = get_weather.invoke(city)
    print(result)

    print(f"[agent] Calling get_weather_by_post for {city!r}...")
    post_result = get_weather_by_post.invoke(city)
    print(post_result)
    return result


if __name__ == "__main__":
    run_agent()
