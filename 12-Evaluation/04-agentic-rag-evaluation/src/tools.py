# The agent's two tools: a knowledge-base search and a live weather lookup.
#
# The search result tags every chunk with "[source: <file>]". The model reads it
# to ground its answer, and the evaluation reads it to know which files the
# agent actually retrieved.
#
# The weather comes from Open-Meteo (open-meteo.com): free for non-commercial use,
# no API key. Because it's live, a golden case can't store the expected temperature;
# Section 14.5 shows how that case describes the expected behavior instead.
# This module is imported, not run.

import requests
from langchain_core.tools import tool

from src.vectorstore import get_collection

SOURCE_TAG = "[source: "
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo reports conditions as WMO weather codes.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


# The search tool: the four chunks closest to the agent's query, each tagged with its source file.
@tool
def search_knowledge_base(query: str, n_results: int = 4) -> str:
    """Search the internal knowledge base (CRM exports, projects, revenue, proposal
    documents). Use it for any question about company data: accounts, opportunities,
    sales reps, projects, revenue or proposals. Write the query in words: to look up
    an ID, say what it is, such as "sales rep REP01", never the bare ID."""
    collection = get_collection()
    if collection.count() == 0:
        return "The knowledge base is empty."
    results = collection.query(query_texts=[query], n_results=n_results)
    documents, metadatas = results["documents"][0], results["metadatas"][0]
    if not documents:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(f"{SOURCE_TAG}{meta['source']}]\n{text}"
                              for text, meta in zip(documents, metadatas))


# The weather tool: live conditions from Open-Meteo, no API key needed.
@tool
def get_weather(location: str) -> str:
    """Get the current weather (temperature, conditions and wind) for a city."""
    city = location.split(",")[0].strip()  # "London, UK" -> "London": the geocoder matches names only
    try:
        places = requests.get(GEOCODE_URL, params={"name": city, "count": 1, "language": "en"}, timeout=10)
        places.raise_for_status()
        matches = places.json().get("results")
        if not matches:
            return f"No place called '{location}' was found."
        place = matches[0]
        forecast = requests.get(FORECAST_URL, timeout=10, params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m", "timezone": "auto",
        })
        forecast.raise_for_status()
        now = forecast.json()["current"]
    except requests.RequestException as exc:
        # Returned, not raised: the model can tell the user, and the trace records why.
        return f"The weather service is unavailable right now ({type(exc).__name__})."
    conditions = WEATHER_CODES.get(now["weather_code"], "unknown conditions")
    return (f"{place['name']}, {place.get('country', '')}: {now['temperature_2m']:.0f}°C, {conditions}, "
            f"wind {now['wind_speed_10m']:.0f} km/h (Open-Meteo, local time {now['time']}).")


ALL_TOOLS = [search_knowledge_base, get_weather]
