# Shared tools for 11_parallel_subgraphs.py (book Chapter 8, section 8.3).
#
# Two small tool sets for the "should I take this trip?" example:
#   WEATHER_TOOLS - get_weekend_forecast: a live call to the free Open-Meteo API
#   BUDGET_TOOLS  - get_train_price, get_hotel_price, add_costs: lookups in
#                   trip_data.json (sample prices you can edit) plus a calculator
#
# They live in their own module so the graph file stays about the graph.
# Nothing to run here directly; 11_parallel_subgraphs.py imports the two lists.
#
# Settings:
#   TRIP_SAMPLE_WEATHER=1   skip the Open-Meteo call and return a fixed sample
#                           forecast (for a locked-down network or a classroom)

import json
import os
from datetime import date, timedelta
from pathlib import Path

import requests
from langchain_core.tools import tool

TRIP_DATA = json.loads((Path(__file__).resolve().parent / "trip_data.json").read_text())
CITIES = TRIP_DATA["cities"]
ORIGIN = TRIP_DATA["origin"]
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _find_city(city: str) -> dict | None:
    return CITIES.get(city.strip().lower())


def _unknown_city(city: str) -> str:
    # Chapter 6's rule: a failed lookup comes back as text the model can act on.
    return f"No data for {city!r}. Known cities: {', '.join(sorted(CITIES))}."


def _coming_weekend(today: date | None = None) -> tuple[date, date]:
    """Return the next Saturday and Sunday. On a Saturday that is today and
    tomorrow; on a Sunday it is the following weekend."""
    today = today or date.today()
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    return saturday, saturday + timedelta(days=1)


# ---------------------------------------------------------
# Weather agent's tool
#
# The model supplies only the city. The CODE works out which dates "this
# weekend" means, because a model does not know today's date.
# ---------------------------------------------------------
@tool
def get_weekend_forecast(city: str) -> str:
    """Get the forecast for the coming Saturday and Sunday in a city.

    Returns one line per day: high and low temperature in Celsius and the
    chance of rain. Use this for any question about weather on a weekend trip.
    """
    place = _find_city(city)
    if place is None:
        return _unknown_city(city)
    saturday, sunday = _coming_weekend()

    if os.getenv("TRIP_SAMPLE_WEATHER") == "1":
        return (f"[sample data, not a live forecast] {city.title()}\n"
                f"Sat {saturday}: high 14 C, low 6 C, rain chance 20%\n"
                f"Sun {sunday}: high 11 C, low 4 C, rain chance 65%")

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "start_date": saturday.isoformat(),
                "end_date": sunday.isoformat(),
                "timezone": "auto",      # Open-Meteo requires a timezone for daily values
            },
            timeout=10,
        )
        response.raise_for_status()
        daily = response.json()["daily"]
    except (requests.RequestException, KeyError, ValueError) as error:
        return f"Weather service failed: {error}"

    lines = [city.title()]
    for day, high, low, rain in zip(daily["time"], daily["temperature_2m_max"],
                                    daily["temperature_2m_min"],
                                    daily["precipitation_probability_max"]):
        weekday = date.fromisoformat(day).strftime("%a")
        lines.append(f"{weekday} {day}: high {high} C, low {low} C, rain chance {rain}%")
    return "\n".join(lines)


# ---------------------------------------------------------
# Budget agent's tools
#
# Prices come from trip_data.json: real lookups, sample numbers.
# ---------------------------------------------------------
@tool
def get_train_price(city: str) -> str:
    """Get the return train fare from the user's home city to a destination city."""
    place = _find_city(city)
    if place is None:
        return _unknown_city(city)
    return f"Return train {ORIGIN} - {city.title()}: {place['train_return']} CAD"


@tool
def get_hotel_price(city: str) -> str:
    """Get the price of one hotel night in a destination city."""
    place = _find_city(city)
    if place is None:
        return _unknown_city(city)
    return f"Hotel in {city.title()}: {place['hotel_per_night']} CAD per night"


@tool
def add_costs(amounts: list[float]) -> float:
    """Add a list of costs and return the total. Use this instead of mental arithmetic."""
    return sum(amounts)


WEATHER_TOOLS = [get_weekend_forecast]
BUDGET_TOOLS = [get_train_price, get_hotel_price, add_costs]
