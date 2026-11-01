# Unit tests for the chatbot's building blocks: chunking, loading and the weather tool.
#
# None of these call a model, Chroma or the network, so they run in a second with no keys.
# Run them with (from the repository root):
#    python -m pytest -q 9-Agentic-RAG/tests
# Expected result: all tests pass.

from src.chunking import fixed_size_chunk
from src.loaders import load_documents


def test_chunks_overlap_so_a_boundary_fact_survives():
    text = "abcdefghij" * 3                      # 30 characters
    chunks = fixed_size_chunk(text, chunk_size=10, chunk_overlap=4)
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ghijabcdef"            # starts 4 characters before the first one ended
    assert all(len(c) <= 10 for c in chunks)


def test_csv_rows_become_column_value_lines(tmp_path):
    (tmp_path / "crm").mkdir()
    (tmp_path / "crm" / "sales_reps.csv").write_text("name,region\nSarah Lee,Europe\n", encoding="utf-8")
    [doc] = load_documents(tmp_path)
    assert doc.source == "crm/sales_reps.csv"
    assert doc.text == "name: Sarah Lee, region: Europe"


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_get_weather_formats_the_live_reading(monkeypatch):
    import src.tools as tools
    replies = {
        tools.GEOCODE_URL: {"results": [{"name": "London", "country": "United Kingdom",
                                         "latitude": 51.5, "longitude": -0.13}]},
        tools.FORECAST_URL: {"current": {"time": "2026-11-01T16:30", "temperature_2m": 11.0,
                                         "weather_code": 3, "wind_speed_10m": 10.1}},
    }
    monkeypatch.setattr(tools.requests, "get", lambda url, **kwargs: _FakeResponse(replies[url]))
    assert tools.get_weather.invoke({"location": "London, UK"}) == \
        "London, United Kingdom: 11°C, overcast, wind 10 km/h (Open-Meteo, local time 2026-11-01T16:30)."


def test_get_weather_reports_unknown_places_and_outages(monkeypatch):
    import src.tools as tools
    monkeypatch.setattr(tools.requests, "get", lambda url, **kwargs: _FakeResponse({}))
    assert tools.get_weather.invoke({"location": "Atlantisville"}) == "No place called 'Atlantisville' was found."

    def offline(url, **kwargs):
        raise tools.requests.ConnectionError("no network")
    monkeypatch.setattr(tools.requests, "get", offline)
    assert tools.get_weather.invoke({"location": "London"}) == \
        "The weather service is unavailable right now (ConnectionError)."
