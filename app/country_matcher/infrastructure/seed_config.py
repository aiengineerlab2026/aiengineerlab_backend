"""Which countries and places the dataset is built from. Edit this, or pass ``--countries`` to the ingest command.

Each entry is a Wikimedia Commons category of street scenes in one city: the city becomes the image's ``city`` label, and
spreading a country over several cities keeps the scenes diverse. Run ``python -m app.country_matcher.cli check-sources``
to verify the categories still exist and have files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountrySeed:
    code: str  # ISO 3166-1 alpha-2
    name: str
    places: tuple[tuple[str, str], ...]  # (Commons category, city)


def _streets(*cities: str) -> tuple[tuple[str, str], ...]:
    return tuple((f"Streets in {c}", c) for c in cities)


DEFAULT_COUNTRIES: tuple[CountrySeed, ...] = (
    CountrySeed("JP", "Japan", _streets("Tokyo", "Kyoto", "Osaka", "Sapporo", "Hiroshima")),
    CountrySeed("DE", "Germany", _streets("Hamburg", "Cologne", "Munich", "Essen", "Dortmund", "Heidelberg", "Stuttgart", "Frankfurt am Main", "Berlin")),
    CountrySeed("FR", "France", _streets("Paris", "Marseille", "Strasbourg", "Bordeaux", "Lille", "Nice")),
    CountrySeed("IT", "Italy", _streets("Rome", "Milan", "Florence", "Naples", "Venice")),
    CountrySeed("GB", "United Kingdom", _streets("London", "Edinburgh", "Manchester", "Liverpool", "Bristol")),
    CountrySeed("BR", "Brazil", _streets("São Paulo", "Rio de Janeiro", "Belo Horizonte", "Porto Alegre", "Curitiba", "Brasília")),
    CountrySeed("IN", "India", _streets("Mumbai", "Delhi", "Kolkata", "Jaipur", "Chennai")),
    CountrySeed("EG", "Egypt", _streets("Cairo", "Alexandria", "Luxor", "Aswan")),
    CountrySeed("MX", "Mexico", _streets("Mexico City", "Oaxaca", "Puebla", "Mérida, Yucatán", "Guanajuato", "Monterrey")),
    CountrySeed("US", "United States", _streets("New York City", "San Francisco", "Chicago", "Los Angeles", "Boston")),
    CountrySeed("ES", "Spain", _streets("Madrid", "Barcelona", "Seville", "Valencia", "Granada", "Bilbao")),
    CountrySeed("SA", "Saudi Arabia", _streets("Riyadh", "Jeddah", "Mecca", "Medina", "Dammam")),
    CountrySeed("CA", "Canada", _streets("Toronto", "Vancouver", "Montreal", "Ottawa", "Calgary")),
    CountrySeed("LB", "Lebanon", _streets("Beirut", "Tripoli", "Sidon", "Byblos")),
)
