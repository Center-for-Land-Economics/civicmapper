from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CityParquet:
    city: str
    state: str
    legacy_filename: str
    # Non-US cities carry a country slug so the filename convention is
    # <city>-<state/province>-<country>-parcels.parquet (e.g. tallinn-harju-ee).
    # US cities are grandfathered: leave country=None → the historic
    # <city>-<state> two-part slug is preserved unchanged.
    country: str | None = None

    @property
    def slug(self) -> str:
        parts = [self.city, self.state] + ([self.country] if self.country else [])
        return "-".join(parts)

    @property
    def canonical_filename(self) -> str:
        return f"{self.slug}-parcels.parquet"

    @property
    def parking_filename(self) -> str:
        """Canonical filename for the isolated parking lots GeoParquet."""
        return f"{self.slug}-parking-lots.parquet"

    @property
    def parking_metadata_filename(self) -> str:
        """Canonical filename for the parking lots metadata JSON."""
        return f"{self.slug}-parking-lots-metadata.json"

    @property
    def land_totals_filename(self) -> str:
        """Canonical filename for the per-city land-totals JSON (parking-share denominators)."""
        return f"{self.city}-{self.state}-land-totals.json"


CITY_PARQUETS: dict[str, CityParquet] = {
    "cincinnati": CityParquet(
        city="cincinnati", state="oh", legacy_filename="cincinnati-oh-parcels.parquet"
    ),
    "cleveland": CityParquet(
        city="cleveland", state="oh", legacy_filename="cleveland-oh-parcels.parquet"
    ),
    "columbus": CityParquet(
        city="columbus", state="oh", legacy_filename="columbus-oh-parcels.parquet"
    ),
    "charlottesville": CityParquet(
        city="charlottesville",
        state="va",
        legacy_filename="charlottesville-va-parcels.parquet",
    ),
    "denver": CityParquet(city="denver", state="co", legacy_filename="denver-co-parcels.parquet"),
    "fortcollins": CityParquet(
        city="fortcollins", state="co", legacy_filename="fortcollins-co-parcels.parquet"
    ),
    "pueblo": CityParquet(city="pueblo", state="co", legacy_filename="pueblo-co-parcels.parquet"),
    "southbend": CityParquet(city="southbend", state="in", legacy_filename="southbend.parquet"),
    "syracuse": CityParquet(
        city="syracuse",
        state="ny",
        legacy_filename="syracuse_parcels_refined_20251001.parquet",
    ),
    "spokane": CityParquet(city="spokane", state="wa", legacy_filename="spokane.parquet"),
    "rochester": CityParquet(city="rochester", state="ny", legacy_filename="rochester.parquet"),
    "bellingham": CityParquet(city="bellingham", state="wa", legacy_filename="bellingham.parquet"),
    "morgantown": CityParquet(city="morgantown", state="wv", legacy_filename="morgantown.parquet"),
    "ibx": CityParquet(city="ibx", state="ny", legacy_filename="nyc-ibx-parcels.parquet"),
    "nyc": CityParquet(city="nyc", state="ny", legacy_filename="nyc-ny-parcels.parquet"),
    "stpaul": CityParquet(city="st-paul", state="mn", legacy_filename="st-paul-mn-parcels.parquet"),
    "baltimore": CityParquet(city="baltimore", state="md", legacy_filename="baltimore-md-parcels.parquet"),
    "albuquerque": CityParquet(
        city="albuquerque", state="nm", legacy_filename="albuquerque-nm-parcels.parquet"
    ),
    "portland": CityParquet(city="portland", state="or", legacy_filename="portland-or-parcels.parquet"),
    "houston": CityParquet(city="houston", state="tx", legacy_filename="houston-tx-parcels.parquet"),
    "austin": CityParquet(city="austin", state="tx", legacy_filename="austin-tx-parcels.parquet"),
    "dallas": CityParquet(city="dallas", state="tx", legacy_filename="dallas-tx-parcels.parquet"),
    "sanantonio": CityParquet(city="sanantonio", state="tx", legacy_filename="sanantonio-tx-parcels.parquet"),
    "bcs": CityParquet(city="bcs", state="tx", legacy_filename="bcs-tx-parcels.parquet"),
    "rockville": CityParquet(
        city="rockville", state="md", legacy_filename="rockville-md-parcels.parquet"
    ),
    "detroit": CityParquet(city="detroit", state="mi", legacy_filename="detroit-mi-parcels.parquet"),
    "chicago": CityParquet(city="chicago", state="il", legacy_filename="chicago-il-parcels.parquet"),
    "tulsa": CityParquet(city="tulsa", state="ok", legacy_filename="tulsa-ok-parcels.parquet"),
    "newportnews": CityParquet(
        city="newportnews", state="va", legacy_filename="newportnews-va-parcels.parquet"
    ),
    "olympia": CityParquet(
        city="olympia", state="wa", legacy_filename="olympia-wa-parcels.parquet"
    ),
    "seattle": CityParquet(
        city="seattle", state="wa", legacy_filename="seattle-wa-parcels.parquet"
    ),
    "vancouver": CityParquet(
        city="vancouver", state="wa", legacy_filename="vancouver-wa-parcels.parquet"
    ),
    "vancouverbc": CityParquet(
        city="vancouver", state="bc", country="ca",
        legacy_filename="vancouver-bc-ca-parcels.parquet",
    ),
    "dmv": CityParquet(city="dmv", state="dc", legacy_filename="dmv-dc-parcels.parquet"),
    "washington": CityParquet(
        city="washington", state="dc", legacy_filename="washington-dc-parcels.parquet"
    ),
    "hartfordmetro": CityParquet(
        city="hartfordmetro", state="ct", legacy_filename="hartfordmetro-ct-parcels.parquet"
    ),
    # First non-US city. New <city>-<province>-<country> convention:
    # province = Harju maakond, country = ee (Estonia).
    "tallinn": CityParquet(
        city="tallinn", state="harju", country="ee",
        legacy_filename="tallinn-harju-ee-parcels.parquet",
    ),
    # Second non-US city. state = Region Hovedstaden, country = dk (Denmark).
    # Geometry via DAWA (open); land value (grundværdi) pending Datafordeler VUR access.
    "copenhagen": CityParquet(
        city="copenhagen", state="hovedstaden", country="dk",
        legacy_filename="copenhagen-hovedstaden-dk-parcels.parquet",
    ),
}


def _load_extra_cities() -> None:
    """Overlay extension point: merge extra cities from CIVICMAPPER_EXTRA_CITIES.

    Point the env var at a directory of frontend-style city registry JSONs
    (<key>.json with at least "state" and "filename" — the same files that live in
    viz/src/cities/). Lets a private overlay repo register cities without editing
    this file. Entries here never override the built-in registry.
    """
    import json
    import os
    from pathlib import Path

    extra_dir = os.environ.get("CIVICMAPPER_EXTRA_CITIES", "").strip()
    if not extra_dir:
        return
    for p in sorted(Path(extra_dir).glob("*.json")):
        key = p.stem.lower()
        if key in CITY_PARQUETS:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        filename = d.get("filename", "")
        # Everything derives from the filename slug: <city>-<state>-parcels.parquet (US)
        # or <city>-<province>-<country>-parcels.parquet (non-US). NOTE the frontend
        # JSON's "state" field is the COUNTRY code for non-US cities — don't use it here.
        if not (filename.endswith("-parcels.parquet") and filename.startswith(f"{key}-")):
            print(f"[parquet_registry] skipping extra city '{key}': filename '{filename}' "
                  f"doesn't match the <key>-...-parcels.parquet convention")
            continue
        rest = filename[len(key) + 1: -len("-parcels.parquet")].split("-")
        if len(rest) not in (1, 2):
            print(f"[parquet_registry] skipping extra city '{key}': cannot parse slug '{filename}'")
            continue
        CITY_PARQUETS[key] = CityParquet(
            city=key, state=rest[0], legacy_filename=filename,
            country=rest[1] if len(rest) == 2 else None,
        )


_load_extra_cities()


def resolve_city(city: str) -> CityParquet:
    key = city.strip().lower()
    if key not in CITY_PARQUETS:
        available = ", ".join(sorted(CITY_PARQUETS))
        raise ValueError(f"Unknown city '{city}'. Available: {available}")
    return CITY_PARQUETS[key]


def list_cities() -> list[str]:
    return sorted(CITY_PARQUETS.keys())
