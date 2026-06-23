"""
FlowMind AI — Live Data Router

APIs used:
  - TomTom Traffic Incidents  (/traffic/services/5/incidentDetails) — your key works
  - TomTom Flow Segment Data  (/traffic/services/4/flowSegmentData) — free, same key,
    gives current speed + free-flow speed per road segment, no Routing API needed
  - OpenStreetMap Overpass    — free, no key, real venue data

TomTom Routing API is NOT used — it requires a separate paid product on your key (403).
Instead we use the Flow API which IS included in the free Traffic tier.
"""

import httpx, os, asyncio, math, logging
from fastapi import APIRouter
from datetime import datetime, timezone, date
from ml.engine import (
    predict_corridor_congestion_now as ml_predict_corridor_now,
    sample_live_incidents as ml_sample_live_incidents,
    estimate_venue_crowd as ml_estimate_venue_crowd,
)

router = APIRouter()
logger = logging.getLogger("flowmind.livedata")

TOMTOM_KEY = os.getenv("TOMTOM_API_KEY", "")

BLR_LAT, BLR_LNG = 12.9716, 77.5946
BLR_BBOX = "77.3791,12.7343,77.8388,13.1435"  # minLon,minLat,maxLon,maxLat (TomTom bbox order)

# ── Module-level caches ───────────────────────────────────────────────────────
# Shared between requests so the 10-min TTL actually works across calls.
# (Local variables inside the function reset every call — that's why the cache
#  wasn't working before.)
_corridor_cache: list = []
_corridor_cache_time: datetime | None = None
CORRIDOR_CACHE_TTL = 600   # 10 minutes → 7 × 144 = 1,008 flow calls/day

_ml_traffic_cache: list = []
_ml_traffic_cache_time: datetime | None = None
ML_CACHE_TTL = 600

# ── TomTom quota backoff ──────────────────────────────────────────────────────
_flow_quota_exhausted_date: date | None = None

def _flow_quota_exhausted() -> bool:
    return _flow_quota_exhausted_date == datetime.now(timezone.utc).date()

def _mark_flow_quota_exhausted() -> None:
    global _flow_quota_exhausted_date
    today = datetime.now(timezone.utc).date()
    if _flow_quota_exhausted_date != today:
        logger.warning(
            "TomTom Flow API daily quota exhausted — ML fallback until midnight UTC (%s).", today
        )
    _flow_quota_exhausted_date = today

def np_clip(v, lo, hi):
    return max(lo, min(hi, v))

# ── Overpass — real OSM venue data ───────────────────────────────────────────
# The public Overpass instances are notoriously flaky under load (504s, 429s)
# and overpass-api.de's Apache front-end specifically returns 406 Not
# Acceptable if it sees an Accept/Accept-Encoding header it doesn't like —
# httpx adds both of those by default even when you don't set them yourself,
# so we explicitly null them out below. We also try several mirrors in order
# before giving up, since any single free instance can be down at any time.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
OVERPASS_QUERY = """
[out:json][timeout:25];
(
  node["amenity"="place_of_worship"](12.85,77.45,13.10,77.75);
  node["leisure"="stadium"](12.85,77.45,13.10,77.75);
  node["amenity"="university"](12.85,77.45,13.10,77.75);
  node["amenity"="hospital"](12.85,77.45,13.10,77.75);
  node["tourism"="attraction"](12.85,77.45,13.10,77.75);
  node["shop"="mall"](12.85,77.45,13.10,77.75);
  node["amenity"="theatre"](12.85,77.45,13.10,77.75);
  node["amenity"="cinema"](12.85,77.45,13.10,77.75);
);
out body 80;
"""

async def fetch_osm_venues() -> list:
    last_error = None
    for url in OVERPASS_URLS:
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                client.headers["User-Agent"] = "FlowMindAI/1.0 (Bengaluru traffic; flowmind-ai@example.com)"
                # Actually remove (not just blank-out — httpx rejects None as a
                # header value) the Accept/Accept-Encoding headers httpx adds
                # by default. Overpass's Apache mod_negotiation 406s on some
                # combinations of these even though we never asked for them.
                for h in ("Accept", "Accept-Encoding"):
                    client.headers.pop(h, None)
                resp = await client.post(url, data={"data": OVERPASS_QUERY})
            if resp.status_code != 200:
                logger.warning("Overpass (%s) returned HTTP %s: %s", url, resp.status_code, resp.text[:200])
                last_error = f"HTTP {resp.status_code}"
                continue  # try the next mirror instead of giving up immediately
            elements = resp.json().get("elements", [])
            venues = []
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name") or tags.get("name:en", "")
                if not name:
                    continue
                amenity = (tags.get("amenity") or tags.get("leisure")
                           or tags.get("tourism") or tags.get("shop", ""))
                TYPE_MAP = {
                    "place_of_worship": "Religious Event", "stadium": "Sports Event",
                    "university": "Academic Event",        "hospital": "Emergency Zone",
                    "attraction": "Tourist Event",          "mall": "Public Gathering",
                    "theatre": "Cultural Event",            "cinema": "Public Event",
                }
                cap_ranges = {
                    "stadium": (5000, 50000), "mall": (2000, 20000),
                    "place_of_worship": (500, 10000), "university": (1000, 8000),
                    "theatre": (200, 2000), "cinema": (100, 1500),
                    "hospital": (200, 1000), "attraction": (500, 5000),
                }.get(amenity, (100, 2000))
                crowd_est = ml_estimate_venue_crowd(*cap_ranges)
                risk = "High" if crowd_est > 10000 else "Moderate" if crowd_est > 3000 else "Low"
                venues.append({
                    "id": f"osm-{el['id']}",
                    "name": name,
                    "event_type": TYPE_MAP.get(amenity, "Public Event"),
                    "amenity": amenity,
                    "latitude": el.get("lat", BLR_LAT),
                    "longitude": el.get("lon", BLR_LNG),
                    "crowd_estimate": crowd_est,
                    "risk_level": risk,
                    "source": "OpenStreetMap (Live)",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            logger.info("Overpass (%s) returned %d venues.", url, len(venues))
            return venues[:60]
        except Exception as e:
            # str(e) is often empty for httpx connection-level errors (e.g.
            # ConnectError, ConnectTimeout) — log the exception type too so
            # the cause is actually diagnosable from the server logs.
            logger.warning("Overpass request to %s failed: %s: %r", url, type(e).__name__, e)
            last_error = f"{type(e).__name__}: {e}"
            continue  # try the next mirror

    logger.warning("All Overpass endpoints failed (%s) — using ML fallback.", last_error)
    return _fallback_events()

# ── TomTom Traffic Incidents ──────────────────────────────────────────────────
async def fetch_tomtom_incidents() -> list:
    if not TOMTOM_KEY:
        return _fallback_incidents()
    fields = (
        "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,"
        "magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,"
        "from,to,length,delay,roadNumbers,timeValidity}}}"
    )
    params = {
        "key": TOMTOM_KEY, "bbox": BLR_BBOX, "fields": fields,
        "language": "en-GB",
        "categoryFilter": "0,1,2,3,4,5,6,7,8,9,10,11,14",
        "timeValidityFilter": "present",
    }
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                "https://api.tomtom.com/traffic/services/5/incidentDetails",
                params=params,
            )
        if resp.status_code != 200:
            logger.warning("TomTom Incidents HTTP %s: %s", resp.status_code, resp.text[:200])
            return _fallback_incidents()
        incidents = []
        for inc in resp.json().get("incidents", [])[:50]:
            props = inc.get("properties", {})
            geo   = inc.get("geometry", {})
            coords = geo.get("coordinates", [])
            if geo.get("type") == "Point" and coords:
                lng, lat = coords[0], coords[1]
            elif geo.get("type") == "LineString" and coords:
                mid = coords[len(coords) // 2]
                lng, lat = mid[0], mid[1]
            else:
                continue
            magnitude = props.get("magnitudeOfDelay", 0)
            events_list = props.get("events", [])
            desc = events_list[0].get("description", "Traffic incident") if events_list else "Traffic incident"
            icon = props.get("iconCategory", 0)
            cause_map = {
                0:"Unknown", 1:"Accident", 2:"Fog", 3:"Dangerous Conditions", 4:"Rain",
                5:"Ice", 6:"Traffic Jam", 7:"Lane Closed", 8:"Road Closed", 9:"Road Works",
                10:"Wind", 11:"Flooding", 12:"Detour", 13:"Cluster", 14:"Broken Down Vehicle",
            }
            incidents.append({
                "id": props.get("id", f"tt-{abs(hash((lat, lng, desc))) % 100000}"),
                "description": desc,
                "cause": cause_map.get(icon, "Traffic Incident"),
                "severity": {0:"Unknown",1:"Minor",2:"Moderate",3:"Major",4:"Undefined"}.get(magnitude, "Unknown"),
                "magnitude": magnitude,
                "latitude": lat, "longitude": lng,
                "from": props.get("from", ""), "to": props.get("to", ""),
                "delay_sec": props.get("delay", 0),
                "length_m": props.get("length", 0),
                "road": (props.get("roadNumbers") or ["Unknown"])[0],
                "source": "TomTom Live",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        if not incidents:
            logger.info("TomTom Incidents: 0 active incidents in Bengaluru right now.")
            return []
        return incidents
    except Exception as e:
        logger.warning("TomTom Incidents request failed: %s", e)
        return _fallback_incidents()

# ── TomTom Flow Segment Data — replaces Routing API ─────────────────────────
# Flow API (/traffic/services/4/flowSegmentData/absolute/10/json) is included
# in the free TomTom Traffic tier (same key as incidents).
# It returns currentSpeed + freeFlowSpeed for a road point, giving us a real
# congestion ratio without needing the Routing product (which caused 403).
#
# One call per corridor midpoint → 7 calls per refresh.

TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

BENGALURU_CORRIDORS = [
    {"name": "Outer Ring Road",   "points": "12.9352,77.6245|12.9716,77.6412|12.9933,77.6897"},
    {"name": "Mysore Road",       "points": "12.9523,77.5150|12.9411,77.4852|12.9305,77.4601"},
    {"name": "Bellary Road",      "points": "13.0240,77.5946|13.0578,77.5870|13.0950,77.5780"},
    {"name": "Hosur Road",        "points": "12.9270,77.6210|12.8910,77.6401|12.8550,77.6602"},
    {"name": "Old Madras Road",   "points": "12.9858,77.6412|13.0100,77.6601|13.0350,77.6920"},
    {"name": "Tumkur Road",       "points": "13.0240,77.5480|13.0550,77.5120|13.0850,77.4850"},
    {"name": "Bannerghatta Road", "points": "12.9050,77.5946|12.8650,77.5946|12.8250,77.5946"},
]

async def _fetch_flow_for_corridor(client: httpx.AsyncClient, corridor: dict):
    pts = corridor["points"].split("|")
    mid = pts[len(pts) // 2].split(",")
    lat, lng = float(mid[0]), float(mid[1])
    params = {
        "key": TOMTOM_KEY,
        "point": f"{lat},{lng}",
        "unit": "KMPH",
        "openLr": "false",
    }
    try:
        resp = await client.get(TOMTOM_FLOW_URL, params=params)

        if resp.status_code == 429:
            _mark_flow_quota_exhausted()
            return None

        if resp.status_code != 200:
            logger.warning(
                "TomTom Flow API error for %s: HTTP %s — %s",
                corridor["name"], resp.status_code, resp.text[:150],
            )
            return None

        data = resp.json().get("flowSegmentData", {})
        current_speed  = float(data.get("currentSpeed", 0))
        freeflow_speed = float(data.get("freeFlowSpeed", 1)) or 1.0
        current_tt     = float(data.get("currentTravelTime", 0))
        freeflow_tt    = float(data.get("freeFlowTravelTime", 1)) or 1.0

        # Congestion = how much slower than free-flow, as a percentage
        congestion_pct = int(np_clip((1 - current_speed / freeflow_speed) * 100, 0, 99))
        level = ("Critical" if congestion_pct > 70 else "High" if congestion_pct > 45
                 else "Moderate" if congestion_pct > 20 else "Free Flow")

        # Estimate distance from corridor point spread
        pts_coords = [p.split(",") for p in pts]
        dist_km = sum(
            math.sqrt((float(pts_coords[i+1][0]) - float(pts_coords[i][0]))**2 +
                      (float(pts_coords[i+1][1]) - float(pts_coords[i][1]))**2) * 111
            for i in range(len(pts_coords) - 1)
        )

        return {
            "corridor": corridor["name"],
            "latitude": lat,
            "longitude": lng,
            "congestion_pct": congestion_pct,
            "congestion_level": level,
            "duration_normal_min": round(freeflow_tt / 60, 1),
            "duration_traffic_min": round(current_tt / 60, 1),
            "distance_km": round(dist_km, 1),
            "delay_min": round((current_tt - freeflow_tt) / 60, 1),
            "current_speed_kmph": round(current_speed, 1),
            "freeflow_speed_kmph": round(freeflow_speed, 1),
            "source": "TomTom Flow (Live)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning("TomTom Flow request failed for %s: %s", corridor["name"], e)
        return None

async def fetch_google_traffic() -> list:
    """
    Function name kept unchanged so no other file needs editing.
    Now uses TomTom Flow Segment API (free tier, same key as incidents).
    Results cached 10 minutes to stay within the 2,500 req/day limit.
    """
    global _corridor_cache, _corridor_cache_time

    if not TOMTOM_KEY:
        logger.info("No TomTom key — using ML fallback for corridors.")
        return _fallback_google_traffic()

    # Serve from cache if still fresh (this actually works now — module-level var)
    if _corridor_cache and _corridor_cache_time:
        age = (datetime.now(timezone.utc) - _corridor_cache_time).total_seconds()
        if age < CORRIDOR_CACHE_TTL:
            return _corridor_cache

    if _flow_quota_exhausted():
        return _fallback_google_traffic()

    async with httpx.AsyncClient(timeout=8) as client:
        results = await asyncio.gather(
            *[_fetch_flow_for_corridor(client, c) for c in BENGALURU_CORRIDORS]
        )
    results = [r for r in results if r is not None]

    if not results:
        logger.info("TomTom Flow: all requests failed — using ML fallback.")
        return _fallback_google_traffic()

    _corridor_cache = results
    _corridor_cache_time = datetime.now(timezone.utc)
    logger.info("TomTom Flow: fetched %d corridors, cached for %ds.", len(results), CORRIDOR_CACHE_TTL)
    return results

# ── ML fallbacks — stable, cached, not random ────────────────────────────────
def _fallback_incidents() -> list:
    return ml_sample_live_incidents(15)

def _fallback_google_traffic() -> list:
    global _ml_traffic_cache, _ml_traffic_cache_time

    if _ml_traffic_cache and _ml_traffic_cache_time:
        age = (datetime.now(timezone.utc) - _ml_traffic_cache_time).total_seconds()
        if age < ML_CACHE_TTL:
            return _ml_traffic_cache

    results = []
    for corridor in BENGALURU_CORRIDORS:
        pts = corridor["points"].split("|")
        mid = pts[len(pts) // 2].split(",")
        pred = ml_predict_corridor_now(corridor["name"])
        base = pred["congestion_pct"]
        level = "Critical" if base > 70 else "High" if base > 45 else "Moderate" if base > 20 else "Free Flow"
        results.append({
            "corridor": corridor["name"],
            "latitude": float(mid[0]),
            "longitude": float(mid[1]),
            "congestion_pct": base,
            "congestion_level": level,
            "duration_normal_min": pred["duration_normal_min"],
            "duration_traffic_min": pred["duration_traffic_min"],
            "distance_km": round(pred["duration_normal_min"] / 2.2, 1),
            "delay_min": pred["delay_min"],
            "source": "ML-predicted (TomTom Flow unavailable)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    _ml_traffic_cache = results
    _ml_traffic_cache_time = datetime.now(timezone.utc)
    return results

def _fallback_events() -> list:
    places = [
        ("Chinnaswamy Stadium",      "Sports Event",    12.9791, 77.5993, 18000, 55000),
        ("Lalbagh Botanical Garden", "Public Gathering", 12.9507, 77.5848,  2500, 11000),
        ("Cubbon Park",              "Public Event",     12.9763, 77.5929,  1500,  9000),
        ("Bannerghatta National Park","Tourist Event",   12.8019, 77.5751,   400,  3500),
        ("ISKCON Temple",            "Religious Event",  13.0094, 77.5510,  4000, 22000),
        ("Palace Grounds",           "Cultural Event",   13.0059, 77.5700,  8000, 42000),
        ("Freedom Park",             "Public Event",     12.9690, 77.5780,   800, 16000),
        ("Kanteerava Stadium",       "Sports Event",     12.9738, 77.5980,  6000, 26000),
        ("UB City Mall",             "Public Gathering", 12.9716, 77.5970,  2500, 13000),
        ("Vidhana Soudha",           "Government Event", 12.9789, 77.5917,   400,  5500),
        ("IIM Bangalore",            "Academic Event",   13.0694, 77.5994,   800,  5200),
        ("Jakkur Aerodrome",         "Special Event",    13.0820, 77.5900,   400,  3200),
        ("Koramangala Park",         "Public Gathering", 12.9340, 77.6270,   400,  5200),
    ]
    events = []
    for name, etype, lat, lng, cap_min, cap_max in places:
        crowd = ml_estimate_venue_crowd(cap_min, cap_max)
        risk = "High" if crowd > 15000 else "Moderate" if crowd > 5000 else "Low"
        events.append({
            "id": f"venue-{name[:6].replace(' ', '')}",
            "name": name, "event_type": etype, "amenity": "venue",
            "latitude": lat, "longitude": lng,
            "crowd_estimate": crowd, "risk_level": risk,
            "source": "ML-predicted (Overpass unavailable)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return events

# ── API Endpoints ─────────────────────────────────────────────────────────────
@router.get("/traffic-incidents")
async def traffic_incidents():
    data = await fetch_tomtom_incidents()
    return {
        "count": len(data),
        "source": data[0]["source"] if data else "TomTom Live (0 active incidents)",
        "incidents": data,
    }

@router.get("/google-traffic")
async def google_traffic():
    data = await fetch_google_traffic()
    return {"count": len(data), "source": data[0]["source"] if data else "none", "corridors": data}

@router.get("/live-events")
async def live_events():
    data = await fetch_osm_venues()
    return {"count": len(data), "source": data[0]["source"] if data else "none", "events": data}

@router.get("/live-snapshot")
async def live_snapshot():
    incidents, traffic, events = await asyncio.gather(
        fetch_tomtom_incidents(),
        fetch_google_traffic(),
        fetch_osm_venues(),
    )
    critical = sum(1 for t in traffic if t["congestion_level"] == "Critical")
    high_inc  = sum(1 for i in incidents if i.get("magnitude", 0) >= 3)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_incidents": len(incidents),
            "critical_corridors": critical,
            "high_severity_incidents": high_inc,
            "live_events": len(events),
            "avg_congestion_pct": round(
                sum(t["congestion_pct"] for t in traffic) / max(len(traffic), 1), 1
            ),
        },
        "incidents": incidents,
        "corridors": traffic,
        "events": events,
    }

@router.get("/config-status")
async def config_status():
    corridor_source = (
        "TomTom Flow (Live)" if (TOMTOM_KEY and not _flow_quota_exhausted()) else "ML-predicted"
    )
    return {
        "tomtom": bool(TOMTOM_KEY),
        "osm": True,
        "simulation_mode": not TOMTOM_KEY,
        "corridor_source": corridor_source,
        "incident_source": "TomTom Live" if TOMTOM_KEY else "ML-predicted",
        "flow_quota_exhausted": _flow_quota_exhausted(),
        "corridor_cache_age_sec": (
            round((datetime.now(timezone.utc) - _corridor_cache_time).total_seconds())
            if _corridor_cache_time else None
        ),
    }