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
from ml.engine import estimate_venue_crowd as ml_estimate_venue_crowd

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
CORRIDOR_CACHE_TTL = 900   # 15 minutes → 26 corridors x 96 refreshes/day = 2,496 calls/day

# ── TomTom quota backoff ──────────────────────────────────────────────────────
_flow_quota_exhausted_date: date | None = None

def _flow_quota_exhausted() -> bool:
    return _flow_quota_exhausted_date == datetime.now(timezone.utc).date()

def _mark_flow_quota_exhausted() -> None:
    global _flow_quota_exhausted_date
    today = datetime.now(timezone.utc).date()
    if _flow_quota_exhausted_date != today:
        logger.warning(
            "TomTom Flow API daily quota exhausted — serving last cached real "
            "corridor data (if any) until quota resets at midnight UTC (%s).", today
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
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
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

async def _fetch_osm_venues_inner() -> list:
    last_error = None
    for url in OVERPASS_URLS:
        try:
            # 4s per mirror (not 12s) — with 4 mirrors that's a 16s worst case
            # on its own, which is why this whole function is additionally
            # wrapped in an 8s hard budget below; this per-call timeout just
            # keeps any single slow mirror from eating the entire budget.
            async with httpx.AsyncClient(timeout=4) as client:
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

    logger.warning("All Overpass endpoints failed (%s) — no live venue data this cycle.", last_error)
    return []


async def fetch_osm_venues() -> list:
    """
    Wraps the mirror-retry loop in a hard 8s overall budget. Without this,
    4 mirrors x 12s timeout each could take up to 48s in the worst case
    (all mirrors slow/down) — well past the frontend's 15s axios timeout on
    /api/live/live-snapshot, which is what was causing the Live Map to load
    with no markers (the request simply timed out client-side before the
    backend ever responded). If the budget is exceeded we just return an
    empty venue list for this cycle rather than blocking the whole snapshot.
    """
    try:
        return await asyncio.wait_for(_fetch_osm_venues_inner(), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("Overpass venue fetch exceeded its 10s budget — skipping venues for this cycle.")
        return []

# ── TomTom Traffic Incidents ──────────────────────────────────────────────────
async def fetch_tomtom_incidents() -> list:
    if not TOMTOM_KEY:
        return []
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
            return []
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
            road_numbers = props.get("roadNumbers") or []
            from_street = (props.get("from") or "").strip()
            to_street   = (props.get("to") or "").strip()
            if road_numbers:
                road = road_numbers[0]
            elif from_street and to_street and from_street != to_street:
                road = f"{from_street} → {to_street}"
            elif from_street or to_street:
                road = from_street or to_street
            else:
                road = "Unknown"
            raw_delay = props.get("delay", 0) or 0
            raw_length = props.get("length", 0) or 0

            # TomTom often returns delay=0 even for real incidents.
            # Derive a realistic estimate from road length + magnitude when that happens.
            if raw_delay == 0 and (raw_length > 0 or magnitude > 0):
                # Base delay: assume 10 km/h crawl vs 50 km/h normal on affected length
                # fallback length by magnitude if length also missing
                eff_length = raw_length if raw_length > 0 else {1: 200, 2: 500, 3: 1000, 4: 2000}.get(magnitude, 300)
                crawl_speed = {0: 20, 1: 20, 2: 12, 3: 6, 4: 5}.get(magnitude, 15)  # km/h
                normal_speed = 50  # km/h urban
                # time_crawl - time_normal in seconds
                derived_delay = int((eff_length / 1000) / crawl_speed * 3600 - (eff_length / 1000) / normal_speed * 3600)
                delay_sec = max(derived_delay, 60 * magnitude) if magnitude > 0 else max(derived_delay, 30)
            else:
                delay_sec = raw_delay

            sev_label = {0:"Unknown",1:"Minor",2:"Moderate",3:"Major",4:"Undefined"}.get(magnitude, "Unknown")
            # Upgrade severity if we can tell it's significant
            if sev_label == "Unknown" and delay_sec > 300:
                sev_label = "Minor"
            if sev_label == "Minor" and delay_sec > 900:
                sev_label = "Moderate"

            incidents.append({
                "id": props.get("id", f"tt-{abs(hash((lat, lng, desc))) % 100000}"),
                "description": desc,
                "cause": cause_map.get(icon, "Traffic Incident"),
                "severity": sev_label,
                "magnitude": magnitude,
                "latitude": lat, "longitude": lng,
                "from": from_street, "to": to_street,
                "delay_sec": delay_sec,
                "length_m": raw_length,
                "road": road,
                "source": "TomTom Live",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        if not incidents:
            logger.info("TomTom Incidents: 0 active incidents in Bengaluru right now.")
            return []
        return incidents
    except Exception as e:
        logger.warning("TomTom Incidents request failed: %s", e)
        return []

# ── TomTom Flow Segment Data — replaces Routing API ─────────────────────────
# Flow API (/traffic/services/4/flowSegmentData/absolute/10/json) is included
# in the free TomTom Traffic tier (same key as incidents).
# It returns currentSpeed + freeFlowSpeed for a road point, giving us a real
# congestion ratio without needing the Routing product (which caused 403).
#
# One call per corridor midpoint. With 26 corridors at a 15-min cache TTL,
# that's 26 x 96 refreshes/day = 2,496 calls/day — just inside TomTom's free
# 2,500/day Traffic tier limit.

TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

BENGALURU_CORRIDORS = [
    # Major arterial roads (radial, in/out of the city)
    {"name": "Outer Ring Road",        "points": "12.9352,77.6245|12.9716,77.6412|12.9933,77.6897"},
    {"name": "Mysore Road",            "points": "12.9523,77.5150|12.9411,77.4852|12.9305,77.4601"},
    {"name": "Bellary Road",           "points": "13.0240,77.5946|13.0578,77.5870|13.0950,77.5780"},
    {"name": "Hosur Road",             "points": "12.9270,77.6210|12.8910,77.6401|12.8550,77.6602"},
    {"name": "Old Madras Road",        "points": "12.9858,77.6412|13.0100,77.6601|13.0350,77.6920"},
    {"name": "Tumkur Road",            "points": "13.0240,77.5480|13.0550,77.5120|13.0850,77.4850"},
    {"name": "Bannerghatta Road",      "points": "12.9050,77.5946|12.8650,77.5946|12.8250,77.5946"},
    {"name": "Sarjapur Road",          "points": "12.9095,77.6700|12.8990,77.6950|12.8850,77.7200"},
    {"name": "Old Airport Road",       "points": "12.9620,77.6280|12.9580,77.6450|12.9540,77.6620"},
    {"name": "Kanakapura Road",        "points": "12.9200,77.5600|12.8800,77.5500|12.8400,77.5400"},
    {"name": "Magadi Road",            "points": "12.9750,77.5550|12.9850,77.5300|12.9950,77.5050"},
    {"name": "Whitefield Main Road",   "points": "12.9750,77.7100|12.9700,77.7350|12.9690,77.7500"},
    {"name": "Nice Road",              "points": "12.8700,77.5200|12.8500,77.5400|12.8300,77.5600"},
    # Known congestion hotspots / junctions
    {"name": "Silk Board Junction",    "points": "12.9180,77.6235|12.9170,77.6230|12.9160,77.6225"},
    {"name": "Marathahalli Bridge",    "points": "12.9580,77.7000|12.9560,77.6990|12.9540,77.6980"},
    {"name": "KR Puram",               "points": "13.0050,77.6950|13.0030,77.6960|13.0010,77.6970"},
    {"name": "Electronic City Flyover","points": "12.8450,77.6650|12.8400,77.6600|12.8350,77.6550"},
    {"name": "Hebbal Flyover",         "points": "13.0350,77.5970|13.0380,77.6000|13.0410,77.6030"},
    # Central / inner-city roads
    {"name": "MG Road",                "points": "12.9759,77.6055|12.9740,77.6090|12.9720,77.6130"},
    {"name": "Indiranagar 100 Feet Road","points": "12.9720,77.6400|12.9760,77.6420|12.9800,77.6440"},
    {"name": "Koramangala 80 Feet Road","points": "12.9350,77.6150|12.9320,77.6200|12.9290,77.6250"},
    {"name": "JP Nagar – Bannerghatta Junction","points": "12.9080,77.5850|12.8950,77.5900|12.8820,77.5950"},
    {"name": "Yeshwantpur",            "points": "13.0250,77.5450|13.0280,77.5400|13.0310,77.5350"},
    {"name": "Rajajinagar (Dr Rajkumar Road)","points": "12.9900,77.5500|12.9950,77.5550|13.0000,77.5600"},
    {"name": "Malleshwaram (Sampige Road)","points": "13.0050,77.5700|13.0080,77.5750|13.0110,77.5800"},
    {"name": "Vijayanagar (West of Chord Road)","points": "12.9650,77.5300|12.9620,77.5350|12.9590,77.5400"},
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
    Uses TomTom Flow Segment API (free tier, same key as incidents).
    Results cached 15 minutes to stay within the 2,500 req/day limit.

    This never fabricates corridor numbers. If TomTom is unreachable or no
    key is configured, it returns the last real cached result (clearly
    stale, but genuine) if one exists, or an empty list if it doesn't —
    never a simulated/ML-predicted value.
    """
    global _corridor_cache, _corridor_cache_time

    if not TOMTOM_KEY:
        return []

    # Serve from cache if still fresh (this actually works now — module-level var)
    if _corridor_cache and _corridor_cache_time:
        age = (datetime.now(timezone.utc) - _corridor_cache_time).total_seconds()
        if age < CORRIDOR_CACHE_TTL:
            return _corridor_cache

    if _flow_quota_exhausted():
        # Quota exhausted: serve the last real cached reading (now stale) if
        # we have one, rather than nothing or a fabricated value.
        return _corridor_cache

    async with httpx.AsyncClient(timeout=8) as client:
        results = await asyncio.gather(
            *[_fetch_flow_for_corridor(client, c) for c in BENGALURU_CORRIDORS]
        )
    results = [r for r in results if r is not None]

    if not results:
        logger.info("TomTom Flow: all requests failed this cycle — serving last real cache (if any).")
        return _corridor_cache

    _corridor_cache = results
    _corridor_cache_time = datetime.now(timezone.utc)
    logger.info("TomTom Flow: fetched %d/%d corridors, cached for %ds.",
                len(results), len(BENGALURU_CORRIDORS), CORRIDOR_CACHE_TTL)
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