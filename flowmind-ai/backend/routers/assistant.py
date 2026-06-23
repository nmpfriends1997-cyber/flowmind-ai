from fastapi import APIRouter
from pydantic import BaseModel
import httpx, os
from datetime import datetime, timezone, timedelta
from ml.engine import get_summary_stats, get_zone_risk, get_cause_distribution, get_closure_by_cause

router = APIRouter()

# ── Get your FREE OpenRouter API key at: https://openrouter.ai ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models in priority order — if first fails, next is tried automatically
FREE_MODELS = [
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free",
]


def get_bangalore_time_context() -> str:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    hour = ist.hour
    time_str = ist.strftime("%I:%M %p IST, %A")

    if (8 <= hour <= 10) or (17 <= hour <= 21):
        level = "PEAK HOURS — heavy congestion expected on ORR, Silk Board, Hebbal, Marathahalli"
    elif 11 <= hour < 17:
        level = "OFF-PEAK — moderate traffic, relatively good time to travel"
    elif hour >= 22 or hour < 6:
        level = "LOW TRAFFIC — roads mostly clear, good time to travel"
    else:
        level = "MODERATE — transitioning in/out of peak hours"

    return f"Current Bangalore time: {time_str}. Status: {level}."


def build_system_prompt() -> str:
    stats       = get_summary_stats()
    causes      = get_cause_distribution()[:6]
    zones       = get_zone_risk()[:3]
    closure     = get_closure_by_cause()
    top_closure = closure[0] if closure else None

    causes_str = ", ".join(f"{c['cause']} ({c['count']})" for c in causes)
    zones_str  = ", ".join(z["name"] for z in zones)
    time_ctx   = get_bangalore_time_context()

    return f"""You are FlowMind AI — an expert traffic intelligence assistant for Bengaluru (Bangalore), Karnataka, India.
You help traffic authorities, commuters, and planners with real-time congestion advice, route planning, incident analysis, and resource deployment.

LIVE ML DATASET (refreshed every request):
- {stats['total_events']:,} total events tracked ({stats['planned_events']:,} planned, {stats['unplanned_events']:,} unplanned)
- Top incident causes: {causes_str}
- {stats['active_events']:,} currently active incidents
- {stats['high_priority']:,} high-priority events ({stats['high_priority']/max(stats['total_events'],1)*100:.1f}% of total)
- {stats['road_closures']:,} road closure events (closure rate: {stats['closure_rate_pct']}%)
- Average incident duration: {stats['avg_duration_min']} minutes
- Top risk zones: {zones_str}
- Highest closure-rate cause: {top_closure['cause'] if top_closure else 'n/a'} ({top_closure['closure_rate'] if top_closure else '-'}% closure rate)

CURRENT TIME CONTEXT:
{time_ctx}

BANGALORE TRAFFIC HOTSPOTS:
1. Silk Board Junction — WORST in city. Weekday peak: up to 2hr delay. ORR meets Hosur Road.
2. Marathahalli Bridge — Daily gridlock on ORR tech corridor.
3. Hebbal Flyover — NH7/NH44 merge. Severe northbound mornings (airport traffic).
4. KR Puram Bridge — Single-lane choke on Old Madras Road.
5. Tin Factory Junction — Poor signal timing, large 6-way intersection.
6. Bellandur Junction — Evening standstill. ORR near Ecospace/Prestige.
7. Electronic City Flyover — NICE Road & Hosur Road merge.
8. Bannerghatta Road — BTM Layout to Hulimavu, school + office hours.
9. Ejipura / Koramangala 6th Block — Inner-city snarl.
10. Yeshwantpur Circle — Tumkur Road meets Ring Road.

KEY ROAD CORRIDORS:
- Outer Ring Road (ORR): Hebbal to Marathahalli to Silk Board (main IT corridor, always heavy)
- Old Airport Road to Indiranagar to Whitefield (east Bangalore tech route)
- MG Road / Brigade Road: CBD, constant congestion
- Tumkur Road (NH48): Peenya industrial area, morning inbound heavy
- Hosur Road (NH44): Electronic City IT hub, peak both directions
- Mysore Road (NH275): westbound congestion, NICE Road alternative available
- Bellary Road (NH44): north Bangalore, airport surges before flight times
- Sarjapur Road: worsening due to new tech parks
- Bannerghatta Road: south Bangalore, slow all day

NAMMA METRO (2025-2026):
- Purple Line (Baiyappanahalli to Mysuru Road): operational, reduces ORR load significantly
- Green Line (Nagasandra to Silk Board): Phase 2 construction, lane closures near Shivajinagar, Majestic, Jayanagar
- Phase 2 active construction at Koramangala, Hebbal, Sarjapur Road — expect 20-40 min extra delays nearby
- Best Metro use cases: MG Road to Indiranagar, Majestic to Byappanahalli, Whitefield corridor

TRAFFIC PATTERNS:
- Morning peak: 8:00-11:00am (inbound to CBD/ORR/IT parks)
- Evening peak: 5:30-9:30pm (outbound, worst 6-8pm)
- Worst days: Monday (start-of-week surge), Friday (early evening exodus)
- Best travel window: 11am-4pm weekdays, before 9am on weekends
- School zones spike 8-9am: JP Nagar, Koramangala, Indiranagar, Malleshwaram, Jayanagar
- Rain adds 30-60 min to all peak routes (roads flood near underpasses)

ALTERNATE ROUTES:
- Avoid Silk Board: use NICE Road (toll) to Electronic City Flyover
- ORR jam: use Sarjapur Road to HSR Layout to BTM Layout
- Hebbal backed up: use Tumkur Road to Jalahalli Cross to Yeshwantpur
- MG Road congested: use Richmond Road to Residency Road
- Whitefield peak: use Old Madras Road via KR Puram (off-peak only)

RESPONSE RULES:
- Always mention whether it is currently peak or off-peak based on the time context above
- Use specific junction and road names, never vague directions
- Reference ML dataset numbers when relevant
- Suggest Namma Metro when it is a viable option
- Keep responses under 200 words unless a detailed route or breakdown is requested
- Always recommend cross-checking with Google Maps for real-time GPS conditions
- Tone: professional, helpful, like a senior Bengaluru traffic control officer
- If asked about a route, give the main route + estimated delay + best alternate"""


class ChatRequest(BaseModel):
    message: str
    history: list = []


async def call_openrouter(messages: list, model: str) -> dict:
    """Call OpenRouter with a specific model and return the raw response data."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.9,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://flowmind-ai.app",
                "X-Title": "FlowMind AI",
            },
            json=payload,
        )
    return resp.json()


@router.post("/chat")
async def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("your_"):
        return {
            "reply": (
                "⚠️ OpenRouter API key not set.\n"
                "Get your FREE key at: https://openrouter.ai\n"
                "Then add OPENROUTER_API_KEY=your_key to your backend/.env file."
            )
        }

    system_prompt = build_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]
    for h in req.history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    # Try each free model in order until one works
    last_error = ""
    for model in FREE_MODELS:
        try:
            data = await call_openrouter(messages, model)

            if "choices" in data and data["choices"]:
                reply = data["choices"][0]["message"]["content"]
                return {"reply": reply}

            # Provider error — try next model
            last_error = data.get("error", {}).get("message", str(data)) if "error" in data else str(data)

        except Exception as e:
            last_error = str(e)
            continue  # try next model

    # All models failed
    return {"reply": f"⚠️ All models unavailable. Last error: {last_error}"}