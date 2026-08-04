"""
Clover Food Lab — Weather Alert Engine (v2)
============================================
Improvements over v1:
  - Seasonal context: email notes whether this week is
    historically high, normal, or low volume
  - Hourly forecast: detects lunch-window rain (9am-1pm)
    and applies timing multiplier to impact estimate
  - Timing shown explicitly in email body

Setup:
    pip install requests python-dotenv

Required GitHub Secrets:
    ALERT_EMAIL_FROM      sender Gmail address
    ALERT_EMAIL_PASSWORD  Gmail App Password
    ALERT_EMAIL_TO        comma-separated recipients
"""

import os
import json
import smtplib
import requests
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

# ── LOCATIONS ────────────────────────────────────────────────────────────────
LOCATIONS = [
    {"code": "HSC", "name": "HSC (Harvard Campus)",     "zip": "02138", "lat": 42.3770, "lon": -71.1167, "indoor": True},
    {"code": "PRU", "name": "PRU (Prudential — mall)",  "zip": "02199", "lat": 42.3467, "lon": -71.0822, "indoor": True},
    {"code": "HFI", "name": "HFI (Kendall/MIT)",        "zip": "02139", "lat": 42.3626, "lon": -71.0940, "indoor": False},
    {"code": "FIN", "name": "FIN (Financial District)", "zip": "02110", "lat": 42.3573, "lon": -71.0523, "indoor": False},
    {"code": "KEE", "name": "KEE (Food Hall — new)",    "zip": "02142", "lat": 42.3682, "lon": -71.0815, "indoor": True},
]

# ── THRESHOLDS ───────────────────────────────────────────────────────────────
THRESHOLDS = {
    "cold_mild":       35,
    "cold_moderate":   25,
    "cold_severe":     15,
    "heat_mild":       85,
    "heat_moderate":   90,
    "precip_mild":     0.05,
    "precip_moderate": 0.15,
    "precip_severe":   0.30,
    "snow_mild":       0.005,
    "snow_moderate":   0.02,
    "snow_severe":     0.05,
    "lunch_rain_heavy":0.10,
    "lunch_rain_light":0.02,
}

LUNCH_START = 9   # 9am inclusive
LUNCH_END   = 14  # 1pm inclusive (hours 9,10,11,12,13)

# ── IMPACT ESTIMATES ─────────────────────────────────────────────────────────
# Replace these with values from weather_impact_report.txt
# after running analyze_weather_sales.py
IMPACT_ESTIMATES = {
    "cold": {"mild": 6,  "moderate": 13, "severe": 22},
    "heat": {"mild": 5,  "moderate": 12, "severe": 15},
    "rain": {"mild": 5,  "moderate": 10, "severe": 14},
    "snow": {"mild": 7,  "moderate": 15, "severe": 25},
}

# ── TIMING MULTIPLIERS ───────────────────────────────────────────────────────
# Replace with values from weather_impact_report.txt
TIMING_MULTIPLIERS = {
    "lunch_heavy": 1.5,   # rain heavy in 9am-1pm window
    "lunch_light": 1.2,   # rain light in 9am-1pm window
    "off_peak":    0.6,   # rain only outside lunch window
    "dry":         1.0,
    "snow":        1.0,
}

# ── LOCATION RAIN OVERRIDES (lift locations) ─────────────────────────────────
# Positive = these locations see a LIFT on rain days
# Update from your regression output
LOCATION_OVERRIDES = {
    "HSC": {"rain": {"mild": +5, "moderate": +7, "severe": +3}},
    "PRU": {"rain": {"mild": +4, "moderate": +6, "severe": +2}},
    "KEE": {"rain": {"mild": +3, "moderate": +4, "severe": +2}},
}

# ── SEASONAL INDEX ───────────────────────────────────────────────────────────
# Week-of-year context labels (update from your regression output)
# 1.0 = typical week, >1.0 = historically busy, <1.0 = historically slow
# Approximate Boston/Cambridge seasonal pattern for now
SEASONAL_CONTEXT = {
    range(1,  4):  ("low",    "Post-holiday slow period"),
    range(4,  8):  ("normal", "Winter semester"),
    range(8,  14): ("high",   "Spring busy season"),
    range(14, 18): ("high",   "Spring peak"),
    range(18, 22): ("normal", "Late spring"),
    range(22, 27): ("low",    "Early summer / exam period"),
    range(27, 36): ("low",    "Summer slow — reduced campus traffic"),
    range(36, 40): ("high",   "Fall return — busy season"),
    range(40, 44): ("high",   "Fall peak"),
    range(44, 48): ("normal", "Late fall"),
    range(48, 53): ("low",    "Holiday / exam period"),
}

TIER_EMOJI = {"clear": "✅", "mild": "🟡", "moderate": "🟠", "severe": "🔴", "lift": "📈"}


# ── WEATHER FETCH ─────────────────────────────────────────────────────────────

def fetch_daily(lat: float, lon: float, day_index: int) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"snowfall_sum,weathercode"
        f"&temperature_unit=fahrenheit"
        f"&precipitation_unit=inch"
        f"&timezone=America%2FNew_York"
        f"&forecast_days=3"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()["daily"]
    return {
        "temp_max": d["temperature_2m_max"][day_index],
        "temp_min": d["temperature_2m_min"][day_index],
        "temp_avg": (d["temperature_2m_max"][day_index] + d["temperature_2m_min"][day_index]) / 2,
        "precip":   d["precipitation_sum"][day_index],
        "snow":     d["snowfall_sum"][day_index],
        "wx_code":  d["weathercode"][day_index],
    }


def fetch_hourly_lunch(lat: float, lon: float, target_date: str) -> dict:
    """Fetch hourly precipitation and compute lunch-window totals."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=precipitation"
        f"&precipitation_unit=inch"
        f"&timezone=America%2FNew_York"
        f"&start_date={target_date}&end_date={target_date}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()["hourly"]

    precip_by_hour = d["precipitation"]  # 24 values for the day

    lunch_precip = sum(
        precip_by_hour[h] for h in range(LUNCH_START, LUNCH_END)
        if h < len(precip_by_hour)
    )
    morning_precip = sum(
        precip_by_hour[h] for h in range(6, LUNCH_START)
        if h < len(precip_by_hour)
    )
    total_precip = sum(precip_by_hour)

    return {
        "lunch_precip":   round(lunch_precip, 3),
        "morning_precip": round(morning_precip, 3),
        "total_precip":   round(total_precip, 3),
    }


def rain_timing_label(daily: dict, hourly: dict) -> str:
    """Classify when the rain falls relative to the lunch window."""
    if daily["precip"] < THRESHOLDS["precip_mild"] and daily["snow"] < THRESHOLDS["snow_mild"]:
        return "dry"
    if daily["snow"] > THRESHOLDS["snow_mild"]:
        return "snow"
    lp = hourly.get("lunch_precip", 0)
    if lp >= THRESHOLDS["lunch_rain_heavy"]:
        return "lunch_heavy"
    if lp >= THRESHOLDS["lunch_rain_light"]:
        return "lunch_light"
    return "off_peak"


# ── SEASONAL CONTEXT ──────────────────────────────────────────────────────────

def get_seasonal_context(week: int) -> tuple[str, str]:
    for week_range, (level, label) in SEASONAL_CONTEXT.items():
        if week in week_range:
            return level, label
    return "normal", "Typical week"


# ── CLASSIFICATION ────────────────────────────────────────────────────────────

def classify(weather: dict, hourly: dict, indoor: bool, location_code: str) -> dict:
    tier   = "clear"
    impact = 0
    triggers = []
    is_lift  = False

    tier_rank = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3, "lift": 1}

    def up(new_tier, new_impact, label):
        nonlocal tier, impact
        if tier_rank.get(new_tier, 0) > tier_rank.get(tier, 0):
            tier = new_tier
        impact = max(impact, new_impact)
        triggers.append(label)

    t = weather["temp_avg"]
    p = weather["precip"]
    s = weather["snow"]

    # Cold
    if   t < THRESHOLDS["cold_severe"]:   up("severe",   IMPACT_ESTIMATES["cold"]["severe"],   f"Extreme cold ({t:.0f}°F)")
    elif t < THRESHOLDS["cold_moderate"]: up("moderate", IMPACT_ESTIMATES["cold"]["moderate"], f"Heavy cold ({t:.0f}°F)")
    elif t < THRESHOLDS["cold_mild"]:     up("mild",     IMPACT_ESTIMATES["cold"]["mild"],     f"Cold ({t:.0f}°F)")

    # Heat
    if   t > THRESHOLDS["heat_moderate"]: up("severe", IMPACT_ESTIMATES["heat"]["severe"], f"Extreme heat ({t:.0f}°F)")
    elif t > THRESHOLDS["heat_mild"]:     up("mild",   IMPACT_ESTIMATES["heat"]["mild"],   f"Heat ({t:.0f}°F)")

    # Snow
    if   s > THRESHOLDS["snow_severe"]:   up("severe",   IMPACT_ESTIMATES["snow"]["severe"],   f"Heavy snow ({s:.2f}\")")
    elif s > THRESHOLDS["snow_moderate"]: up("moderate", IMPACT_ESTIMATES["snow"]["moderate"], f"Snow ({s:.2f}\")")
    elif s > THRESHOLDS["snow_mild"]:     up("mild",     IMPACT_ESTIMATES["snow"]["mild"],     f"Flurries")

    # Rain
    rain = max(0, p - s) if s > 0 else p
    if s < THRESHOLDS["snow_mild"]:
        if   rain > THRESHOLDS["precip_severe"]:   up("moderate", IMPACT_ESTIMATES["rain"]["severe"],   f"Heavy rain ({rain:.2f}\")")
        elif rain > THRESHOLDS["precip_moderate"]: up("mild",     IMPACT_ESTIMATES["rain"]["moderate"], f"Rain ({rain:.2f}\")")
        elif rain > THRESHOLDS["precip_mild"]:     up("mild",     IMPACT_ESTIMATES["rain"]["mild"],     f"Light rain")

    # Location rain override (lift locations)
    overrides = LOCATION_OVERRIDES.get(location_code, {})
    if overrides.get("rain") and rain > THRESHOLDS["precip_mild"] and s < THRESHOLDS["snow_mild"]:
        rain_tier = ("severe" if rain > THRESHOLDS["precip_severe"]
                     else "moderate" if rain > THRESHOLDS["precip_moderate"]
                     else "mild")
        override_val = overrides["rain"].get(rain_tier)
        if override_val is not None and override_val > 0:
            impact  = override_val
            tier    = "lift"
            is_lift = True
            triggers.append("Rain lift expected")

    # Apply timing multiplier (rain only, non-lift, non-snow)
    timing = rain_timing_label(weather, hourly)
    if not is_lift and s < THRESHOLDS["snow_mild"] and rain > THRESHOLDS["precip_mild"]:
        multiplier = TIMING_MULTIPLIERS.get(timing, 1.0)
        impact = round(impact * multiplier)
        if timing == "lunch_heavy":
            triggers.append("🕙 Lunch window — heavy rain")
        elif timing == "lunch_light":
            triggers.append("🕙 Lunch window — light rain")
        elif timing == "off_peak":
            triggers.append("🕔 Off-peak timing")

    # Indoor dampening (non-lift)
    if indoor and not is_lift and tier != "clear":
        impact = round(impact * 0.5)
        if tier == "severe":
            tier = "moderate"
        triggers.append("Indoor — reduced impact")

    return {
        "tier":    tier,
        "impact":  impact,
        "is_lift": is_lift,
        "timing":  timing,
        "triggers": triggers,
    }


# ── EMAIL BUILDER ─────────────────────────────────────────────────────────────

def build_email(results: list, target_str: str, send_type: str,
                week: int) -> tuple[str, str]:

    alerts  = [r for r in results if r["classification"]["tier"] not in ("clear", "lift")]
    lifts   = [r for r in results if r["classification"]["tier"] == "lift"]
    severe  = [r for r in alerts  if r["classification"]["tier"] == "severe"]

    season_level, season_label = get_seasonal_context(week)
    season_color = {"high": "#22d68f", "normal": "#7A9DB5", "low": "#f0a500"}[season_level]
    season_emoji = {"high": "📈", "normal": "→", "low": "📉"}[season_level]

    prefix = "📅 ADVANCE —" if send_type == "advance" else "☀️ DAY-OF —"
    send_label = "Advance Planning Alert" if send_type == "advance" else "Day-Of Confirmation"

    if not alerts and not lifts:
        subject = f"✅ {prefix} Clover Weather — All Clear · {target_str}"
    elif severe:
        subject = f"🔴 {prefix} Clover Weather — {len(severe)} SEVERE · {target_str}"
    else:
        subject = f"⚠️ {prefix} Clover Weather — {len(alerts)} alert(s) · {target_str}"

    tier_colors = {
        "clear":    "#22d68f",
        "lift":     "#00c4b4",
        "mild":     "#f0a500",
        "moderate": "#e8621a",
        "severe":   "#e03045",
    }

    rows_html = ""
    for r in results:
        loc = r["location"]
        wx  = r["weather"]
        hr  = r["hourly"]
        cl  = r["classification"]
        tier = cl["tier"]
        color = tier_colors.get(tier, "#7A9DB5")

        impact_str = ("—" if tier == "clear"
                      else f"+{cl['impact']}%" if cl["is_lift"]
                      else f"-{cl['impact']}%")

        timing_str = ""
        if cl["timing"] == "lunch_heavy":
            timing_str = f"<br><span style='color:#f0a500;font-size:11px'>🕙 Rain peaks 9am–1pm ({hr.get('lunch_precip',0):.2f}\")</span>"
        elif cl["timing"] == "lunch_light":
            timing_str = f"<br><span style='color:#f0a500;font-size:11px'>🕙 Light lunch-window rain ({hr.get('lunch_precip',0):.2f}\")</span>"
        elif cl["timing"] == "off_peak":
            timing_str = "<br><span style='color:#7A9DB5;font-size:11px'>🕔 Rain after lunch rush</span>"

        precip_str = (f"{wx['snow']:.2f}\" snow" if wx['snow'] > 0.01
                      else f"{wx['precip']:.2f}\" rain" if wx['precip'] > 0.05
                      else "Dry")

        indoor_tag = ("🏢" if loc.get("indoor") else "🌿")
        tier_label = {"clear":"CLEAR","lift":"LIFT","mild":"MILD","moderate":"MOD","severe":"SEV"}.get(tier, tier.upper())

        rows_html += f"""
        <tr>
          <td style="padding:14px 16px;border-bottom:1px solid #1e2a3a;">
            <strong style="color:#e2e6f0">{indoor_tag} {loc['name']}</strong>
            <br><span style="font-family:monospace;font-size:10px;color:#4a6275">{loc['zip']}</span>
          </td>
          <td style="padding:14px 16px;border-bottom:1px solid #1e2a3a;text-align:center">
            <span style="background:{color}22;color:{color};border:1px solid {color}44;
                         padding:3px 10px;border-radius:20px;font-size:11px;font-family:monospace">
              {TIER_EMOJI.get(tier,'?')} {tier_label}
            </span>
          </td>
          <td style="padding:14px 16px;border-bottom:1px solid #1e2a3a;font-family:monospace;
                     font-size:14px;color:{color};text-align:center;font-weight:500">
            {impact_str}
          </td>
          <td style="padding:14px 16px;border-bottom:1px solid #1e2a3a;font-size:12px;color:#8aa0b5">
            {wx['temp_min']:.0f}°–{wx['temp_max']:.0f}°F · {precip_str}
            {timing_str}
          </td>
          <td style="padding:14px 16px;border-bottom:1px solid #1e2a3a;font-size:11px;color:#4a6275">
            {', '.join(cl['triggers']) if cl['triggers'] else '—'}
          </td>
        </tr>"""

    mild_c = sum(1 for r in alerts if r["classification"]["tier"] == "mild")
    mod_c  = sum(1 for r in alerts if r["classification"]["tier"] == "moderate")
    sev_c  = len(severe)
    lift_c = len(lifts)
    clear_c = len(results) - len(alerts) - lift_c

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width"></head>
<body style="margin:0;padding:0;background:#080b12;font-family:'Helvetica Neue',Arial,sans-serif;">
<div style="max-width:740px;margin:0 auto;padding:32px 16px;">

  <!-- Header -->
  <div style="display:flex;align-items:center;margin-bottom:8px;">
    <div style="background:rgba(34,214,143,0.1);border:1px solid rgba(34,214,143,0.2);
                width:40px;height:40px;border-radius:10px;text-align:center;
                line-height:40px;font-size:20px;margin-right:14px">🍀</div>
    <div>
      <div style="font-size:17px;font-weight:600;color:#e2e6f0">Clover Weather Monitor</div>
      <div style="font-size:10px;color:#4a6275;font-family:monospace;margin-top:2px;letter-spacing:0.5px">
        {send_label.upper()} · {target_str.upper()}
      </div>
    </div>
  </div>

  <!-- Seasonal context banner -->
  <div style="background:#0f1824;border:1px solid #1e2a3a;border-radius:10px;
              padding:10px 16px;margin-bottom:20px;margin-top:16px;
              display:flex;align-items:center;gap:10px;">
    <span style="font-size:16px">{season_emoji}</span>
    <div>
      <span style="font-size:12px;color:{season_color};font-weight:500">
        {season_label}
      </span>
      <span style="font-size:11px;color:#4a6275;font-family:monospace;margin-left:8px">
        Week {week} · Seasonal volume: {season_level.upper()}
      </span>
    </div>
  </div>

  <!-- Summary chips -->
  <div style="display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap">
    <span style="background:#22d68f22;color:#22d68f;border:1px solid #22d68f44;
                 padding:4px 12px;border-radius:20px;font-size:11px;font-family:monospace">
      ✅ Clear: {clear_c}
    </span>
    <span style="background:#00c4b422;color:#00c4b4;border:1px solid #00c4b444;
                 padding:4px 12px;border-radius:20px;font-size:11px;font-family:monospace">
      📈 Lift: {lift_c}
    </span>
    <span style="background:#f0a50022;color:#f0a500;border:1px solid #f0a50044;
                 padding:4px 12px;border-radius:20px;font-size:11px;font-family:monospace">
      🟡 Mild: {mild_c}
    </span>
    <span style="background:#e8621a22;color:#e8621a;border:1px solid #e8621a44;
                 padding:4px 12px;border-radius:20px;font-size:11px;font-family:monospace">
      🟠 Moderate: {mod_c}
    </span>
    <span style="background:#e0304522;color:#e03045;border:1px solid #e0304544;
                 padding:4px 12px;border-radius:20px;font-size:11px;font-family:monospace">
      🔴 Severe: {sev_c}
    </span>
  </div>

  <!-- Location table -->
  <div style="background:#0d1420;border:1px solid #1e2a3a;border-radius:12px;overflow:hidden;margin-bottom:24px">
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="background:#111c2c">
          <th style="padding:10px 16px;text-align:left;font-size:10px;color:#4a6275;font-family:monospace;letter-spacing:0.5px;font-weight:500">LOCATION</th>
          <th style="padding:10px 16px;text-align:center;font-size:10px;color:#4a6275;font-family:monospace;letter-spacing:0.5px;font-weight:500">TIER</th>
          <th style="padding:10px 16px;text-align:center;font-size:10px;color:#4a6275;font-family:monospace;letter-spacing:0.5px;font-weight:500">IMPACT</th>
          <th style="padding:10px 16px;text-align:left;font-size:10px;color:#4a6275;font-family:monospace;letter-spacing:0.5px;font-weight:500">FORECAST</th>
          <th style="padding:10px 16px;text-align:left;font-size:10px;color:#4a6275;font-family:monospace;letter-spacing:0.5px;font-weight:500">TRIGGERS</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <!-- Note -->
  <div style="background:#0d1420;border:1px solid #1e2a3a;border-radius:10px;
              padding:14px 18px;margin-bottom:24px;font-size:11px;color:#4a6275;line-height:1.7">
    <strong style="color:#7A9DB5">Impact estimates</strong> are seasonally adjusted —
    measured vs expected sales for this week of the year, not annual average.
    Rain impacts include a timing multiplier based on whether precipitation
    falls in the 9am–1pm lunch window.
  </div>

  <!-- Footer -->
  <div style="border-top:1px solid #1e2a3a;padding-top:18px;font-size:10px;
              color:#2a3a4a;font-family:monospace;line-height:1.8">
    Weather: Open-Meteo API · Sent {send_label} at
    {'3:00 PM' if send_type == 'advance' else '7:00 AM'} ET daily<br>
    Clover Food Lab Operations Intelligence
  </div>

</div></body></html>"""

    return subject, html


# ── EMAIL SENDER ──────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    from_addr = os.environ["ALERT_EMAIL_FROM"]
    to_addrs  = [e.strip() for e in os.environ["ALERT_EMAIL_TO"].split(",")]
    password  = os.environ["ALERT_EMAIL_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Clover Weather Monitor <{from_addr}>"
    msg["To"]      = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addrs, msg.as_string())

    print(f"✓ Email sent to: {', '.join(to_addrs)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    et  = ZoneInfo("America/New_York")
    now = datetime.now(et)

    if now.hour >= 14:
        target_day = now + timedelta(days=1)
        send_type  = "advance"
        day_index  = 1
    else:
        target_day = now
        send_type  = "dayof"
        day_index  = 0

    target_str  = target_day.strftime("%A, %b %-d")
    target_date = target_day.strftime("%Y-%m-%d")
    week        = target_day.isocalendar()[1]

    season_level, season_label = get_seasonal_context(week)
    send_label = "Advance" if send_type == "advance" else "Day-Of"

    print(f"🍀 Clover Weather Alert Engine (v2) — {target_str} [{send_label}]")
    print(f"   Week {week} · Seasonal context: {season_label}")
    print(f"   Checking {len(LOCATIONS)} locations...\n")

    results = []
    for loc in LOCATIONS:
        try:
            daily  = fetch_daily(loc["lat"], loc["lon"], day_index)
            hourly = fetch_hourly_lunch(loc["lat"], loc["lon"], target_date)
            cl     = classify(daily, hourly, loc.get("indoor", False), loc["code"])

            results.append({
                "location":       loc,
                "weather":        daily,
                "hourly":         hourly,
                "classification": cl,
            })

            tier   = cl["tier"]
            emoji  = TIER_EMOJI.get(tier, "?")
            impact_str = (f"+{cl['impact']}%" if cl["is_lift"]
                          else f"-{cl['impact']}%" if tier != "clear"
                          else "clear")
            timing_note = {
                "lunch_heavy": " 🕙 lunch-window rain",
                "lunch_light": " 🕙 light lunch rain",
                "off_peak":    " 🕔 off-peak rain",
            }.get(cl["timing"], "")
            print(f"  {emoji} {loc['name']:<32} {impact_str:<8} {timing_note}")

        except Exception as e:
            print(f"  ❌ {loc['name']}: {e}")

    subject, html = build_email(results, target_str, send_type, week)
    print(f"\n  Subject: {subject}")
    send_email(subject, html)
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
