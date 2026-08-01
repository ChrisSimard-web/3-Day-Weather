"""
Clover Food Lab — Weather Alert Engine
Runs daily at 7:00 AM ET via GitHub Actions.
Fetches tomorrow's forecast for each location and sends email alerts.

Setup:
  pip install requests python-dotenv

Required GitHub Secrets:
  ALERT_EMAIL_FROM     — sender address (Gmail recommended)
  ALERT_EMAIL_PASSWORD — Gmail App Password (not your main password)
  ALERT_EMAIL_TO       — comma-separated list of recipient emails
"""

import os
import json
import smtplib
import requests
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

LOCATIONS = [
    {"code": "HSC", "name": "HSC (Harvard Campus)",     "zip": "02138", "lat": 42.3770, "lon": -71.1167, "indoor": True},
    {"code": "PRU", "name": "PRU (Prudential — mall)",  "zip": "02199", "lat": 42.3467, "lon": -71.0822, "indoor": True},
    {"code": "HFI", "name": "HFI (Kendall/MIT)",        "zip": "02139", "lat": 42.3626, "lon": -71.0940, "indoor": False},
    {"code": "FIN", "name": "FIN (Financial District)", "zip": "02110", "lat": 42.3573, "lon": -71.0523, "indoor": False},
    {"code": "KEE", "name": "KEE (Food Hall — new)",    "zip": "02142", "lat": 42.3682, "lon": -71.0815, "indoor": True},
]

# Tiered thresholds
THRESHOLDS = {
    "cold_mild":      35,   # °F avg — Mild tier trigger
    "cold_moderate":  25,   # °F avg — Moderate tier trigger
    "cold_severe":    15,   # °F avg — Severe tier trigger
    "heat_mild":      85,   # °F avg — Mild tier trigger
    "heat_moderate":  90,   # °F avg — Severe tier trigger (your setting)
    "precip_mild":    0.05, # inches rain — Mild trigger
    "precip_moderate":0.15, # inches rain — Moderate trigger
    "precip_severe":  0.30, # inches rain — Severe trigger
    "snow_mild":      0.005,# inches snow — Mild trigger
    "snow_moderate":  0.02, # inches snow — Moderate trigger
    "snow_severe":    0.05, # inches snow — Severe trigger
}

# Estimated sales impact by tier (update with your real regression data)
IMPACT_ESTIMATES = {
    "cold":   {"mild": 6,  "moderate": 13, "severe": 22},
    "heat":   {"mild": 5,  "moderate": 12, "severe": 15},
    "rain":   {"mild": 5,  "moderate": 10, "severe": 14},
    "snow":   {"mild": 7,  "moderate": 15, "severe": 25},
}

TIER_EMOJI = {"clear": "✅", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}


# ─── WEATHER FETCH ────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, day_index: int = 1) -> dict:
    """
    Fetch forecast from Open-Meteo (free, no key needed).
    day_index: 0 = today, 1 = tomorrow
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,weathercode"
        f"&temperature_unit=fahrenheit"
        f"&precipitation_unit=inch"
        f"&timezone=America%2FNew_York"
        f"&forecast_days=3"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()["daily"]

    return {
        "temp_max":  d["temperature_2m_max"][day_index],
        "temp_min":  d["temperature_2m_min"][day_index],
        "temp_avg":  (d["temperature_2m_max"][day_index] + d["temperature_2m_min"][day_index]) / 2,
        "precip":    d["precipitation_sum"][day_index],
        "snow":      d["snowfall_sum"][day_index],
        "wx_code":   d["weathercode"][day_index],
    }


# ─── CLASSIFICATION ───────────────────────────────────────────────────────────

def classify(weather: dict, indoor: bool = False) -> dict:
    """
    Returns tier (clear/mild/moderate/severe), estimated impact %, and trigger list.
    Indoor locations (mall, campus, kiosk, food hall) get dampened impact estimates
    since foot traffic is less directly affected by outdoor conditions.
    """
    tier = "clear"
    impact = 0
    triggers = []

    tier_rank = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3}
    def upgrade(new_tier, new_impact, label):
        nonlocal tier, impact
        if tier_rank[new_tier] > tier_rank[tier]:
            tier = new_tier
        impact = max(impact, new_impact)
        triggers.append(label)

    t = weather["temp_avg"]
    p = weather["precip"]
    s = weather["snow"]

    # Cold
    if t < THRESHOLDS["cold_severe"]:
        upgrade("severe", IMPACT_ESTIMATES["cold"]["severe"], f"Extreme cold ({t:.0f}°F)")
    elif t < THRESHOLDS["cold_moderate"]:
        upgrade("moderate", IMPACT_ESTIMATES["cold"]["moderate"], f"Heavy cold ({t:.0f}°F)")
    elif t < THRESHOLDS["cold_mild"]:
        upgrade("mild", IMPACT_ESTIMATES["cold"]["mild"], f"Cold ({t:.0f}°F)")

    # Heat
    if t > THRESHOLDS["heat_moderate"]:
        upgrade("severe", IMPACT_ESTIMATES["heat"]["severe"], f"Extreme heat ({t:.0f}°F)")
    elif t > THRESHOLDS["heat_mild"]:
        upgrade("mild", IMPACT_ESTIMATES["heat"]["mild"], f"Heat ({t:.0f}°F)")

    # Snow (check before rain since precip includes snowfall)
    if s > THRESHOLDS["snow_severe"]:
        upgrade("severe", IMPACT_ESTIMATES["snow"]["severe"], f"Heavy snow ({s:.2f}\")")
    elif s > THRESHOLDS["snow_moderate"]:
        upgrade("moderate", IMPACT_ESTIMATES["snow"]["moderate"], f"Snow ({s:.2f}\")")
    elif s > THRESHOLDS["snow_mild"]:
        upgrade("mild", IMPACT_ESTIMATES["snow"]["mild"], f"Flurries ({s:.2f}\")")

    # Rain (only if not mostly snow)
    rain = p - s if p > s else p
    if s < 0.01:
        if rain > THRESHOLDS["precip_severe"]:
            upgrade("moderate", IMPACT_ESTIMATES["rain"]["severe"], f"Heavy rain ({rain:.2f}\")")
        elif rain > THRESHOLDS["precip_moderate"]:
            upgrade("mild", IMPACT_ESTIMATES["rain"]["moderate"], f"Rain ({rain:.2f}\")")
        elif rain > THRESHOLDS["precip_mild"]:
            upgrade("mild", IMPACT_ESTIMATES["rain"]["mild"], f"Light rain ({rain:.2f}\")")

    # Indoor locations: dampen impact by ~50% and cap tier at moderate
    # (customers still arrive — they just commute through the weather)
    if indoor and tier != "clear":
        impact = round(impact * 0.5)
        if tier == "severe":
            tier = "moderate"
        triggers.append("indoor — reduced impact")

    return {"tier": tier, "impact": impact, "triggers": triggers}


# ─── EMAIL BUILDER ────────────────────────────────────────────────────────────

def build_email(results: list, tomorrow_str: str, send_type: str = "advance") -> tuple[str, str]:
    """Returns (subject, html_body) for the alert email."""

    alerts = [r for r in results if r["classification"]["tier"] != "clear"]
    severe = [r for r in alerts if r["classification"]["tier"] == "severe"]

    prefix = "📅 ADVANCE —" if send_type == "advance" else "☀️ DAY-OF —"
    send_label = "Advance Planning Alert" if send_type == "advance" else "Day-Of Confirmation"

    if not alerts:
        subject = f"✅ {prefix} Clover Weather — All Clear for {tomorrow_str}"
    elif severe:
        subject = f"🔴 {prefix} Clover Weather — {len(severe)} SEVERE location(s) — {tomorrow_str}"
    else:
        subject = f"⚠️ {prefix} Clover Weather — {len(alerts)} location(s) affected — {tomorrow_str}"

    # Build location rows
    rows_html = ""
    for r in results:
        loc = r["location"]
        wx = r["weather"]
        cl = r["classification"]
        tier = cl["tier"]
        emoji = TIER_EMOJI[tier]
        tier_colors = {
            "clear":    "#2ecc8a",
            "mild":     "#f5c842",
            "moderate": "#f58a1f",
            "severe":   "#e8445a",
        }
        color = tier_colors[tier]
        impact_str = f"-{cl['impact']}%" if cl['impact'] > 0 else "—"
        triggers_str = ", ".join(cl["triggers"]) if cl["triggers"] else "No triggers"

        rows_html += f"""
        <tr>
          <td style="padding:12px 16px; border-bottom:1px solid #1e2230;">
            <strong style="color:#e8eaf0">{loc['name']}</strong>
            <br><span style="font-size:11px; color:#5a6080; font-family:monospace">{loc['zip']}</span>
          </td>
          <td style="padding:12px 16px; border-bottom:1px solid #1e2230; text-align:center">
            <span style="background:{color}22; color:{color}; border:1px solid {color}44;
                         padding:3px 10px; border-radius:20px; font-size:12px; font-family:monospace">
              {emoji} {tier.upper()}
            </span>
          </td>
          <td style="padding:12px 16px; border-bottom:1px solid #1e2230; font-family:monospace;
                     font-size:13px; color:{color}; text-align:center">
            {impact_str}
          </td>
          <td style="padding:12px 16px; border-bottom:1px solid #1e2230; font-size:12px; color:#a0a8c0">
            {wx['temp_min']:.0f}°–{wx['temp_max']:.0f}°F
            · {wx['precip']:.2f}" rain
            {f"· {wx['snow']:.2f}\" snow" if wx['snow'] > 0.01 else ""}
          </td>
          <td style="padding:12px 16px; border-bottom:1px solid #1e2230; font-size:12px; color:#5a6080">
            {triggers_str}
          </td>
        </tr>
        """

    # Count summary
    mild_count = sum(1 for r in alerts if r["classification"]["tier"] == "mild")
    mod_count = sum(1 for r in alerts if r["classification"]["tier"] == "moderate")
    sev_count = len(severe)
    clear_count = len(results) - len(alerts)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width">
</head>
<body style="margin:0; padding:0; background:#0d0f14; font-family:'Helvetica Neue', Arial, sans-serif;">
  <div style="max-width:720px; margin:0 auto; padding:32px 16px;">

    <!-- Header -->
    <div style="display:flex; align-items:center; margin-bottom:32px;">
      <div style="background:#2ecc8a; width:40px; height:40px; border-radius:10px;
                  display:flex; align-items:center; justify-content:center;
                  font-size:20px; margin-right:14px; line-height:40px; text-align:center">🍀</div>
      <div>
        <div style="font-size:18px; font-weight:700; color:#e8eaf0">Clover Weather Monitor</div>
        <div style="font-size:11px; color:#5a6080; font-family:monospace; margin-top:2px">
          DAILY FORECAST ALERT · {send_label.upper()} · {tomorrow_str.upper()}
        </div>
      </div>
    </div>

    <!-- Summary chips -->
    <div style="display:flex; gap:10px; margin-bottom:28px; flex-wrap:wrap">
      <span style="background:#2ecc8a22; color:#2ecc8a; border:1px solid #2ecc8a44;
                   padding:5px 14px; border-radius:20px; font-size:12px; font-family:monospace">
        ✅ Clear: {clear_count}
      </span>
      <span style="background:#f5c84222; color:#f5c842; border:1px solid #f5c84244;
                   padding:5px 14px; border-radius:20px; font-size:12px; font-family:monospace">
        🟡 Mild: {mild_count}
      </span>
      <span style="background:#f58a1f22; color:#f58a1f; border:1px solid #f58a1f44;
                   padding:5px 14px; border-radius:20px; font-size:12px; font-family:monospace">
        🟠 Moderate: {mod_count}
      </span>
      <span style="background:#e8445a22; color:#e8445a; border:1px solid #e8445a44;
                   padding:5px 14px; border-radius:20px; font-size:12px; font-family:monospace">
        🔴 Severe: {sev_count}
      </span>
    </div>

    <!-- Location table -->
    <div style="background:#13161e; border:1px solid #252a38; border-radius:12px; overflow:hidden; margin-bottom:28px">
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr style="background:#1a1e2a">
            <th style="padding:10px 16px; text-align:left; font-size:10px; color:#5a6080;
                       font-family:monospace; letter-spacing:0.5px; font-weight:500">LOCATION</th>
            <th style="padding:10px 16px; text-align:center; font-size:10px; color:#5a6080;
                       font-family:monospace; letter-spacing:0.5px; font-weight:500">TIER</th>
            <th style="padding:10px 16px; text-align:center; font-size:10px; color:#5a6080;
                       font-family:monospace; letter-spacing:0.5px; font-weight:500">IMPACT</th>
            <th style="padding:10px 16px; text-align:left; font-size:10px; color:#5a6080;
                       font-family:monospace; letter-spacing:0.5px; font-weight:500">FORECAST</th>
            <th style="padding:10px 16px; text-align:left; font-size:10px; color:#5a6080;
                       font-family:monospace; letter-spacing:0.5px; font-weight:500">TRIGGERS</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <!-- Note -->
    <div style="background:#13161e; border:1px solid #252a38; border-radius:10px;
                padding:16px 20px; margin-bottom:28px; font-size:12px; color:#5a6080; line-height:1.6">
      <strong style="color:#a0a8c0">Note:</strong> Impact estimates are baseline projections.
      Actual sales variance depends on day-of-week, local events, and historical patterns for each location.
      Update <code style="color:#4a9eff">IMPACT_ESTIMATES</code> in <code style="color:#4a9eff">alert_engine.py</code>
      with your regression results for higher accuracy.
    </div>

    <!-- Footer -->
    <div style="border-top:1px solid #1e2230; padding-top:20px; font-size:10px;
                color:#3a4060; font-family:monospace; line-height:1.8">
      Weather data: Open-Meteo API · This alert sent daily at 7:00 AM ET<br>
      Clover Food Lab Operations Intelligence · Manage alerts in alert_engine.py
    </div>

  </div>
</body>
</html>
"""
    return subject, html


# ─── EMAIL SENDER ─────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    """Send via Gmail SMTP using App Password."""
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


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    hour = now.hour

    # Determine send type based on time of day:
    # 3 PM send (hour >= 14) = advance alert, looks at TOMORROW's weather
    # 7 AM send (hour < 14)  = day-of confirmation, looks at TODAY's weather
    if hour >= 14:
        target_day = now + timedelta(days=1)
        send_type = "advance"
    else:
        target_day = now
        send_type = "dayof"

    tomorrow_str = target_day.strftime("%A, %b %-d")

    print(f"🍀 Clover Weather Alert Engine — {tomorrow_str} ({'Advance' if send_type == 'advance' else 'Day-Of'})")
    print(f"   Checking {len(LOCATIONS)} locations...\n")

    day_index = 1 if send_type == "advance" else 0

    results = []
    for loc in LOCATIONS:
        try:
            weather = fetch_weather(loc["lat"], loc["lon"], day_index=day_index)
            classification = classify(weather, indoor=loc.get("indoor", False))
            results.append({
                "location": loc,
                "weather": weather,
                "classification": classification,
            })
            tier = classification["tier"]
            emoji = TIER_EMOJI[tier]
            indoor_tag = " [indoor]" if loc.get("indoor") else ""
            print(f"  {emoji} {loc['name']:32s} {tier.upper():10s} "
                  f"({weather['temp_avg']:.0f}°F, {weather['precip']:.2f}\" rain, "
                  f"{weather['snow']:.2f}\" snow){indoor_tag}")
        except Exception as e:
            print(f"  ❌ {loc['name']}: {e}")

    subject, html = build_email(results, tomorrow_str, send_type=send_type)
    print(f"\n  Subject: {subject}")
    send_email(subject, html)
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
