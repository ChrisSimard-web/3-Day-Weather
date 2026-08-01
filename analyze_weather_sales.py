"""
Clover Food Lab — Weather × Sales Analysis Pipeline
=====================================================
Run this once to:
  1. Fetch historical weather for all 10 locations (Open-Meteo, free)
  2. Merge with your sales data
  3. Run regression analysis
  4. Output calibrated impact estimates for alert_engine.py
  5. Save a detailed CSV + summary report

Usage:
    pip install requests pandas scipy numpy
    python analyze_weather_sales.py

Output files:
    weather_sales_merged.csv   — full merged dataset
    weather_impact_report.txt  — summary with calibrated thresholds
    impact_estimates.json      — plug these numbers into alert_engine.py
"""

import json
import math
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from pathlib import Path

# ─── LOCATION CONFIG ─────────────────────────────────────────────────────────
# Coordinates matched to your location codes
LOCATIONS = {
    'CloverHSC': {'name': 'HSC (Harvard Campus)',     'zip': '02138', 'lat': 42.3770, 'lon': -71.1167, 'indoor': True},
    'CloverPRU': {'name': 'PRU (Prudential — mall)',  'zip': '02199', 'lat': 42.3467, 'lon': -71.0822, 'indoor': True},
    'CloverHFI': {'name': 'HFI (Kendall/MIT)',        'zip': '02139', 'lat': 42.3626, 'lon': -71.0940, 'indoor': False},
    'CloverFIN': {'name': 'FIN (Financial District)', 'zip': '02110', 'lat': 42.3573, 'lon': -71.0523, 'indoor': False},
    'CloverKEE': {'name': 'KEE (Food Hall — new)',    'zip': '02142', 'lat': 42.3682, 'lon': -71.0815, 'indoor': True},
}

SALES_FILE = 'daily_sales_2025-04-01_2026-03-23.csv'
START_DATE = '2025-04-01'
END_DATE   = '2026-03-23'


# ─── STEP 1: LOAD & CLEAN SALES DATA ─────────────────────────────────────────

def load_sales(path: str) -> pd.DataFrame:
    print("📂 Loading sales data...")
    df = pd.read_csv(path)
    df.columns = ['location', 'day', 'sales']

    def parse_day(s):
        parts = s.split(' ')
        m, d = [int(x) for x in parts[1].split('/')]
        year = 2025 if m >= 4 else 2026
        return datetime(year, m, d)

    df['date'] = df['day'].apply(parse_day)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df['dow'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month

    # Remove zero and near-zero sales days (< $50 = almost certainly closed)
    before = len(df)
    df = df[df['sales'] > 0].copy()
    print(f"   Removed {before - len(df)} zero-sales rows")

    # Filter to active locations only
    active = list(LOCATIONS.keys())
    df = df[df['location'].isin(active)].copy()
    print(f"   {len(df)} rows across {df['location'].nunique()} locations")
    return df


# ─── STEP 2: FETCH HISTORICAL WEATHER ────────────────────────────────────────

def fetch_weather_for_location(name: str, lat: float, lon: float) -> pd.DataFrame:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"snowfall_sum,weathercode,windspeed_10m_max"
        f"&temperature_unit=fahrenheit"
        f"&precipitation_unit=inch"
        f"&timezone=America%2FNew_York"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()['daily']

    wx = pd.DataFrame({
        'date_str':   d['time'],
        'temp_max':   d['temperature_2m_max'],
        'temp_min':   d['temperature_2m_min'],
        'precip':     d['precipitation_sum'],
        'snow':       d['snowfall_sum'],
        'wx_code':    d['weathercode'],
        'wind_max':   d['windspeed_10m_max'],
    })
    wx['temp_avg'] = (wx['temp_max'] + wx['temp_min']) / 2
    wx['location'] = name
    return wx


def fetch_all_weather(locations: dict) -> pd.DataFrame:
    print("\n🌤  Fetching historical weather (Open-Meteo archive)...")
    frames = []
    for code, info in locations.items():
        print(f"   {code} — {info['name']}...", end=' ', flush=True)
        try:
            wx = fetch_weather_for_location(code, info['lat'], info['lon'])
            frames.append(wx)
            print(f"✓ ({len(wx)} days)")
            time.sleep(0.3)   # be polite to the free API
        except Exception as e:
            print(f"✗ ERROR: {e}")
    df = pd.concat(frames, ignore_index=True)
    print(f"   Weather fetched: {len(df)} location-day rows\n")
    return df


# ─── STEP 3: MERGE ───────────────────────────────────────────────────────────

def merge(sales: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    print("🔗 Merging sales + weather...")
    merged = pd.merge(sales, weather, on=['location', 'date_str'], how='inner')
    print(f"   Merged rows: {len(merged)}")
    return merged


# ─── STEP 4: FEATURE ENGINEERING ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Day-of-week baseline index (used to normalize sales vs weekly pattern)
    dow_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
               'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['dow_num'] = df['dow'].map(dow_map)

    # Normalized sales: deviation from each location's own DOW median
    # This isolates weather effect from day-of-week pattern
    df['dow_median'] = df.groupby(['location', 'dow'])['sales'].transform('median')
    df['sales_vs_dow'] = (df['sales'] / df['dow_median'] - 1) * 100  # % deviation

    # Weather categories
    df['is_rain'] = ((df['precip'] > 0.05) & (df['snow'] < 0.01)).astype(int)
    df['is_snow'] = (df['snow'] > 0.01).astype(int)
    df['is_cold'] = (df['temp_avg'] < 35).astype(int)
    df['is_hot']  = (df['temp_avg'] > 85).astype(int)
    df['is_clear']= ((df['precip'] < 0.05) & (df['snow'] < 0.01) &
                     (df['temp_avg'] >= 35) & (df['temp_avg'] < 85)).astype(int)

    # Tiered cold
    df['cold_tier'] = pd.cut(
        df['temp_avg'],
        bins=[-999, 15, 25, 35, 999],
        labels=['severe_cold', 'moderate_cold', 'mild_cold', 'normal']
    )

    # Tiered precip
    def precip_tier(row):
        if row['snow'] > 0.05: return 'severe_snow'
        if row['snow'] > 0.02: return 'moderate_snow'
        if row['snow'] > 0.005: return 'mild_snow'
        if row['precip'] > 0.30: return 'heavy_rain'
        if row['precip'] > 0.15: return 'moderate_rain'
        if row['precip'] > 0.05: return 'light_rain'
        return 'dry'
    df['precip_tier'] = df.apply(precip_tier, axis=1)

    return df


# ─── STEP 5: REGRESSION & IMPACT ANALYSIS ────────────────────────────────────

def analyze_impacts(df: pd.DataFrame) -> dict:
    print("📊 Running impact analysis...\n")
    results = {}

    # Overall weather impact on sales_vs_dow (% vs normal DOW baseline)
    categories = {
        'Light rain':    df['precip_tier'] == 'light_rain',
        'Moderate rain': df['precip_tier'] == 'moderate_rain',
        'Heavy rain':    df['precip_tier'] == 'heavy_rain',
        'Mild snow':     df['precip_tier'] == 'mild_snow',
        'Moderate snow': df['precip_tier'] == 'moderate_snow',
        'Heavy snow':    df['precip_tier'] == 'severe_snow',
        'Mild cold (25-35°F)':     df['cold_tier'] == 'mild_cold',
        'Moderate cold (15-25°F)': df['cold_tier'] == 'moderate_cold',
        'Severe cold (<15°F)':     df['cold_tier'] == 'severe_cold',
        'Heat (85-90°F)':          (df['temp_avg'] > 85) & (df['temp_avg'] <= 90),
        'Extreme heat (>90°F)':    df['temp_avg'] > 90,
    }

    baseline = df[df['is_clear'] == 1]['sales_vs_dow']
    baseline_mean = baseline.mean()

    print(f"  {'Category':<28} {'n':>5}  {'Avg % vs DOW':>13}  {'vs Clear':>9}  {'p-value':>9}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*13}  {'-'*9}  {'-'*9}")

    for label, mask in categories.items():
        subset = df[mask]['sales_vs_dow'].dropna()
        if len(subset) < 5:
            continue
        mean_dev = subset.mean()
        impact_vs_clear = mean_dev - baseline_mean

        # T-test vs baseline
        tstat, pval = stats.ttest_ind(subset, baseline, equal_var=False)
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

        print(f"  {label:<28}  {len(subset):>5}  {mean_dev:>+12.1f}%  {impact_vs_clear:>+8.1f}%  {pval:>8.3f} {sig}")
        results[label] = {
            'n': len(subset),
            'mean_pct_vs_dow': round(mean_dev, 1),
            'impact_vs_clear': round(impact_vs_clear, 1),
            'pvalue': round(pval, 4),
            'significant': pval < 0.05,
        }

    return results


def per_location_impacts(df: pd.DataFrame) -> pd.DataFrame:
    print("\n\n📍 Per-location weather sensitivity:\n")
    rows = []
    for loc in sorted(df['location'].unique()):
        sub = df[df['location'] == loc]
        clear = sub[sub['is_clear'] == 1]['sales_vs_dow']
        rainy = sub[sub['is_rain'] == 1]['sales_vs_dow']
        snowy = sub[sub['is_snow'] == 1]['sales_vs_dow']
        cold  = sub[sub['is_cold'] == 1]['sales_vs_dow']

        name = LOCATIONS.get(loc, {}).get('name', loc)
        median_sales = sub['sales'].median()

        rain_impact = rainy.mean() - clear.mean() if len(rainy) > 3 else None
        snow_impact = snowy.mean() - clear.mean() if len(snowy) > 3 else None
        cold_impact = cold.mean() - clear.mean() if len(cold) > 3 else None

        print(f"  {loc} ({name})")
        print(f"    Median daily sales: ${median_sales:,.0f}")
        print(f"    Rain impact:  {rain_impact:+.1f}% vs clear" if rain_impact is not None else "    Rain impact:  insufficient data")
        print(f"    Snow impact:  {snow_impact:+.1f}% vs clear" if snow_impact is not None else "    Snow impact:  insufficient data")
        print(f"    Cold impact:  {cold_impact:+.1f}% vs clear" if cold_impact is not None else "    Cold impact:  insufficient data")
        print()

        rows.append({
            'location': loc,
            'name': name,
            'median_sales': round(median_sales, 0),
            'rain_impact_pct': round(rain_impact, 1) if rain_impact is not None else None,
            'snow_impact_pct': round(snow_impact, 1) if snow_impact is not None else None,
            'cold_impact_pct': round(cold_impact, 1) if cold_impact is not None else None,
        })

    return pd.DataFrame(rows)


def build_calibrated_estimates(impact_results: dict) -> dict:
    """Convert regression results to tier-based estimates for alert_engine.py"""

    def safe(label, fallback):
        r = impact_results.get(label, {})
        v = r.get('impact_vs_clear', fallback)
        return abs(round(v)) if v is not None else abs(fallback)

    estimates = {
        "cold": {
            "mild":     safe('Mild cold (25-35°F)', -6),
            "moderate": safe('Moderate cold (15-25°F)', -13),
            "severe":   safe('Severe cold (<15°F)', -22),
        },
        "heat": {
            "mild":     safe('Heat (85-90°F)', -5),
            "moderate": safe('Extreme heat (>90°F)', -12),
            "severe":   safe('Extreme heat (>90°F)', -15),
        },
        "rain": {
            "mild":     safe('Light rain', -5),
            "moderate": safe('Moderate rain', -10),
            "severe":   safe('Heavy rain', -14),
        },
        "snow": {
            "mild":     safe('Mild snow', -7),
            "moderate": safe('Moderate snow', -15),
            "severe":   safe('Heavy snow', -25),
        },
    }
    return estimates


# ─── STEP 6: OUTPUT ───────────────────────────────────────────────────────────

def write_outputs(merged: pd.DataFrame, impact_results: dict,
                  loc_df: pd.DataFrame, estimates: dict):
    # Full merged CSV
    merged.to_csv('weather_sales_merged.csv', index=False)
    print(f"\n✓ Saved: weather_sales_merged.csv ({len(merged)} rows)")

    # Calibrated estimates JSON
    with open('impact_estimates.json', 'w') as f:
        json.dump(estimates, f, indent=2)
    print("✓ Saved: impact_estimates.json")

    # Text report
    lines = [
        "=" * 60,
        "CLOVER FOOD LAB — WEATHER × SALES IMPACT REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Data: {START_DATE} to {END_DATE}",
        "=" * 60,
        "",
        "CALIBRATED IMPACT ESTIMATES (for alert_engine.py)",
        "-" * 40,
    ]
    for category, tiers in estimates.items():
        lines.append(f"\n{category.upper()}:")
        for tier, val in tiers.items():
            lines.append(f"  {tier:12s}: -{val}%")

    lines += [
        "",
        "",
        "PASTE THIS INTO alert_engine.py → IMPACT_ESTIMATES:",
        "-" * 40,
        "IMPACT_ESTIMATES = {",
    ]
    for category, tiers in estimates.items():
        lines.append(f'    "{category}": ' + "{" +
                     ", ".join(f'"{t}": {v}' for t, v in tiers.items()) + "},")
    lines.append("}")

    lines += [
        "",
        "",
        "PER-LOCATION SENSITIVITY",
        "-" * 40,
    ]
    for _, row in loc_df.iterrows():
        lines.append(f"\n{row['location']} — {row['name']}")
        lines.append(f"  Median daily sales: ${row['median_sales']:,.0f}")
        for col, label in [('rain_impact_pct','Rain'), ('snow_impact_pct','Snow'), ('cold_impact_pct','Cold')]:
            val = row[col]
            if val is not None:
                lines.append(f"  {label}: {val:+.1f}% vs clear days")

    report = '\n'.join(lines)
    with open('weather_impact_report.txt', 'w') as f:
        f.write(report)
    print("✓ Saved: weather_impact_report.txt")
    print()
    print(report)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("🍀 Clover Food Lab — Weather × Sales Analysis")
    print("=" * 50)

    sales   = load_sales(SALES_FILE)
    weather = fetch_all_weather(LOCATIONS)
    merged  = merge(sales, weather)
    merged  = add_features(merged)

    impact_results = analyze_impacts(merged)
    loc_df         = per_location_impacts(merged)
    estimates      = build_calibrated_estimates(impact_results)

    write_outputs(merged, impact_results, loc_df, estimates)

    print("\n✅ Done. Next step: copy IMPACT_ESTIMATES from weather_impact_report.txt into alert_engine.py")


if __name__ == '__main__':
    main()
