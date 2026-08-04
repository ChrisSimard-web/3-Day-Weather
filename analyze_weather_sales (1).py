"""
Clover Food Lab — Weather × Sales Analysis Pipeline (v2)
=========================================================
Improvements over v1:
  - Seasonal index: normalizes sales by location × week-of-year
    so weather impact is measured vs the expected seasonal baseline
  - Hourly precipitation: identifies whether rain falls in the
    critical lunch window (9am–1pm) and quantifies timing effect
  - Outputs timing multipliers alongside tier impact estimates

Usage:
    pip install requests pandas scipy numpy
    python analyze_weather_sales.py

Input:
    daily_sales_2025-04-01_2026-03-23.csv  (same folder)

Output:
    weather_sales_merged.csv       full merged dataset
    weather_impact_report.txt      summary + paste-ready estimates
    impact_estimates.json          tier impacts for alert_engine.py
    timing_multipliers.json        lunch-window multipliers
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from pathlib import Path

# ── LOCATION CONFIG ──────────────────────────────────────────────────────────
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

LUNCH_START = 9   # 9am
LUNCH_END   = 14  # up to but not including 2pm (covers 9,10,11,12,13)

THRESHOLDS = {
    'precip_mild':     0.05,
    'precip_moderate': 0.15,
    'precip_severe':   0.30,
    'snow_mild':       0.005,
    'snow_moderate':   0.02,
    'snow_severe':     0.05,
    'cold_mild':       35,
    'cold_moderate':   25,
    'cold_severe':     15,
    'heat_mild':       85,
    'heat_severe':     90,
    'lunch_rain_heavy': 0.10,  # inches in lunch window = heavy lunch impact
    'lunch_rain_light': 0.02,  # inches in lunch window = light lunch impact
}


# ── STEP 1: LOAD & CLEAN SALES ───────────────────────────────────────────────

def load_sales(path: str) -> pd.DataFrame:
    print('📂 Loading sales data...')
    df = pd.read_csv(path)
    df.columns = ['location', 'day', 'sales']

    def parse_day(s):
        parts = s.split(' ')
        m, d = [int(x) for x in parts[1].split('/')]
        year = 2025 if m >= 4 else 2026
        return datetime(year, m, d)

    df['date']     = df['day'].apply(parse_day)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df['dow']      = df['date'].dt.day_name()
    df['week']     = df['date'].dt.isocalendar().week.astype(int)
    df['month']    = df['date'].dt.month

    before = len(df)
    df = df[df['sales'] > 0].copy()
    print(f'   Removed {before - len(df)} zero-sales rows')

    active = list(LOCATIONS.keys())
    df = df[df['location'].isin(active)].copy()
    print(f'   {len(df)} rows across {df["location"].nunique()} locations\n')
    return df


# ── STEP 2: FETCH DAILY + HOURLY WEATHER ────────────────────────────────────

def fetch_daily_weather(code: str, lat: float, lon: float) -> pd.DataFrame:
    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={lat}&longitude={lon}'
        f'&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,'
        f'snowfall_sum,weathercode,windspeed_10m_max'
        f'&temperature_unit=fahrenheit'
        f'&precipitation_unit=inch'
        f'&timezone=America%2FNew_York'
        f'&start_date={START_DATE}&end_date={END_DATE}'
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()['daily']
    df = pd.DataFrame({
        'date_str': d['time'],
        'temp_max': d['temperature_2m_max'],
        'temp_min': d['temperature_2m_min'],
        'precip':   d['precipitation_sum'],
        'snow':     d['snowfall_sum'],
        'wx_code':  d['weathercode'],
        'wind_max': d['windspeed_10m_max'],
    })
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
    df['location'] = code
    return df


def fetch_hourly_weather(code: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch hourly precipitation to detect lunch-window rain."""
    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={lat}&longitude={lon}'
        f'&hourly=precipitation'
        f'&precipitation_unit=inch'
        f'&timezone=America%2FNew_York'
        f'&start_date={START_DATE}&end_date={END_DATE}'
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()['hourly']

    hourly = pd.DataFrame({
        'datetime': pd.to_datetime(d['time']),
        'precip_hr': d['precipitation'],
    })
    hourly['date_str'] = hourly['datetime'].dt.strftime('%Y-%m-%d')
    hourly['hour']     = hourly['datetime'].dt.hour

    # Lunch window rain
    lunch = hourly[hourly['hour'].between(LUNCH_START, LUNCH_END - 1)]
    lunch_sum = lunch.groupby('date_str')['precip_hr'].sum().reset_index()
    lunch_sum.columns = ['date_str', 'lunch_precip']

    # Morning (pre-lunch) rain: 6–9am
    morning = hourly[hourly['hour'].between(6, 8)]
    morning_sum = morning.groupby('date_str')['precip_hr'].sum().reset_index()
    morning_sum.columns = ['date_str', 'morning_precip']

    merged = lunch_sum.merge(morning_sum, on='date_str', how='outer').fillna(0)
    merged['location'] = code
    return merged


def fetch_all_weather(locations: dict):
    print('🌤  Fetching historical weather (Open-Meteo archive)...')
    daily_frames   = []
    hourly_frames  = []

    for code, info in locations.items():
        print(f'   {code}...', end=' ', flush=True)
        try:
            daily  = fetch_daily_weather(code, info['lat'], info['lon'])
            hourly = fetch_hourly_weather(code, info['lat'], info['lon'])
            daily_frames.append(daily)
            hourly_frames.append(hourly)
            print(f'✓ ({len(daily)} days)')
            time.sleep(0.4)
        except Exception as e:
            print(f'✗ ERROR: {e}')

    daily_df  = pd.concat(daily_frames,  ignore_index=True)
    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    print(f'   Done: {len(daily_df)} daily rows, {len(hourly_df)} hourly-summary rows\n')
    return daily_df, hourly_df


# ── STEP 3: MERGE ────────────────────────────────────────────────────────────

def merge_all(sales, daily_wx, hourly_wx) -> pd.DataFrame:
    print('🔗 Merging sales + daily weather + hourly weather...')
    df = pd.merge(sales, daily_wx, on=['location', 'date_str'], how='inner')
    df = pd.merge(df, hourly_wx,  on=['location', 'date_str'], how='left')
    df['lunch_precip']   = df['lunch_precip'].fillna(0)
    df['morning_precip'] = df['morning_precip'].fillna(0)
    print(f'   Merged rows: {len(df)}\n')
    return df


# ── STEP 4: FEATURE ENGINEERING ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    # ── Seasonal index ───────────────────────────────────────────────────────
    # Step 1: DOW median per location
    df['dow_median'] = df.groupby(['location', 'dow'])['sales'].transform('median')

    # Step 2: Weekly median per location (smoothed with 3-week rolling window)
    # Use pivot to apply rolling across weeks, then map back
    weekly = (df.groupby(['location', 'week'])['sales']
                .median()
                .reset_index()
                .rename(columns={'sales': 'week_median'}))

    # Smooth each location's weekly median with rolling 3-week window
    smoothed = []
    for loc in weekly['location'].unique():
        sub = weekly[weekly['location'] == loc].copy().sort_values('week')
        sub['week_median_smooth'] = (sub['week_median']
                                     .rolling(3, center=True, min_periods=1)
                                     .mean())
        smoothed.append(sub)
    weekly_smooth = pd.concat(smoothed)

    df = df.merge(weekly_smooth[['location', 'week', 'week_median_smooth']],
                  on=['location', 'week'], how='left')

    # Step 3: Overall location median
    df['loc_median'] = df.groupby('location')['sales'].transform('median')

    # Step 4: Seasonal index = smoothed weekly median / overall location median
    df['seasonal_index'] = df['week_median_smooth'] / df['loc_median']

    # Step 5: Expected sales = DOW median × seasonal index
    df['expected_sales'] = df['dow_median'] * df['seasonal_index']

    # Step 6: Seasonally adjusted % deviation
    df['sales_vs_expected'] = (df['sales'] / df['expected_sales'] - 1) * 100

    # ── Rain timing classification ───────────────────────────────────────────
    def rain_timing(row):
        if row['precip'] < THRESHOLDS['precip_mild'] and row['snow'] < THRESHOLDS['snow_mild']:
            return 'dry'
        if row['snow'] > THRESHOLDS['snow_mild']:
            return 'snow'
        lp = row['lunch_precip']
        if lp >= THRESHOLDS['lunch_rain_heavy']:
            return 'lunch_heavy'
        if lp >= THRESHOLDS['lunch_rain_light']:
            return 'lunch_light'
        return 'off_peak'

    df['rain_timing'] = df.apply(rain_timing, axis=1)

    # ── Weather tiers ────────────────────────────────────────────────────────
    def precip_tier(row):
        if row['snow'] > THRESHOLDS['snow_severe']:   return 'severe_snow'
        if row['snow'] > THRESHOLDS['snow_moderate']: return 'moderate_snow'
        if row['snow'] > THRESHOLDS['snow_mild']:     return 'mild_snow'
        rain = max(0, row['precip'] - row['snow'])
        if rain > THRESHOLDS['precip_severe']:        return 'heavy_rain'
        if rain > THRESHOLDS['precip_moderate']:      return 'moderate_rain'
        if rain > THRESHOLDS['precip_mild']:          return 'light_rain'
        return 'dry'

    df['precip_tier'] = df.apply(precip_tier, axis=1)

    df['cold_tier'] = pd.cut(
        df['temp_avg'],
        bins=[-999, THRESHOLDS['cold_severe'], THRESHOLDS['cold_moderate'],
              THRESHOLDS['cold_mild'], 999],
        labels=['severe_cold', 'moderate_cold', 'mild_cold', 'normal']
    )

    df['is_clear'] = (
        (df['precip'] < THRESHOLDS['precip_mild']) &
        (df['snow']   < THRESHOLDS['snow_mild']) &
        (df['temp_avg'] >= THRESHOLDS['cold_mild']) &
        (df['temp_avg'] <  THRESHOLDS['heat_mild'])
    ).astype(int)

    df['is_rain'] = ((df['precip'] > THRESHOLDS['precip_mild']) & (df['snow'] < THRESHOLDS['snow_mild'])).astype(int)
    df['is_snow'] = (df['snow'] > THRESHOLDS['snow_mild']).astype(int)
    df['is_cold'] = (df['temp_avg'] < THRESHOLDS['cold_mild']).astype(int)

    return df


# ── STEP 5: REGRESSION ───────────────────────────────────────────────────────

def analyze_impacts(df: pd.DataFrame) -> dict:
    print('📊 Weather tier impact (seasonally adjusted):\n')

    baseline = df[df['is_clear'] == 1]['sales_vs_expected'].dropna()
    baseline_mean = baseline.mean()

    categories = {
        'Light rain':              df['precip_tier'] == 'light_rain',
        'Moderate rain':           df['precip_tier'] == 'moderate_rain',
        'Heavy rain':              df['precip_tier'] == 'heavy_rain',
        'Mild snow':               df['precip_tier'] == 'mild_snow',
        'Moderate snow':           df['precip_tier'] == 'moderate_snow',
        'Heavy snow':              df['precip_tier'] == 'severe_snow',
        'Mild cold (25–35°F)':     df['cold_tier']   == 'mild_cold',
        'Moderate cold (15–25°F)': df['cold_tier']   == 'moderate_cold',
        'Severe cold (<15°F)':     df['cold_tier']   == 'severe_cold',
        'Heat (85–90°F)':          (df['temp_avg'] > THRESHOLDS['heat_mild']) & (df['temp_avg'] <= THRESHOLDS['heat_severe']),
        'Extreme heat (>90°F)':    df['temp_avg'] > THRESHOLDS['heat_severe'],
    }

    print(f"  {'Category':<28} {'n':>5}  {'Seasonally adj %':>17}  {'vs Clear':>9}  {'p-val':>8}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*17}  {'-'*9}  {'-'*8}")

    results = {}
    for label, mask in categories.items():
        subset = df[mask]['sales_vs_expected'].dropna()
        if len(subset) < 5:
            continue
        mean_dev = subset.mean()
        impact   = mean_dev - baseline_mean
        _, pval  = stats.ttest_ind(subset, baseline, equal_var=False)
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {label:<28}  {len(subset):>5}  {mean_dev:>+16.1f}%  {impact:>+8.1f}%  {pval:>7.3f} {sig}")
        results[label] = {'n': len(subset), 'impact_vs_clear': round(impact, 1), 'pvalue': round(pval, 4)}

    return results


def analyze_timing(df: pd.DataFrame) -> dict:
    print('\n\n📊 Rain timing impact (lunch window vs off-peak):\n')

    rain_days = df[df['is_rain'] == 1].copy()
    baseline  = df[df['is_clear'] == 1]['sales_vs_expected'].dropna()

    timing_cats = {
        'Rain — lunch window (heavy)': rain_days['rain_timing'] == 'lunch_heavy',
        'Rain — lunch window (light)': rain_days['rain_timing'] == 'lunch_light',
        'Rain — off-peak only':        rain_days['rain_timing'] == 'off_peak',
    }

    print(f"  {'Timing':<32} {'n':>5}  {'Seasonally adj %':>17}  {'vs Clear':>9}  {'p-val':>8}")
    print(f"  {'-'*32}  {'-'*5}  {'-'*17}  {'-'*9}  {'-'*8}")

    timing_results = {}
    baseline_mean = baseline.mean()

    for label, mask in timing_cats.items():
        subset = rain_days[mask]['sales_vs_expected'].dropna()
        if len(subset) < 3:
            print(f"  {label:<32}  {len(subset):>5}  (insufficient data)")
            continue
        mean_dev = subset.mean()
        impact   = mean_dev - baseline_mean
        _, pval  = stats.ttest_ind(subset, baseline, equal_var=False)
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {label:<32}  {len(subset):>5}  {mean_dev:>+16.1f}%  {impact:>+8.1f}%  {pval:>7.3f} {sig}")
        timing_results[label] = {'n': len(subset), 'impact_vs_clear': round(impact, 1), 'pvalue': round(pval, 4)}

    return timing_results


def per_location_impacts(df: pd.DataFrame):
    print('\n\n📍 Per-location weather sensitivity (seasonally adjusted):\n')
    rows = []
    for loc in sorted(df['location'].unique()):
        sub   = df[df['location'] == loc]
        clear = sub[sub['is_clear'] == 1]['sales_vs_expected'].dropna()
        rain  = sub[sub['is_rain']  == 1]['sales_vs_expected'].dropna()
        snow  = sub[sub['is_snow']  == 1]['sales_vs_expected'].dropna()
        cold  = sub[sub['is_cold']  == 1]['sales_vs_expected'].dropna()

        lunch_rain = sub[sub['rain_timing'] == 'lunch_heavy']['sales_vs_expected'].dropna()
        offpk_rain = sub[sub['rain_timing'] == 'off_peak']['sales_vs_expected'].dropna()

        def impact(subset):
            return round(subset.mean() - clear.mean(), 1) if len(subset) > 3 else None

        name = LOCATIONS[loc]['name']
        median_sales = sub['sales'].median()

        print(f"  {loc} ({name})")
        print(f"    Median daily sales: ${median_sales:,.0f}")
        for label, val in [
            ('Rain (all)',        impact(rain)),
            ('Rain — lunch',      impact(lunch_rain)),
            ('Rain — off-peak',   impact(offpk_rain)),
            ('Snow',              impact(snow)),
            ('Cold',              impact(cold)),
        ]:
            if val is not None:
                print(f"    {label:<20}: {val:+.1f}% vs clear")
            else:
                print(f"    {label:<20}: insufficient data")
        print()

        rows.append({
            'location': loc, 'name': name,
            'median_sales': round(median_sales, 0),
            'rain_impact':       impact(rain),
            'rain_lunch_impact': impact(lunch_rain),
            'rain_offpk_impact': impact(offpk_rain),
            'snow_impact':       impact(snow),
            'cold_impact':       impact(cold),
        })
    return pd.DataFrame(rows)


# ── STEP 6: BUILD CALIBRATED ESTIMATES ───────────────────────────────────────

def build_estimates(impact_results: dict, timing_results: dict) -> tuple[dict, dict]:

    def safe(label, fallback):
        v = impact_results.get(label, {}).get('impact_vs_clear', fallback)
        return abs(round(v)) if v is not None else abs(fallback)

    estimates = {
        'cold': {
            'mild':     safe('Mild cold (25–35°F)', 6),
            'moderate': safe('Moderate cold (15–25°F)', 13),
            'severe':   safe('Severe cold (<15°F)', 22),
        },
        'heat': {
            'mild':     safe('Heat (85–90°F)', 5),
            'moderate': safe('Extreme heat (>90°F)', 12),
            'severe':   safe('Extreme heat (>90°F)', 15),
        },
        'rain': {
            'mild':     safe('Light rain', 5),
            'moderate': safe('Moderate rain', 10),
            'severe':   safe('Heavy rain', 14),
        },
        'snow': {
            'mild':     safe('Mild snow', 7),
            'moderate': safe('Moderate snow', 15),
            'severe':   safe('Heavy snow', 25),
        },
    }

    # Timing multipliers derived from regression
    # lunch_heavy / all_rain gives the relative multiplier
    lunch_impact  = timing_results.get('Rain — lunch window (heavy)', {}).get('impact_vs_clear')
    offpk_impact  = timing_results.get('Rain — off-peak only',        {}).get('impact_vs_clear')
    base_rain     = impact_results.get('Moderate rain',               {}).get('impact_vs_clear')

    if lunch_impact and base_rain and base_rain != 0:
        lunch_mult = round(abs(lunch_impact) / abs(base_rain), 2)
    else:
        lunch_mult = 1.5  # fallback

    if offpk_impact and base_rain and base_rain != 0:
        offpk_mult = round(abs(offpk_impact) / abs(base_rain), 2)
    else:
        offpk_mult = 0.6  # fallback

    timing_multipliers = {
        'lunch_heavy': lunch_mult,
        'lunch_light': round((lunch_mult + 1.0) / 2, 2),
        'off_peak':    offpk_mult,
        'dry':         1.0,
        'snow':        1.0,
    }

    return estimates, timing_multipliers


# ── STEP 7: OUTPUT ────────────────────────────────────────────────────────────

def write_outputs(merged, impact_results, timing_results, loc_df, estimates, timing_multipliers):
    merged.to_csv('weather_sales_merged.csv', index=False)
    print(f'\n✓ Saved: weather_sales_merged.csv ({len(merged)} rows)')

    with open('impact_estimates.json', 'w') as f:
        json.dump(estimates, f, indent=2)
    print('✓ Saved: impact_estimates.json')

    with open('timing_multipliers.json', 'w') as f:
        json.dump(timing_multipliers, f, indent=2)
    print('✓ Saved: timing_multipliers.json')

    lines = [
        '=' * 60,
        'CLOVER FOOD LAB — WEATHER × SALES IMPACT REPORT (v2)',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Data: {START_DATE} to {END_DATE}',
        'Method: Seasonally adjusted (DOW × weekly index)',
        '=' * 60,
        '',
        'PASTE INTO alert_engine.py → IMPACT_ESTIMATES:',
        '-' * 40,
        'IMPACT_ESTIMATES = {',
    ]
    for cat, tiers in estimates.items():
        lines.append(f'    "{cat}": ' + '{' + ', '.join(f'"{t}": {v}' for t, v in tiers.items()) + '},')
    lines.append('}')

    lines += [
        '',
        'PASTE INTO alert_engine.py → TIMING_MULTIPLIERS:',
        '-' * 40,
        'TIMING_MULTIPLIERS = {',
    ]
    for k, v in timing_multipliers.items():
        lines.append(f'    "{k}": {v},')
    lines.append('}')

    lines += ['', '', 'PER-LOCATION SENSITIVITY', '-' * 40]
    for _, row in loc_df.iterrows():
        lines.append(f"\n{row['location']} — {row['name']}")
        lines.append(f"  Median daily sales: ${row['median_sales']:,.0f}")
        for col, label in [
            ('rain_impact',       'Rain (all)'),
            ('rain_lunch_impact', 'Rain — lunch window'),
            ('rain_offpk_impact', 'Rain — off-peak'),
            ('snow_impact',       'Snow'),
            ('cold_impact',       'Cold'),
        ]:
            val = row[col]
            if val is not None:
                lines.append(f'  {label}: {val:+.1f}% vs clear')

    report = '\n'.join(lines)
    with open('weather_impact_report.txt', 'w') as f:
        f.write(report)
    print('✓ Saved: weather_impact_report.txt\n')
    print(report)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print('🍀 Clover Food Lab — Weather × Sales Analysis (v2)')
    print('=' * 52)

    sales              = load_sales(SALES_FILE)
    daily_wx, hourly_wx = fetch_all_weather(LOCATIONS)
    merged             = merge_all(sales, daily_wx, hourly_wx)
    merged             = add_features(merged)
    impact_results     = analyze_impacts(merged)
    timing_results     = analyze_timing(merged)
    loc_df             = per_location_impacts(merged)
    estimates, timing_multipliers = build_estimates(impact_results, timing_results)
    write_outputs(merged, impact_results, timing_results, loc_df, estimates, timing_multipliers)

    print('\n✅ Done.')
    print('   Copy IMPACT_ESTIMATES and TIMING_MULTIPLIERS from')
    print('   weather_impact_report.txt into alert_engine.py')


if __name__ == '__main__':
    main()
