"""
SolarFix — Offline celestial navigation using the sun.
No GPS. No satellites. Just the sun, a clock, and math that's worked for 500 years.
"""

import math
import numpy as np
import streamlit as st
import ephem
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime, timezone
from PIL import Image
import cv2
import io

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarFix · Celestial Navigation",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { color: #FFD700; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; }
  .metric-label { color: #aaa !important; }
  div[data-testid="metric-container"] > div:first-child { color: #aaa; }
  div[data-testid="metric-container"] > div:nth-child(2) { color: #FFD700; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CELESTIAL NAVIGATION ENGINE
# ─────────────────────────────────────────────────────────────

def get_sun_ephemeris(dt_utc: datetime) -> tuple[float, float]:
    """
    Returns (GHA, Declination) of the Sun in degrees for a given UTC datetime.
    GHA = Greenwich Hour Angle (0–360°, measured westward from Greenwich meridian).
    Dec = Declination (−23.5° to +23.5°, + = North).
    Uses the ephem library (same algorithms as nautical almanacs).
    """
    obs = ephem.Observer()
    obs.lon = "0:0:0"
    obs.lat = "0:0:0"
    obs.elevation = 0
    obs.pressure = 0  # disable atmospheric refraction for raw geometry
    obs.date = ephem.Date(dt_utc.strftime("%Y/%m/%d %H:%M:%S"))

    sun = ephem.Sun()
    sun.compute(obs)

    # GHA = Greenwich Mean Sidereal Time − Sun's Right Ascension
    gha = math.degrees(float(obs.sidereal_time())) - math.degrees(float(sun.g_ra))
    gha %= 360
    dec = math.degrees(float(sun.g_dec))
    return round(gha, 5), round(dec, 5)


def compute_hc_zn(lat: float, lon: float, gha: float, dec: float) -> tuple[float, float]:
    """
    St. Hilaire method: compute the sun's altitude (Hc) and azimuth (Zn)
    for an observer at (lat, lon) given ephemeris (gha, dec).
    This is the inverse of what a sextant measures — used to verify/compute.
    """
    lha = (gha + lon) % 360  # Local Hour Angle
    lat_r, dec_r, lha_r = math.radians(lat), math.radians(dec), math.radians(lha)

    sin_hc = (math.sin(lat_r) * math.sin(dec_r)
              + math.cos(lat_r) * math.cos(dec_r) * math.cos(lha_r))
    sin_hc = float(np.clip(sin_hc, -1.0, 1.0))
    hc = math.degrees(math.asin(sin_hc))

    denom = math.cos(math.radians(hc)) * math.cos(lat_r)
    if abs(denom) < 1e-9:
        return round(hc, 5), 0.0

    cos_z = (math.sin(dec_r) - math.sin(math.radians(hc)) * math.sin(lat_r)) / denom
    cos_z = float(np.clip(cos_z, -1.0, 1.0))
    z = math.degrees(math.acos(cos_z))
    zn = z if lha >= 180 else 360.0 - z  # true azimuth from north
    return round(hc, 5), round(zn % 360, 5)


def noon_sight_latitude(obs_alt: float, dt_utc: datetime, sun_north: bool = False) -> tuple[float, float]:
    """
    Classic noon sight: at solar noon the sun reaches its maximum altitude.
    Latitude = Declination + Zenith Distance  (sun south of observer, N hemisphere)
    Latitude = Declination − Zenith Distance  (sun north of observer)
    Returns (latitude, declination).
    """
    _, dec = get_sun_ephemeris(dt_utc)
    zd = 90.0 - obs_alt
    lat = (dec - zd) if sun_north else (dec + zd)
    return round(lat, 5), round(dec, 5)


def longitude_from_noon_time(noon_dt_utc: datetime) -> float:
    """
    At local apparent noon, the sun's GHA equals your longitude (west-positive).
    With east-positive convention: longitude = −GHA (normalised to ±180°).
    This is why an accurate clock was so revolutionary for navigation.
    """
    gha, _ = get_sun_ephemeris(noon_dt_utc)
    lon = ((180.0 - gha) % 360.0) - 180.0
    return round(lon, 5)


def circle_of_position(gha: float, dec: float, obs_alt: float, n: int = 720) -> tuple[list, list]:
    """
    Generate the Circle of Equal Altitude (CoP).
    Every point on Earth where the sun appears at angle obs_alt is equidistant
    from the sun's Geographical Position (GP). That distance = zenith distance.
    Returns (latitudes, longitudes) of the circle.
    """
    zd_r = math.radians(90.0 - obs_alt)

    # GP: the point on Earth directly below the sun
    gp_lat_r = math.radians(dec)
    gp_lon = ((180.0 - gha) % 360.0) - 180.0
    gp_lon_r = math.radians(gp_lon)

    lats, lons = [], []
    for i in range(n + 1):
        brg_r = math.radians(i * 360.0 / n)

        sin_pt_lat = (math.cos(zd_r) * math.sin(gp_lat_r)
                      + math.sin(zd_r) * math.cos(gp_lat_r) * math.cos(brg_r))
        sin_pt_lat = float(np.clip(sin_pt_lat, -1.0, 1.0))
        pt_lat = math.degrees(math.asin(sin_pt_lat))

        denom = math.cos(math.radians(pt_lat)) * math.cos(gp_lat_r)
        if abs(denom) < 1e-9:
            lats.append(pt_lat)
            lons.append(gp_lon)
            continue

        cos_dlon = (math.cos(zd_r) - math.sin(math.radians(pt_lat)) * math.sin(gp_lat_r)) / denom
        cos_dlon = float(np.clip(cos_dlon, -1.0, 1.0))
        dlon = math.degrees(math.acos(cos_dlon))
        if math.sin(brg_r) < 0:
            dlon = -dlon

        pt_lon = ((gp_lon + dlon + 180.0) % 360.0) - 180.0
        lats.append(pt_lat)
        lons.append(pt_lon)

    return lats, lons


def two_sight_fix(
    dt1: datetime, alt1: float, dt2: datetime, alt2: float,
    northern_hemisphere: bool = True,
) -> tuple[float, float, float]:
    """
    Numerically solve for the position that satisfies two altitude observations.
    Two circles of equal altitude intersect at TWO points on Earth (antipodal ambiguity).
    The hemisphere flag selects which solution to return — in practice navigators use
    their Dead Reckoning position to resolve this.
    Residual close to 0 means a confident fix.
    """
    gha1, dec1 = get_sun_ephemeris(dt1)
    gha2, dec2 = get_sun_ephemeris(dt2)

    def cost(pos):
        lat, lon = pos
        if abs(lat) > 90:
            return 1e10
        hc1, _ = compute_hc_zn(lat, lon, gha1, dec1)
        hc2, _ = compute_hc_zn(lat, lon, gha2, dec2)
        return (hc1 - alt1) ** 2 + (hc2 - alt2) ** 2

    # Search only in the indicated hemisphere to avoid antipodal ambiguity
    lat_seeds = [5, 30, 60, 80] if northern_hemisphere else [-5, -30, -60, -80]
    best, best_val = None, float("inf")
    for lat0 in lat_seeds:
        for lon0 in [-120, -60, 0, 60, 120]:
            res = minimize(cost, [lat0, lon0], method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 10000})
            if res.fun < best_val:
                best_val, best = res.fun, res

    return round(best.x[0], 4), round(best.x[1], 4), best_val


# ─────────────────────────────────────────────────────────────
# COMPUTER VISION
# ─────────────────────────────────────────────────────────────

def detect_sun(img_rgb: np.ndarray) -> tuple:
    """
    Detect the sun in a sky photo using brightness thresholding + blob analysis.
    Returns: (annotated_image, x_pct, y_pct, elevation_estimate) or None.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 238, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    h, w = img_rgb.shape[:2]
    x_pct = cx / w * 100
    y_pct = (1 - cy / h) * 100  # 0% = bottom (horizon), 100% = top (zenith)
    elevation_est = round(y_pct * 0.9, 1)  # rough: frame height ≈ 90° of sky

    result = img_rgb.copy()
    cv2.drawContours(result, [largest], -1, (255, 200, 0), 3)
    cv2.circle(result, (cx, cy), 40, (255, 140, 0), 3)
    cv2.line(result, (cx, h - 1), (cx, cy), (255, 140, 0), 2)
    cv2.putText(result, f"~{elevation_est}° elevation",
                (min(cx + 45, w - 200), max(cy - 15, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    return result, x_pct, y_pct, elevation_est


# ─────────────────────────────────────────────────────────────
# GLOBE VISUALISATION
# ─────────────────────────────────────────────────────────────

OCEAN = "#0a1628"
LAND  = "#1a2e45"
COAST = "#4a7fa5"
GRID  = "#1e3050"

def make_globe(
    fix_points=None,     # [(lat, lon, label, color), ...]
    circles=None,        # [(lats, lons, label, color), ...]
    sun_gp=None,         # (lat, lon)
    center=None,         # (lat, lon) to rotate globe toward
    title="",
    height=520,
) -> go.Figure:
    fig = go.Figure()

    if circles:
        for lats, lons, label, color in circles:
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=2.5, color=color),
                name=label, opacity=0.85,
            ))

    if sun_gp:
        fig.add_trace(go.Scattergeo(
            lat=[sun_gp[0]], lon=[sun_gp[1]],
            mode="markers+text",
            marker=dict(size=18, color="#FFD700", symbol="star",
                        line=dict(width=1.5, color="#FFA500")),
            text=["☀️ Sun GP"], textposition="top center",
            textfont=dict(color="#FFD700", size=12),
            name="Sun GP",
        ))

    if fix_points:
        for lat, lon, label, color in fix_points:
            fig.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon], mode="markers+text",
                marker=dict(size=14, color=color, symbol="circle",
                            line=dict(width=2, color="white")),
                text=[label], textposition="top center",
                textfont=dict(color=color, size=12),
                name=label,
            ))

    rot = dict(lon=center[1], lat=center[0]) if center else dict(lon=0, lat=20)

    fig.update_geos(
        showland=True, landcolor=LAND,
        showocean=True, oceancolor=OCEAN,
        showcoastlines=True, coastlinecolor=COAST,
        showframe=False, bgcolor="#0a0a1a",
        showlakes=True, lakecolor=OCEAN,
        showcountries=True, countrycolor=GRID,
        showrivers=False,
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lataxis_gridcolor=GRID, lonaxis_gridcolor=GRID,
        projection_type="orthographic",
        projection_rotation=rot,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFD700", size=15)),
        paper_bgcolor="#0a0a1a",
        height=height,
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(bgcolor="rgba(10,16,40,0.85)", font=dict(color="white"),
                    bordercolor=COAST, borderwidth=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def to_dms(deg: float, is_lat: bool) -> str:
    d = int(abs(deg))
    m_full = (abs(deg) - d) * 60
    m = int(m_full)
    s = (m_full - m) * 60
    dir_label = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    return f"{d}° {m}' {s:.1f}\" {dir_label}"


def compass_point(az: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[int((az + 11.25) / 22.5) % 16]


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
  <h1 style='font-size:3rem; margin-bottom:0;'>☀️ SolarFix</h1>
  <p style='color:#aaa; font-size:1.05rem; margin-top:0.2rem;'>
    Offline celestial navigation &nbsp;·&nbsp; No GPS &nbsp;·&nbsp; No satellites &nbsp;·&nbsp;
    Just the sun, a clock, and 500 years of math
  </p>
</div>
<hr style='border-color:#1e3050; margin-bottom:1.5rem;'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🌅  Noon Sight",
    "📐  Two-Sight Fix",
    "🌍  Sun Tracker",
    "📸  CV Sun Finder",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 · NOON SIGHT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Noon Sight — The Sailor's Classic Fix")
    st.markdown("""
    At **solar noon** the sun reaches its maximum altitude. Measure that angle with a sextant (or phone app),
    note the exact UTC time, and you have your **latitude and longitude** — the same method used by every
    navigator from Columbus to Cook.
    """)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### 📏 Your Noon Observations")
        obs_date  = st.date_input("Date (UTC)", value=datetime.utcnow().date())
        noon_time = st.time_input(
            "UTC time when sun was at its highest",
            value=datetime.utcnow().replace(hour=12, minute=0, second=0).time(),
        )
        max_alt = st.number_input(
            "Maximum sun altitude (° above horizon)",
            min_value=0.1, max_value=89.9, value=34.5, step=0.1,
        )
        sun_north = st.checkbox(
            "Sun was NORTH of me at noon",
            help="Only tick this if you're south of the sun's declination (e.g. tropics or S hemisphere)",
        )
        go_btn = st.button("🧭 Calculate My Position", type="primary", use_container_width=True)

    with c2:
        if go_btn:
            noon_dt = datetime.combine(obs_date, noon_time).replace(tzinfo=timezone.utc)
            lat, dec = noon_sight_latitude(max_alt, noon_dt, sun_north)
            lon      = longitude_from_noon_time(noon_dt)
            gha, _   = get_sun_ephemeris(noon_dt)
            gp_lat, gp_lon = dec, ((180.0 - gha) % 360.0) - 180.0

            m1, m2 = st.columns(2)
            m1.metric("Latitude",  to_dms(lat, True))
            m2.metric("Longitude", to_dms(lon, False))
            m1.metric("Decimal Lat", f"{lat:+.5f}°")
            m2.metric("Decimal Lon", f"{lon:+.5f}°")

            st.markdown("---")
            st.markdown("#### 🧮 The maths behind your fix")
            st.code(f"""
Solar Declination (δ)       = {dec:+.4f}°   ← where the sun is "above" on Earth today
Observed Altitude (h)       = {max_alt:.1f}°   ← what your sextant read
Zenith Distance (ZD)        = 90° − {max_alt:.1f}° = {90-max_alt:.1f}°

LATITUDE  = δ + ZD          = {dec:+.4f}° + {90-max_alt:.4f}° = {lat:+.5f}°

Sun's GHA at local noon     = {gha:.4f}°   ← how far west of Greenwich the sun was
LONGITUDE = −GHA            = {lon:+.5f}°  ← each 15° = 1 hour from UTC noon
""", language="text")

            fig = make_globe(
                fix_points=[(lat, lon, "📍 You", "#00FF88")],
                sun_gp=(gp_lat, gp_lon),
                center=((lat + gp_lat) / 2, (lon + gp_lon) / 2),
                title=f"Your noon sight fix · {obs_date}",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Enter your noon sight observations on the left and click **Calculate My Position**.")
            dt_now = datetime.utcnow().replace(tzinfo=timezone.utc)
            gha, dec = get_sun_ephemeris(dt_now)
            gp_lat, gp_lon = dec, ((180.0 - gha) % 360.0) - 180.0
            fig = make_globe(
                sun_gp=(gp_lat, gp_lon),
                center=(gp_lat, gp_lon),
                title=f"Sun's current GP · {dt_now.strftime('%H:%M:%S')} UTC",
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 · TWO-SIGHT FIX
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Two-Sight Fix — Full Position from Any Two Observations")
    st.markdown("""
    Take **two altitude readings** at different times (ideally 3+ hours apart).
    Each gives a **Circle of Equal Altitude** — every point on Earth where the sun appears at that angle.
    Where the two circles intersect: **that's you**.
    """)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### 👁 First Sight")
        s1d = st.date_input("Date", value=datetime.utcnow().date(), key="s1d")
        s1t = st.time_input("UTC Time",
                            value=datetime.utcnow().replace(hour=9, minute=0, second=0).time(),
                            key="s1t")
        s1a = st.number_input("Altitude (°)", 0.1, 89.9, 25.0, 0.1, key="s1a")

        st.markdown("#### 👁 Second Sight")
        s2d = st.date_input("Date", value=datetime.utcnow().date(), key="s2d")
        s2t = st.time_input("UTC Time",
                            value=datetime.utcnow().replace(hour=15, minute=0, second=0).time(),
                            key="s2t")
        s2a = st.number_input("Altitude (°)", 0.1, 89.9, 20.5, 0.1, key="s2a")

        hemi = st.radio("Your hemisphere", ["Northern", "Southern"], horizontal=True)
        fix_btn = st.button("🎯 Compute Fix", type="primary", use_container_width=True)

    with c2:
        if fix_btn:
            dt1 = datetime.combine(s1d, s1t).replace(tzinfo=timezone.utc)
            dt2 = datetime.combine(s2d, s2t).replace(tzinfo=timezone.utc)

            with st.spinner("Computing circles of position and numerical fix…"):
                gha1, dec1 = get_sun_ephemeris(dt1)
                gha2, dec2 = get_sun_ephemeris(dt2)

                cop1_lats, cop1_lons = circle_of_position(gha1, dec1, s1a)
                cop2_lats, cop2_lons = circle_of_position(gha2, dec2, s2a)

                fix_lat, fix_lon, residual = two_sight_fix(
                    dt1, s1a, dt2, s2a, northern_hemisphere=(hemi == "Northern")
                )

            m1, m2 = st.columns(2)
            m1.metric("Latitude",  to_dms(fix_lat, True))
            m2.metric("Longitude", to_dms(fix_lon, False))

            if residual < 0.01:
                st.success(f"✅ High-confidence fix — residual error: {residual:.6f}°²")
            elif residual < 0.5:
                st.warning(f"⚠️ Moderate confidence — residual: {residual:.4f}°² (check your altitudes)")
            else:
                st.error(f"❌ Low confidence — residual: {residual:.2f}°² (sights may be inconsistent)")

            gp1_lat, gp1_lon = dec1, ((180 - gha1) % 360) - 180
            gp2_lat, gp2_lon = dec2, ((180 - gha2) % 360) - 180

            fig = make_globe(
                fix_points=[(fix_lat, fix_lon, "📍 Your Fix", "#00FF88")],
                circles=[
                    (cop1_lats, cop1_lons, f"Circle 1 · {s1t.strftime('%H:%M')} UTC, alt={s1a}°", "#00BFFF"),
                    (cop2_lats, cop2_lons, f"Circle 2 · {s2t.strftime('%H:%M')} UTC, alt={s2a}°", "#FF6B35"),
                ],
                sun_gp=(gp2_lat, gp2_lon),
                center=(fix_lat, fix_lon),
                title="Two-sight fix · circles of equal altitude",
                height=560,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Enter two altitude observations on the left and click **Compute Fix**.")
            fig = make_globe(title="Enter observations to see your circles of position")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 · SUN TRACKER
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Sun Tracker — Verify Your Calculations")
    st.markdown(
        "Given a known position, compute exactly where the sun should be "
        "and plot its path across the sky for the whole day. Use this to plan your sights."
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### 📍 Your Location")
        t_lat  = st.number_input("Latitude",  -90.0,  90.0,  53.33, 0.01, key="tlat")
        t_lon  = st.number_input("Longitude", -180.0, 180.0, -6.25, 0.01, key="tlon")
        t_date = st.date_input("Date", value=datetime.utcnow().date(), key="tdate")
        use_now = st.checkbox("Use current UTC time", value=True)
        if not use_now:
            t_time = st.time_input("UTC Time", value=datetime.utcnow().time(), key="ttime")
        track_btn = st.button("🌞 Track the Sun", type="primary", use_container_width=True)

    with c2:
        if track_btn:
            if use_now:
                t_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
            else:
                t_dt = datetime.combine(t_date, t_time).replace(tzinfo=timezone.utc)

            gha, dec = get_sun_ephemeris(t_dt)
            alt, az  = compute_hc_zn(t_lat, t_lon, gha, dec)
            gp_lat   = dec
            gp_lon   = ((180.0 - gha) % 360.0) - 180.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Sun Altitude",    f"{alt:.2f}°")
            m2.metric("Sun Azimuth",     f"{az:.1f}° ({compass_point(az)})")
            m3.metric("Declination",     f"{dec:+.2f}°")

            if alt < 0:
                st.error("🌙 Sun is below the horizon right now.")
            elif alt < 8:
                st.warning(f"🌅 Sun is very low — look {compass_point(az)} ({az:.0f}°), altitude {alt:.1f}°")
            else:
                st.success(f"☀️ Look {compass_point(az)} ({az:.0f}°) — sun is {alt:.1f}° above the horizon")

            # Build full-day sun path (every 5 min)
            hours, alts, azs = [], [], []
            for minute in range(0, 24 * 60 + 1, 5):
                h, m_i = divmod(minute, 60)
                try:
                    dt_i = datetime(t_date.year, t_date.month, t_date.day,
                                    h % 24, m_i, 0, tzinfo=timezone.utc)
                    g_i, d_i = get_sun_ephemeris(dt_i)
                    a_i, z_i = compute_hc_zn(t_lat, t_lon, g_i, d_i)
                    hours.append(minute / 60)
                    alts.append(a_i)
                    azs.append(z_i)
                except Exception:
                    pass

            current_h = t_dt.hour + t_dt.minute / 60 + t_dt.second / 3600

            fig_path = go.Figure()
            fig_path.add_trace(go.Scatter(
                x=hours, y=alts, mode="lines",
                line=dict(color="#FFD700", width=2.5),
                fill="tozeroy", fillcolor="rgba(255,215,0,0.08)",
                name="Altitude",
            ))
            fig_path.add_vline(
                x=current_h, line_dash="dash", line_color="#FF6B35",
                annotation_text=f"Now ({t_dt.strftime('%H:%M')} UTC)",
                annotation_font_color="#FF6B35",
            )
            fig_path.add_hline(y=0, line_dash="dot", line_color=COAST)
            fig_path.update_layout(
                title=dict(text=f"Sun path · {t_date} · Lat {t_lat}° Lon {t_lon}°",
                           font=dict(color="#FFD700")),
                paper_bgcolor="#0a0a1a", plot_bgcolor="#0a1628",
                font=dict(color="white"),
                xaxis=dict(title="UTC Hour", color="white", gridcolor=GRID,
                           tickvals=list(range(0, 25, 3)),
                           ticktext=[f"{h:02d}:00" for h in range(0, 25, 3)]),
                yaxis=dict(title="Altitude (°)", color="white", gridcolor=GRID),
                height=300, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_path, use_container_width=True)

            fig = make_globe(
                fix_points=[(t_lat, t_lon, "📍 You", "#00FF88")],
                sun_gp=(gp_lat, gp_lon),
                center=((t_lat + gp_lat) / 2, (t_lon + gp_lon) / 2),
                title="Your position and the sun's Geographical Position (GP)",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Enter your location and click **Track the Sun**.")


# ══════════════════════════════════════════════════════════════
# TAB 4 · CV SUN FINDER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### CV Sun Finder — Computer Vision Prototype")
    st.markdown("""
    Upload a photo of the sky with the sun visible. The app detects the sun using OpenCV brightness
    analysis and estimates its elevation angle. This is the seed of your **AI Sextant** concept.
    """)

    uploaded = st.file_uploader("Upload a sky photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_arr = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(img, use_column_width=True)

        with col2:
            result = detect_sun(img_arr)
            if result:
                annotated, x_pct, y_pct, elev = result
                st.markdown("**Sun detected ☀️**")
                st.image(annotated, use_column_width=True)

                m1, m2 = st.columns(2)
                m1.metric("Est. Elevation", f"~{elev}°")
                m2.metric("Horizontal Pos", f"{x_pct:.0f}% from left")

                st.success(f"""
**Next steps for a full fix:**
1. Note the exact UTC time you took this photo
2. Use elevation `~{elev}°` as your altitude in the **Two-Sight Fix** tab
3. Take a second photo 3+ hours later and repeat
4. → Your position, no GPS required
                """)
            else:
                st.warning(
                    "No sun detected. Try a photo where the sun is clearly visible "
                    "and not behind thick cloud."
                )
    else:
        st.markdown("""
<div style='background:#0a1628; border:1px dashed #4a7fa5; border-radius:10px; padding:1.5rem; margin-top:1rem;'>
<h4 style='color:#FFD700; margin-top:0;'>📱 How the full product would work</h4>
<ol style='color:#ccc; line-height:1.8;'>
  <li><b>Point your phone at the sun</b> (solar filter recommended)</li>
  <li><b>CV detects the sun</b> in real-time from the camera feed</li>
  <li><b>Gyroscope gives elevation</b> — how high above horizon</li>
  <li><b>Magnetometer gives azimuth</b> — compass direction to sun</li>
  <li><b>Clock gives UTC time</b> — no network required, just accurate device time</li>
  <li><b>→ Instant lat/lon fix</b>, completely offline</li>
</ol>
<p style='color:#888; margin-bottom:0;'>
  Accuracy: ~10–30 nautical miles with basic sextant-level measurements.<br>
  Use case: offshore sailing, remote expeditions, EMP/disaster scenarios, education.
</p>
</div>
        """, unsafe_allow_html=True)
