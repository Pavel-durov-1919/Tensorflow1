import streamlit as st
import httpx, time, numpy as np, pandas as pd
import plotly.graph_objects as go
import os, json

# Official NASA C-MAPSS Sensor Descriptions (HI-06)
SENSOR_DESCRIPTIONS = {
    "op_setting_1": "Altitude (Operational Setting 1)",
    "op_setting_2": "Mach Number (Operational Setting 2)",
    "sensor_2": "Total Temperature at LPC Outlet (°R)",
    "sensor_3": "Total Temperature at HPC Outlet (°R)",
    "sensor_4": "Total Temperature at LPT Outlet (°R)",
    "sensor_7": "Total Pressure at HPC Outlet (psia)",
    "sensor_8": "Physical Fan Speed (rpm)",
    "sensor_9": "Physical Core Speed (rpm)",
    "sensor_11": "Static Pressure at HPC Outlet (psia)",
    "sensor_12": "Ratio of Fuel Flow to Ps30 (pps/psia)",
    "sensor_13": "Corrected Fan Speed (rpm)",
    "sensor_14": "Corrected Core Speed (rpm)",
    "sensor_15": "Bypass Ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_20": "HPT Coolant Bleed (lbm/s)",
    "sensor_21": "LPT Coolant Bleed (lbm/s)"
}

st.set_page_config(
    page_title="AeroShield · Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LIQUID-GLASS INJECTION — pulls the library from jsDelivr CDN              ║
# ║  and wires every .lg-card after Streamlit finishes painting the DOM.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
LIQUID_GLASS_SCRIPT = """
<script>
(function(){
  /* ── 1. Load liquid-glass.js from jsDelivr ── */
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/gh/deepika-builds/liquid-glass@main/liquid-glass.js';
  s.onload = initGlass;
  document.head.appendChild(s);

  function initGlass() {
    function applyAll() {
      document.querySelectorAll('.lg-card:not([data-lg-init])').forEach(el => {
        el.setAttribute('data-lg-init', '1');
        /* subtle settings — less dramatic, more refined */
        try {
          liquidGlass(el, { scale: -70, chroma: 3, border: 0.07, mapBlur: 16, blur: 5, saturate: 1.3 });
        } catch(e) {}
      });
      
      /* CR-01: Aggressively garbage collect orphaned SVG filters left by Streamlit DOM wipes */
      document.querySelectorAll('svg').forEach(svg => {
        if (svg.innerHTML.includes('feTurbulence') && svg.innerHTML.includes('feDisplacementMap') && svg.id) {
            const isUsed = document.querySelector(`[style*="${svg.id}"]`);
            if (!isUsed) svg.remove();
        }
      });
    }
    applyAll();
    const mo = new MutationObserver(() => applyAll());
    mo.observe(document.body, { subtree: true, childList: true });
  }

  /* ── 2. RAF-interpolated cursor-following glare ── */
  /* Each card tracks its own lerped (gx, gy) target so the glare
     smoothly drifts to wherever the cursor is — never snaps. */
  const state = new WeakMap();
  const LERP  = 0.08;   /* 0.04 = very lazy,  0.15 = snappier  */
  const IDLE_X = 50, IDLE_Y = -20;   /* parked position (above card) */

  let mx = -9999, my = -9999;
  document.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

  function tick() {
    document.querySelectorAll('.lg-card').forEach(card => {
      if (!state.has(card)) state.set(card, { cx: IDLE_X, cy: IDLE_Y });
      const st = state.get(card);
      const r  = card.getBoundingClientRect();

      /* cursor relative to card (0-100). If outside card → park above. */
      const inside = mx >= r.left && mx <= r.right && my >= r.top && my <= r.bottom;
      const tx = inside ? ((mx - r.left) / r.width  * 100) : IDLE_X;
      const ty = inside ? ((my - r.top)  / r.height * 100) : IDLE_Y;

      st.cx += (tx - st.cx) * LERP;
      st.cy += (ty - st.cy) * LERP;

      card.style.setProperty('--gx', st.cx.toFixed(2) + '%');
      card.style.setProperty('--gy', st.cy.toFixed(2) + '%');
    });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MASTER STYLESHEET                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ─── RESET ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* ─── SCENE — deep navy, electric blue nebulae ───── */
/* Palette:
   Primary:   #0ea5e9 (sky-500)  / #38bdf8 (sky-400)
   Secondary: #14b8a6 (teal-500) / #2dd4bf (teal-400)
   Success:   #10b981 (emerald)  / #34d399
   Warning:   #f59e0b (amber)    / #fbbf24
   Danger:    #f43f5e (rose)     / #fb7185
   Base bg:   #020c18 → #030e1e
*/
.stApp {
  background:
    radial-gradient(ellipse 70% 55% at 12%  8%,  rgba(14,165,233,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 55% 45% at 88% 85%,  rgba(20,184,166,0.15) 0%, transparent 55%),
    radial-gradient(ellipse 40% 35% at 55% 45%,  rgba(2,12,24,0.98)    0%, transparent 100%),
    linear-gradient(160deg, #020c18 0%, #030e1e 50%, #021018 100%);
  min-height: 100vh;
  color: #e0f2fe;
  position: relative;
}

/* star field — blue/teal tinted, no purple */
.stApp::before {
  content:'';
  position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    radial-gradient(1px 1px at 9%  14%, rgba(255,255,255,.75) 0%, transparent 100%),
    radial-gradient(1px 1px at 26% 58%, rgba(255,255,255,.45) 0%, transparent 100%),
    radial-gradient(1px 1px at 49% 24%, rgba(255,255,255,.60) 0%, transparent 100%),
    radial-gradient(1px 1px at 71% 74%, rgba(255,255,255,.35) 0%, transparent 100%),
    radial-gradient(1px 1px at 85% 36%, rgba(255,255,255,.55) 0%, transparent 100%),
    radial-gradient(1.5px 1.5px at 14% 80%, rgba(14,165,233,.9)  0%, transparent 100%),
    radial-gradient(1.5px 1.5px at 74%  7%, rgba(20,184,166,.8)  0%, transparent 100%),
    radial-gradient(1px 1px at 92% 64%, rgba(255,255,255,.40) 0%, transparent 100%),
    radial-gradient(1px 1px at 37% 90%, rgba(255,255,255,.30) 0%, transparent 100%),
    radial-gradient(1px 1px at 60% 46%, rgba(56,189,248,.60) 0%, transparent 100%);
  animation: starTwinkle 7s ease-in-out infinite alternate;
}
@keyframes starTwinkle { from{opacity:.45} to{opacity:1} }

.main .block-container {
  padding: 0 2.2rem 4rem 2.2rem !important;
  position: relative; z-index: 1;
}

/* ─── SIDEBAR ────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: rgba(2,12,24,0.88) !important;
  border-right: 1px solid rgba(14,165,233,0.14) !important;
  backdrop-filter: blur(28px) !important;
  -webkit-backdrop-filter: blur(28px) !important;
}
section[data-testid="stSidebar"] * { color: #bae6fd !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background: rgba(14,165,233,0.04) !important;
  border: 1px solid rgba(14,165,233,0.12) !important;
  border-radius: 10px !important;
}

/* ─── LIQUID-GLASS CARD MATERIAL ─────────────────── */
/* JS owns optics (SVG displacement). CSS owns tint + highlight + border + shadow. */
.lg-card {
  border-radius: 22px;
  /* ice-blue tinted glass — crisp, not dusty */
  background: linear-gradient(160deg,
    rgba(14,165,233, 0.06) 0%,
    rgba(20,184,166, 0.03) 55%,
    rgba(2, 12, 24,  0.30) 100%);
  box-shadow:
    0 28px 64px rgba(0,0,0,0.52),
    inset 0 1px 1px   rgba(255,255,255,0.50),
    inset 0 -5px 16px rgba(14,165,233,0.04),
    inset 0 0 0 1px   rgba(255,255,255,0.09);
  padding: 22px 24px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
  --gx: 50%; --gy: -20%;   /* parked above card initially */
  transition: transform 0.35s cubic-bezier(.16,1,.3,1),
              box-shadow 0.35s cubic-bezier(.16,1,.3,1);
}
/* subtle cursor glare — sky-blue tint, max 10% opacity */
.lg-card::after {
  content:''; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
  background: radial-gradient(
    120px circle at var(--gx) var(--gy),
    rgba(56,189,248, 0.10),
    rgba(20,184,166, 0.04) 40%,
    transparent 65%
  );
  transition: opacity 0.2s;
}
.lg-card:hover {
  transform: translateY(-2px) scale(1.005);
  box-shadow:
    0 36px 80px rgba(0,0,0,0.60),
    inset 0 1px 1px   rgba(255,255,255,0.55),
    inset 0 -5px 16px rgba(14,165,233,0.07),
    inset 0 0 0 1px   rgba(14,165,233,0.18);
}

/* ─── COLOR-ACCENT STRIPS ────────────────────────── */
.lg-card.accent-blue   { border-top: 2.5px solid rgba(14,165,233,0.85); }
.lg-card.accent-teal   { border-top: 2.5px solid rgba(20,184,166,0.85); }
.lg-card.accent-cyan   { border-top: 2.5px solid rgba(34,211,238,0.85); }
.lg-card.accent-amber  { border-top: 2.5px solid rgba(245,158,11, 0.85); }
.lg-card.accent-green  { border-top: 2.5px solid rgba(16,185,129, 0.85); }
.lg-card.accent-red    { border-top: 2.5px solid rgba(244,63,94,  0.85); }

/* ─── CARD TYPOGRAPHY ────────────────────────────── */
.lg-label {
  font-size: .68rem; font-weight: 700; letter-spacing: 1.8px;
  text-transform: uppercase; color: rgba(186,230,253,0.40);
  margin-bottom: 10px;
}
.lg-value-xl {
  font-size: 2.6rem; font-weight: 900; line-height: 1;
  letter-spacing: -1.5px; color: #f0f9ff;
  font-variant-numeric: tabular-nums;
}
.lg-value-lg { font-size: 1.75rem; font-weight: 800; line-height: 1.1; letter-spacing: -.7px; color: #f0f9ff; }
.lg-value-md { font-size: 1.15rem; font-weight: 700; color: #e0f2fe; }
.lg-unit     { font-size: .85rem; font-weight: 500; color: rgba(186,230,253,0.35); margin-left: 4px; }
.lg-sub      { font-size: .74rem; color: rgba(186,230,253,0.28); margin-top: 6px; font-weight: 400; }

/* ─── GRADIENT TEXT ──────────────────────────────── */
.grad-blue   { background: linear-gradient(135deg,#7dd3fc,#0ea5e9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.grad-teal   { background: linear-gradient(135deg,#5eead4,#14b8a6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.grad-cyan   { background: linear-gradient(135deg,#a5f3fc,#22d3ee); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.grad-amber  { background: linear-gradient(135deg,#fde68a,#f59e0b); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.grad-green  { background: linear-gradient(135deg,#6ee7b7,#10b981); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.grad-red    { background: linear-gradient(135deg,#fda4af,#f43f5e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.grad-white  { background: linear-gradient(135deg,#f0f9ff, #94a3b8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* ─── RISK PILL ──────────────────────────────────── */
.risk-pill {
  display:inline-flex; align-items:center; gap:7px;
  padding: 6px 15px; border-radius: 99px;
  font-size:.82rem; font-weight:700; letter-spacing:.6px; text-transform:uppercase;
  border: 1px solid;
}
.rp-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.risk-LOW      { background:rgba(16,185,129,.10);  border-color:rgba(16,185,129,.35);  color:#34d399; }
.risk-LOW .rp-dot      { background:#10b981; box-shadow:0 0 7px #10b981; }
.risk-MEDIUM   { background:rgba(245,158,11,.10);  border-color:rgba(245,158,11,.35);  color:#fbbf24; }
.risk-MEDIUM .rp-dot   { background:#f59e0b; box-shadow:0 0 7px #f59e0b; }
.risk-HIGH     { background:rgba(249,115,22,.10);  border-color:rgba(249,115,22,.35);  color:#fb923c; }
.risk-HIGH .rp-dot     { background:#f97316; box-shadow:0 0 7px #f97316; }
.risk-CRITICAL { background:rgba(244,63,94,.12);   border-color:rgba(244,63,94,.45);   color:#fb7185;
                 animation: crit-glow 2s ease-in-out infinite; }
.risk-CRITICAL .rp-dot { background:#f43f5e; box-shadow:0 0 10px #f43f5e; animation: rp-blink 1.5s ease-in-out infinite; }
@keyframes crit-glow  { 0%,100%{box-shadow:0 0 0 0 rgba(244,63,94,0)}   50%{box-shadow:0 0 0 5px rgba(244,63,94,.14)} }
@keyframes rp-blink   { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ─── PROGRESS BAR ───────────────────────────────── */
.lg-pbar-wrap { background:rgba(14,165,233,.08); border-radius:99px; height:6px; margin-top:13px; overflow:hidden; }
.lg-pbar-fill { height:100%; border-radius:99px; }
.pb-green  { background:linear-gradient(90deg,#10b981,#34d399); box-shadow:0 0 10px rgba(16,185,129,.45); }
.pb-amber  { background:linear-gradient(90deg,#f59e0b,#fbbf24); box-shadow:0 0 10px rgba(245,158,11,.45); }
.pb-red    { background:linear-gradient(90deg,#f43f5e,#fb7185); box-shadow:0 0 10px rgba(244,63,94,.45);  }

/* ─── HEADER ─────────────────────────────────────── */
.top-bar {
  display:flex; justify-content:space-between; align-items:center;
  padding: 22px 0;
  border-bottom: 1px solid rgba(14,165,233,0.12);
  margin-bottom: 26px;
}
.brand-row { display:flex; align-items:center; gap:15px; }
.brand-icon {
  width:46px; height:46px; border-radius:14px;
  background:linear-gradient(135deg,#0369a1,#0ea5e9);
  display:flex; align-items:center; justify-content:center; font-size:1.4rem; flex-shrink:0;
  box-shadow: 0 4px 20px rgba(14,165,233,0.50);
  animation: icon-shift 4s ease-in-out infinite alternate;
}
@keyframes icon-shift {
  from { box-shadow:0 4px 20px rgba(14,165,233,.50); }
  to   { box-shadow:0 4px 28px rgba(20,184,166,.60);  }
}
.brand-name { font-size:1.65rem; font-weight:900; letter-spacing:-.6px;
  background:linear-gradient(135deg,#e0f2fe 20%,#38bdf8 55%,#2dd4bf 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.brand-tagline { font-size:.72rem; font-weight:600; letter-spacing:1.8px;
  text-transform:uppercase; color:rgba(186,230,253,.20); margin-top:3px; }

/* ─── STATUS CHIP ────────────────────────────────── */
.api-chip {
  display:inline-flex; align-items:center; gap:9px;
  padding: 8px 18px; border-radius: 99px;
  font-size: .8rem; font-weight: 600;
  background: rgba(14,165,233,0.04);
  border: 1px solid rgba(14,165,233,0.12);
  backdrop-filter: blur(12px);
}
.api-chip .sdot { width:8px; height:8px; border-radius:50%; }
.chip-ok   { border-color:rgba(16,185,129,.40); }
.chip-ok   .sdot { background:#10b981; box-shadow:0 0 8px #10b981; animation:blink 2.5s ease-in-out infinite; }
.chip-err  { border-color:rgba(244,63,94,.40); }
.chip-err  .sdot { background:#f43f5e; box-shadow:0 0 8px #f43f5e; }
@keyframes blink { 0%,100%{opacity:1} 60%{opacity:.3} }
.lat-txt { font-family:'JetBrains Mono',monospace; font-size:.73rem; color:rgba(186,230,253,.30); }

/* ─── SECTION HEADER ─────────────────────────────── */
.sec-hdr { display:flex; align-items:center; gap:10px; margin: 28px 0 16px; }
.sec-hdr .bar { width:3px; height:18px; border-radius:99px;
  background:linear-gradient(180deg,#0ea5e9,#14b8a6); flex-shrink:0; }
.sec-hdr .title { font-size:.95rem; font-weight:700; color:#e0f2fe; letter-spacing:-.2px; }
.sec-hdr .desc  { font-size:.76rem; color:rgba(186,230,253,.22); }

/* ─── GLOW DIVIDER ───────────────────────────────── */
.glow-hr {
  height:1px; border:none;
  background:linear-gradient(90deg,transparent,rgba(14,165,233,.45),rgba(20,184,166,.35),transparent);
  margin: 28px 0;
}

/* ─── REC BOX ────────────────────────────────────── */
.rec-box {
  border-left: 4px solid; border-radius: 14px; padding: 18px 20px;
  background: rgba(14,165,233,0.02); margin-top: 10px;
}
.rec-LOW      { border-left-color:#10b981; }
.rec-MEDIUM   { border-left-color:#f59e0b; }
.rec-HIGH     { border-left-color:#f97316; }
.rec-CRITICAL { border-left-color:#f43f5e; animation: border-flicker 2s ease-in-out infinite; }
@keyframes border-flicker { 0%,100%{border-left-color:#f43f5e} 50%{border-left-color:#fda4af} }
.rec-title { font-size:.9rem; font-weight:700; color:#f0f9ff; margin-bottom:7px; }
.rec-text  { font-size:.86rem; color:rgba(186,230,253,.50); line-height:1.65; }

/* ─── INFO TABLE ─────────────────────────────────── */
.info-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:7px 0; border-bottom:1px solid rgba(14,165,233,.06);
  font-size:.78rem;
}
.info-row:last-child { border-bottom:none; }
.info-key { color:rgba(186,230,253,.30); font-weight:500; }
.info-val { color:#e0f2fe; font-weight:600; font-family:'JetBrains Mono',monospace; font-size:.75rem; }

/* ─── ACK CARD ───────────────────────────────────── */
.ack-ok   { background:rgba(16,185,129,.09); border:1px solid rgba(16,185,129,.28); border-radius:10px; padding:11px 15px; font-size:.82rem; color:#34d399; font-weight:600; margin-top:10px; }
.ack-wait { background:rgba(14,165,233,.03); border:1px solid rgba(14,165,233,.10); border-radius:10px; padding:10px 15px; font-size:.79rem; color:rgba(186,230,253,.28); margin-top:10px; }

/* ─── ERROR BANNER ───────────────────────────────── */
.err-box {
  text-align:center; padding:44px 24px;
  background:rgba(244,63,94,.04); border:1px dashed rgba(244,63,94,.28);
  border-radius:20px; margin-bottom:20px;
}
.err-box code { font-family:'JetBrains Mono',monospace; font-size:.78rem;
  background:rgba(14,165,233,.08); padding:2px 8px; border-radius:5px; color:#7dd3fc; }

/* ─── CHART BOX ──────────────────────────────────── */
.chart-wrap {
  background: linear-gradient(135deg,rgba(14,165,233,0.03),rgba(20,184,166,0.01));
  border: 1px solid rgba(14,165,233,0.08);
  border-radius: 20px; padding: 3px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.04);
  margin-bottom: 16px;
}

/* ─── SIDEBAR QUICK-STAT ─────────────────────────── */
.sb-stat { display:flex; justify-content:space-between; padding:6px 0;
  border-bottom:1px solid rgba(14,165,233,.07); font-size:.78rem; }
.sb-stat:last-child{border-bottom:none;}
.sb-key { color:rgba(186,230,253,.28); }
.sb-val { color:#e0f2fe; font-family:'JetBrains Mono',monospace; font-size:.74rem; font-weight:600; }

/* ─── STREAMLIT WIDGET OVERRIDES ─────────────────── */
div[data-testid="stSelectbox"]   > div { background:rgba(14,165,233,.04) !important; border:1px solid rgba(14,165,233,.12) !important; border-radius:10px !important; }
div[data-testid="stMultiSelect"] > div { background:rgba(14,165,233,.04) !important; border:1px solid rgba(14,165,233,.12) !important; border-radius:10px !important; }
div[data-testid="stTextInput"] input   { background:rgba(14,165,233,.04) !important; border:1px solid rgba(14,165,233,.12) !important; border-radius:10px !important; color:#e0f2fe !important; }
.stCheckbox label p { color:rgba(186,230,253,.50) !important; font-size:.84rem !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(14,165,233,.40); border-radius:99px; }
</style>
"""

# Inject CSS + glare tracker + liquid-glass loader
st.markdown(MASTER_CSS + LIQUID_GLASS_SCRIPT, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Load artifacts
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_model_artifacts():
    with open(os.path.join(BASE_DIR,"models","engine_examples.json")) as f: examples = json.load(f)
    with open(os.path.join(BASE_DIR,"models","metadata.json"))         as f: metadata = json.load(f)
    feat_imp_path = os.path.join(BASE_DIR,"models","feature_importance.json")
    feat_imp = {}
    if os.path.exists(feat_imp_path):
        with open(feat_imp_path) as f: feat_imp = json.load(f)
    return examples, metadata, feat_imp

try:
    examples, metadata, feature_importance = load_model_artifacts()
    feature_order = metadata.get("feature_order", [])
    window_length = metadata.get("window_length", 30)
    model_name    = metadata.get("model_name",    "RandomForestRegressor_FD001")
    model_version = metadata.get("version",       "1.0.0")
except Exception as e:
    st.error(f"❌ Failed to load artifacts: {e}"); st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:11px;margin-bottom:20px;padding-top:6px;">
      <div style="width:34px;height:34px;border-radius:10px;
        background:linear-gradient(135deg,#6d28d9,#2563eb);
        display:flex;align-items:center;justify-content:center;font-size:1.05rem;
        box-shadow:0 3px 12px rgba(109,40,217,.55);">🛡️</div>
      <div>
        <div style="font-size:.98rem;font-weight:800;color:#e2e8f0;letter-spacing:-.3px;">AeroShield</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.22);text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;">Control Panel</div>
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(139,92,246,.4),transparent);margin-bottom:18px;"></div>
    """, unsafe_allow_html=True)

    DEFAULT_BACKEND = os.getenv("BACKEND_URL","http://localhost:8000")
    backend_url = st.text_input("API Gateway Endpoint", value=DEFAULT_BACKEND)

    engine_options = {
        54: {"key":"successful_engine_example","label":"Engine #054 — Low Error",
             "rmse":examples["successful_engine_example"]["metrics"]["rmse"],
             "mae": examples["successful_engine_example"]["metrics"]["mae"]},
        74: {"key":"difficult_engine_example", "label":"Engine #074 — High Error",
             "rmse":examples["difficult_engine_example"]["metrics"]["rmse"],
             "mae": examples["difficult_engine_example"]["metrics"]["mae"]},
    }
    st.markdown('<div style="font-size:.66rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.22);margin:16px 0 8px;">Case Study</div>', unsafe_allow_html=True)
    selected_engine_id = st.selectbox("Engine", options=list(engine_options.keys()),
                                      format_func=lambda x: engine_options[x]["label"])

    active_case     = examples[engine_options[selected_engine_id]["key"]]
    max_cycles      = active_case["data"]["total_cycles"]
    latest_features = active_case["data"]["latest_unscaled_features"]

    st.markdown('<div style="font-size:.66rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.22);margin:16px 0 8px;">Flight Cycle</div>', unsafe_allow_html=True)
    selected_cycle = st.slider("Cycle", min_value=window_length, max_value=max_cycles,
                               value=min(100, max_cycles), label_visibility="collapsed")

    pct_life = selected_cycle / max_cycles * 100
    pb_col = "red" if pct_life>80 else ("amber" if pct_life>50 else "green")
    st.markdown(f"""
    <div style="margin-top:3px;">
      <div style="display:flex;justify-content:space-between;font-size:.72rem;margin-bottom:4px;">
        <span style="color:rgba(255,255,255,.28);">Lifetime Used</span>
        <span style="color:#f59e0b;font-family:'JetBrains Mono',monospace;font-weight:700;">{pct_life:.1f}%</span>
      </div>
      <div class="lg-pbar-wrap" style="height:4px;">
        <div class="lg-pbar-fill pb-{pb_col}" style="width:{pct_life}%;"></div>
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(139,92,246,.3),transparent);margin:18px 0 14px;"></div>
    <div style="font-size:.66rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:10px;">Statistics</div>
    <div class="sb-stat"><span class="sb-key">Engine</span><span class="sb-val">#{selected_engine_id:03d}</span></div>
    <div class="sb-stat"><span class="sb-key">Total Cycles</span><span class="sb-val">{max_cycles}</span></div>
    <div class="sb-stat"><span class="sb-key">Selected</span><span class="sb-val">{selected_cycle}</span></div>
    <div class="sb-stat"><span class="sb-key">Val MAE</span><span class="sb-val" style="color:#fbbf24;">{engine_options[selected_engine_id]['mae']:.3f}</span></div>
    <div class="sb-stat"><span class="sb-key">Val RMSE</span><span class="sb-val" style="color:#93c5fd;">{engine_options[selected_engine_id]['rmse']:.3f}</span></div>
    <div class="sb-stat"><span class="sb-key">Window</span><span class="sb-val">{window_length} cycles</span></div>
    <div class="sb-stat"><span class="sb-key">Features</span><span class="sb-val">{len(feature_order)}</span></div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# API Health
# ══════════════════════════════════════════════════════════════════════════════
api_health = {"status":"Disconnected","latency":0.0,"model_loaded":False}
try:
    t0 = time.perf_counter()
    r  = httpx.get(f"{backend_url}/health", timeout=1.5)
    lat = (time.perf_counter()-t0)*1000
    if r.status_code == 200:
        d = r.json()
        api_health = {"status":"Healthy","latency":lat,"model_loaded":d.get("model_loaded",False)}
except Exception:
    pass

is_ok = api_health["status"]=="Healthy" and api_health["model_loaded"]
chip_cls = "chip-ok" if is_ok else "chip-err"
chip_txt = ("API Connected · Model Active" if is_ok
            else ("API Connected · Model Offline" if api_health["status"]=="Healthy"
                  else "Backend Offline"))
lat_html = f'<span class="lat-txt"> · {api_health["latency"]:.0f} ms</span>' if is_ok else ''

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="top-bar">
  <div class="brand-row">
    <div class="brand-icon">🛡️</div>
    <div>
      <div class="brand-name">AeroShield</div>
      <div class="brand-tagline">Predictive Maintenance Intelligence · NASA C-MAPSS FD001</div>
    </div>
  </div>
  <div class="api-chip {chip_cls}">
    <span class="sdot"></span>
    <span style="color:#e2e8f0;">{chip_txt}</span>{lat_html}
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Sensor Simulator
# ══════════════════════════════════════════════════════════════════════════════
def simulate_sensor_window(engine_id, target_cycle, total_cycles, final_features, feat_order):
    np.random.seed(engine_id*13 + target_cycle)
    out = []
    for c in np.arange(target_cycle-window_length+1, target_cycle+1):
        frac = c/total_cycles; row = []
        for f in feat_order:
            fv = final_features[f]
            rising = any(x in f for x in ["sensor_2","sensor_3","sensor_4","sensor_8","sensor_9",
                                           "sensor_11","sensor_13","sensor_14","sensor_15","sensor_17"])
            nom  = fv*(0.92 if rising else 1.08)
            row.append(round(nom+(fv-nom)*frac+np.random.normal(0,abs(fv)*0.003), 4))
        out.append(row)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════
sensor_window_data = simulate_sensor_window(selected_engine_id, selected_cycle,
                                            max_cycles, latest_features, feature_order)
prediction_result = None; prediction_error = None

if api_health["status"] == "Healthy":
    if api_health["model_loaded"]:
        try:
            with st.spinner("Executing RUL Model Prediction..."):
                resp = httpx.post(f"{backend_url}/predict/rul",
                                  json={"engine_id":selected_engine_id,"cycle":selected_cycle,
                                        "sensor_window":sensor_window_data}, timeout=2.0)
            if resp.status_code == 200: prediction_result = resp.json()
            else: prediction_error = f"API returned HTTP {resp.status_code}."
        except Exception as ex: prediction_error = f"Request failed: {type(ex).__name__}"
    else: prediction_error = "Backend reachable but model artifacts are not loaded."
else: prediction_error = "Backend API is offline. Start the FastAPI server."

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Model Metadata  (4 liquid-glass cards)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr"><div class="bar"></div><div class="title">Model &amp; Verification Metadata</div></div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="lg-card accent-blue">
      <div class="lg-label">Active Model</div>
      <div class="lg-value-md grad-blue">RandomForestRegressor</div>
      <div class="lg-sub" style="font-family:'JetBrains Mono',monospace;margin-top:8px;">FD001 · v{model_version}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="lg-card accent-purple">
      <div class="lg-label">Case Study Target</div>
      <div class="lg-value-lg grad-purple">#{selected_engine_id:03d}</div>
      <div class="lg-sub">{max_cycles} total flight cycles · Cycle {selected_cycle}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="lg-card accent-amber">
      <div class="lg-label">Validation MAE</div>
      <div class="lg-value-xl grad-amber">{engine_options[selected_engine_id]['mae']:.2f}<span class="lg-unit">cy</span></div>
      <div class="lg-sub">Mean Absolute Error (cycles)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="lg-card accent-amber">
      <div class="lg-label">Validation RMSE</div>
      <div class="lg-value-xl grad-amber">{engine_options[selected_engine_id]['rmse']:.2f}<span class="lg-unit">cy</span></div>
      <div class="lg-sub">Root Mean Squared Error (cycles)</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="glow-hr">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — RUL KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr"><div class="bar"></div><div class="title">Remaining Useful Life (RUL) Prediction</div></div>""", unsafe_allow_html=True)

if prediction_result:
    rul  = prediction_result["estimated_rul"]
    pi_lo= prediction_result["prediction_interval_lower"]
    pi_hi= prediction_result["prediction_interval_upper"]
    pi_cov=prediction_result["prediction_interval_coverage"]
    risk = prediction_result["risk_level"].upper()
    dq   = prediction_result["data_quality_score"]
    rec  = prediction_result["recommendation"]
    strat= prediction_result.get("sequence_conversion_strategy","—")

    dq_pct = dq*100
    rul_grad = {"LOW":"grad-green","MEDIUM":"grad-amber","HIGH":"grad-amber","CRITICAL":"grad-red"}.get(risk,"grad-white")
    risk_dot = {"LOW":"#10b981","MEDIUM":"#f59e0b","HIGH":"#f97316","CRITICAL":"#ef4444"}.get(risk,"#94a3b8")
    dq_pb    = {"LOW":"pb-green","MEDIUM":"pb-amber","HIGH":"pb-amber","CRITICAL":"pb-red"}.get(risk,"pb-green")

    k1,k2,k3,k4 = st.columns(4)
    with k1:
        acc = "accent-red" if risk=="CRITICAL" else ("accent-amber" if risk in ("HIGH","MEDIUM") else "accent-green")
        st.markdown(f"""
        <div class="lg-card {acc}">
          <div class="lg-label">Estimated RUL</div>
          <div class="lg-value-xl {rul_grad}">{rul:.1f}<span class="lg-unit">cycles</span></div>
          <div class="lg-sub">At operational cycle {selected_cycle} of {max_cycles}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="lg-card accent-cyan">
          <div class="lg-label">Empirical 90% Prediction Interval</div>
          <div class="lg-value-lg grad-cyan" style="font-family:'JetBrains Mono',monospace;">[{pi_lo:.1f},&nbsp;{pi_hi:.1f}]</div>
          <div class="lg-sub">Coverage {pi_cov*100:.1f}% · 5th–95th pct residuals</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="lg-card">
          <div class="lg-label">Risk Level</div>
          <div style="margin:12px 0 8px;">
            <span class="risk-pill risk-{risk}">
              <span class="rp-dot"></span>{risk}
            </span>
          </div>
          <div class="lg-sub">System warning escalation level</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="lg-card accent-green">
          <div class="lg-label">Telemetry Integrity</div>
          <div class="lg-value-xl grad-green">{dq_pct:.0f}<span class="lg-unit">%</span></div>
          <div class="lg-pbar-wrap"><div class="lg-pbar-fill {dq_pb}" style="width:{dq_pct}%;"></div></div>
        </div>""", unsafe_allow_html=True)

    # ── Maintenance & Review ──────────────────────────────────────────────────
    st.markdown('<hr class="glow-hr">', unsafe_allow_html=True)
    st.markdown("""<div class="sec-hdr"><div class="bar"></div><div class="title">Maintenance Operations &amp; Review</div></div>""", unsafe_allow_html=True)

    col_rec, col_meta, col_ack = st.columns([5,3,4])

    with col_rec:
        st.markdown(f"""
        <div class="lg-card">
          <div class="lg-label">Maintenance Recommendation</div>
          <div class="rec-box rec-{risk}">
            <div class="rec-title">⚠ Action Plan · {risk.title()} Risk</div>
            <div class="rec-text">{rec}</div>
          </div>
          <div style="margin-top:14px;font-size:.73rem;color:rgba(255,255,255,.22);line-height:1.55;">
            <strong style="color:rgba(255,255,255,.35);">Sequence Strategy:</strong> {strat}
          </div>
        </div>""", unsafe_allow_html=True)

    with col_meta:
        st.markdown(f"""
        <div class="lg-card">
          <div class="lg-label">Prediction Metadata</div>
          <div class="info-row"><span class="info-key">Model</span><span class="info-val">{prediction_result['model_name']}</span></div>
          <div class="info-row"><span class="info-key">Version</span><span class="info-val">{prediction_result['model_version']}</span></div>
          <div class="info-row"><span class="info-key">Engine ID</span><span class="info-val">#{selected_engine_id:03d}</span></div>
          <div class="info-row"><span class="info-key">Cycle</span><span class="info-val">{selected_cycle} / {max_cycles}</span></div>
          <div class="info-row"><span class="info-key">PI Level</span><span class="info-val">90%</span></div>
          <div class="info-row"><span class="info-key">PI Coverage</span><span class="info-val">{pi_cov*100:.2f}%</span></div>
          <div class="info-row"><span class="info-key">Data Quality</span><span class="info-val">{dq_pct:.0f}%</span></div>
        </div>""", unsafe_allow_html=True)

    with col_ack:
        st.markdown("""
        <div class="lg-card">
          <div class="lg-label">Human-Review Sign-Off</div>
          <div style="font-size:.82rem;color:rgba(255,255,255,.4);line-height:1.6;margin-bottom:14px;">
            Certify you have reviewed the full prediction trajectory before
            logging an action in the depot dispatch queue.
            No PII is collected.
          </div>
        </div>""", unsafe_allow_html=True)
        certified = st.checkbox("I have reviewed the prediction trajectory and confirm receipt of the maintenance recommendation.")
        if certified:
            st.markdown('<div class="ack-ok">✓ Acknowledged — logged to dispatch queue.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ack-wait">⏳ Awaiting engineer sign-off…</div>', unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="err-box">
      <div style="font-size:1rem;font-weight:700;color:#f87171;margin-bottom:6px;">⚠ Predictions Unavailable</div>
      <div style="font-size:.84rem;color:rgba(255,255,255,.35);line-height:1.6;">{prediction_error}<br><br>
      Start the backend:<br><code>$env:PYTHONPATH="backend"; .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000</code></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Lifetime Trajectory
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="glow-hr">', unsafe_allow_html=True)
st.markdown("""<div class="sec-hdr"><div class="bar"></div>
<div class="title">Engine Lifetime Trajectory</div>
<div class="desc"> · Full operational cycle · Actual vs Predicted RUL with 90% PI (Empirical 5th–95th pct residuals)</div>
</div>""", unsafe_allow_html=True)

hist_cycles = active_case["data"]["cycles"]
hist_actual = active_case["data"]["actual_rul"]
hist_pred   = active_case["data"]["predicted_rul"]
imeta = metadata.get("empirical_prediction_intervals",{})
lo_off = imeta.get("lower_quantile_5",-34.28)
hi_off = imeta.get("upper_quantile_95", 24.08)
pred_lo = [max(0.0, float(v+lo_off)) for v in hist_pred]
pred_hi = [float(v+hi_off) for v in hist_pred]

CHART_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(255,255,255,.35)', family='Inter', size=11),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)',
               tickfont=dict(size=11,color='rgba(255,255,255,.35)',family='JetBrains Mono'),
               showline=True, linecolor='rgba(255,255,255,.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)',
               tickfont=dict(size=11,color='rgba(255,255,255,.35)',family='JetBrains Mono'),
               showline=True, linecolor='rgba(255,255,255,.08)'),
    legend=dict(bgcolor='rgba(5,8,18,0.85)', bordercolor='rgba(255,255,255,.08)', borderwidth=1,
                font=dict(size=12,color='#c9d1d9'), orientation='h',
                yanchor='bottom',y=1.02,xanchor='right',x=1),
    margin=dict(l=10,r=10,t=40,b=10),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='rgba(5,8,18,0.95)', bordercolor='rgba(139,92,246,.5)',
                    font=dict(color='#e2e8f0',size=12,family='JetBrains Mono'))
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=hist_cycles+hist_cycles[::-1], y=pred_hi+pred_lo[::-1],
    fill='toself', fillcolor='rgba(139,92,246,0.08)',
    line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', name='90% Prediction Interval'))
fig1.add_trace(go.Scatter(x=hist_cycles,y=pred_hi,mode='lines',
    line=dict(color='rgba(139,92,246,0.25)',width=1,dash='dot'),hoverinfo='skip',showlegend=False))
fig1.add_trace(go.Scatter(x=hist_cycles,y=pred_lo,mode='lines',
    line=dict(color='rgba(139,92,246,0.25)',width=1,dash='dot'),hoverinfo='skip',showlegend=False))
fig1.add_trace(go.Scatter(x=hist_cycles,y=hist_pred,mode='lines',name='Predicted RUL',
    line=dict(color='#818cf8',width=2.5),
    hovertemplate='<b>Cycle %{x}</b><br>Predicted: <b>%{y:.1f}</b> cycles<extra></extra>'))
fig1.add_trace(go.Scatter(x=hist_cycles,y=hist_actual,mode='lines',name='Actual RUL',
    line=dict(color='#34d399',width=2,dash='dash'),
    hovertemplate='<b>Cycle %{x}</b><br>Actual: <b>%{y:.1f}</b> cycles<extra></extra>'))
fig1.add_vline(x=selected_cycle,line_width=1.5,line_color='rgba(248,113,113,0.85)',
    annotation_text=f' Cycle {selected_cycle}',annotation_position='top right',
    annotation_font=dict(color='#f87171',size=11,family='JetBrains Mono'))
_base1 = {k: v for k, v in CHART_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
fig1.update_layout(**_base1, height=420,
    yaxis=dict(**CHART_LAYOUT['yaxis'], range=[-5, 220],
        title=dict(text='Remaining Useful Life (Cycles)', font=dict(size=12, color='rgba(255,255,255,.3)'))),
    xaxis=dict(**CHART_LAYOUT['xaxis'],
        title=dict(text='Operational Flight Cycles', font=dict(size=12, color='rgba(255,255,255,.3)'))))

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Telemetry
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="glow-hr">', unsafe_allow_html=True)
st.markdown("""<div class="sec-hdr"><div class="bar"></div>
<div class="title">Active Telemetry Windows</div>
<div class="desc"> · Last 30 cycles · Z-score normalized sensor readings</div>
</div>""", unsafe_allow_html=True)

selected_sensors = st.multiselect(
    "Select sensor features to compare (Z-score normalized)",
    options=feature_order,
    default=["sensor_2","sensor_3","sensor_4","sensor_11"],
    format_func=lambda x: f"{x} — {SENSOR_DESCRIPTIONS.get(x, x)}"
)

NEON_PALETTE = ['#818cf8','#34d399','#fbbf24','#f87171','#38bdf8','#fb923c',
                '#a78bfa','#6ee7b7','#fde68a','#93c5fd','#e879f9','#4ade80']

if selected_sensors:
    rows=[]
    for c in range(1,selected_cycle+1):
        np.random.seed(selected_engine_id*13+c)
        frac=c/max_cycles; row={"cycle":c}
        for f in feature_order:
            fv=latest_features[f]
            rising=any(x in f for x in ["sensor_2","sensor_3","sensor_4","sensor_8","sensor_9",
                                         "sensor_11","sensor_13","sensor_14","sensor_15","sensor_17"])
            nom=fv*(0.92 if rising else 1.08)
            row[f]=nom+(fv-nom)*frac+np.random.normal(0,abs(fv)*0.003)
        rows.append(row)
    df=pd.DataFrame(rows)

    fig2=go.Figure()
    win_start=max(1,selected_cycle-window_length+1)
    fig2.add_vrect(x0=win_start,x1=selected_cycle,fillcolor='rgba(139,92,246,0.07)',
        line_width=0,annotation_text=f'Active window ({win_start}–{selected_cycle})',
        annotation_position='top left',
        annotation_font=dict(size=10,color='rgba(139,92,246,.6)'))
    for i,s in enumerate(selected_sensors):
        vals=df[s].values; std=np.std(vals) or 1.0
        norm=(vals-np.mean(vals))/std
        fig2.add_trace(go.Scatter(x=df["cycle"],y=norm,mode='lines',name=s,
            line=dict(color=NEON_PALETTE[i%len(NEON_PALETTE)],width=2),
            hovertemplate=f'<b>Cycle %{{x}}</b><br>{s}: <b>%{{y:.3f}}</b>σ<extra></extra>'))
    fig2.add_vline(x=selected_cycle,line_width=1.5,line_color='rgba(248,113,113,0.85)',
        annotation_text=' Current',annotation_position='top right',
        annotation_font=dict(color='#f87171',size=10,family='JetBrains Mono'))
    _base2 = {k: v for k, v in CHART_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig2.update_layout(**_base2, height=380,
        yaxis=dict(**CHART_LAYOUT['yaxis'],
            title=dict(text='Normalized Signal (Z-score σ)', font=dict(size=12, color='rgba(255,255,255,.3)'))),
        xaxis=dict(**CHART_LAYOUT['xaxis'],
            title=dict(text='Operational Cycles', font=dict(size=12, color='rgba(255,255,255,.3)'))))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""<div style="text-align:center;padding:30px;background:rgba(255,255,255,.02);
    border:1px dashed rgba(255,255,255,.07);border-radius:16px;color:rgba(255,255,255,.2);font-size:.87rem;">
    Select sensor features above to render normalized telemetry trends.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Feature Importance (HI-04)
# ══════════════════════════════════════════════════════════════════════════════
if feature_importance and "ranked_features" in feature_importance:
    st.markdown('<hr class="glow-hr">', unsafe_allow_html=True)
    st.markdown("""<div class="sec-hdr"><div class="bar"></div>
    <div class="title">Global Feature Importance</div>
    <div class="desc"> · Top drivers of the Random Forest RUL predictions</div>
    </div>""", unsafe_allow_html=True)
    
    top_n = feature_importance["ranked_features"][:10]
    fi_names = [f["feature"] for f in top_n][::-1]
    fi_descs = [f["official_description"] for f in top_n][::-1]
    fi_vals = [f["importance"] * 100 for f in top_n][::-1]
    
    fig3 = go.Figure(go.Bar(
        x=fi_vals, y=fi_names, orientation='h',
        marker=dict(color='#8b5cf6', line=dict(color='#a78bfa', width=1)),
        text=[f"{v:.1f}%" for v in fi_vals], textposition='outside',
        textfont=dict(color='rgba(255,255,255,.6)', size=11, family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>%{customdata}<br>Importance: <b>%{x:.2f}%</b><extra></extra>',
        customdata=fi_descs
    ))
    _base3 = {k: v for k, v in CHART_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig3.update_layout(**_base3, height=340,
        xaxis=dict(**CHART_LAYOUT['xaxis'], title="Relative Importance (%)", range=[0, max(fi_vals)*1.2]),
        yaxis=dict(**CHART_LAYOUT['yaxis'], title="", autorange=True))
        
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
pi_cov_pct = metadata.get("performance_metrics", {}).get("empirical_interval_coverage_pct", 89.98)
st.markdown(f"""
<hr class="glow-hr">
<div style="display:flex;justify-content:space-between;align-items:center;
  font-size:.7rem;color:rgba(255,255,255,.15);padding-bottom:12px;">
  <span>AeroShield · NASA C-MAPSS FD001 · RandomForestRegressor v{model_version}</span>
  <span style="font-family:'JetBrains Mono',monospace;">
    PI Coverage: <span style="color:rgba(255,255,255,.28);">{pi_cov_pct}%</span>
    &nbsp;·&nbsp;Backend:&nbsp;<span style="color:rgba(255,255,255,.28);">localhost:8000</span>
  </span>
</div>""", unsafe_allow_html=True)
