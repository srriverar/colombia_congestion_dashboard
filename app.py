"""
Dashboard SIN Nodal — Análisis de Congestión: Sistema Eléctrico Colombiano
============================================================================
Análisis interactivo de congestión de transmisión en el Sistema
Interconectado Nacional (SIN) de Colombia.

Basado en Grid2Poster (open-energy-transition/grid2poster),
Awesome-Electrical-Grid-Mapping, y modelado LOPF con HiGHS.

Desarrollado por Prof. Sergio Rivera — Universidad Nacional de Colombia
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.ERROR)

def _check_solver():
    """The dispatch model only needs SciPy (HiGHS is bundled inside it).

    Deliberately NO PyPSA / geopandas / shapely: those pull a compiled GEOS
    dependency that breaks on interpreters without a matching wheel.
    """
    try:
        from scipy.optimize import linprog  # noqa: F401
        return True
    except Exception:
        return False


HAS_SOLVER = _check_solver()

st.set_page_config(page_title="SIN Nodal — Congestión Colombia",
                   page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DISCLAIMER = """<div style='background:#fff3cd;padding:8px 12px;border-radius:6px;
border-left:4px solid #ffc107;margin-bottom:10px;font-size:0.75em;'>
⚠️ <b>Plataforma Concept Test</b> — Desarrollada por <b>Prof. Sergio Rivera</b> 
(Universidad Nacional de Colombia). Datos de red: OpenStreetMap / Grid2Poster / 
Awesome-Electrical-Grid-Mapping. Modelos de congestión: LOPF + solver HiGHS (SciPy).</div>"""

# ═══════════════════════════════════════════════════════════════════════
#  COLOMBIAN DATA SOURCES (from Awesome-Electrical-Grid-Mapping)
# ═══════════════════════════════════════════════════════════════════════
CO_DATASETS = [
    {"name": "UPME - Plan de Expansión de Transmisión 2025-2039",
     "url": "https://www.minenergia.gov.co/documents/15647/Plan-expansion-2025-2039.pdf",
     "type": "Reporte", "year": "2025", "tags": "capacitydata, plan",
     "desc": "Plan oficial de expansión de transmisión. Incluye proyectos priorizados, corredores y cronogramas."},
    {"name": "Subestaciones construidas STN - STR",
     "url": "https://energia-upme.hub.arcgis.com/datasets/subestaciones-construidas-stn-str/explore",
     "type": "Dataset + Mapa", "year": "2024", "tags": "dataset, map, GIS",
     "desc": "Inventario geoespacial de todas las subestaciones del SIN (STN y STR). Formato ArcGIS."},
    {"name": "Transmisión de energía eléctrica - UPME",
     "url": "https://www.upme.gov.co/simec/energia-electrica/transmision-de-energia-electrica/",
     "type": "Reporte", "year": "2025", "tags": "report, statistics",
     "desc": "Estadísticas oficiales de transmisión: flujos, pérdidas, expansión."},
    {"name": "UPME - Líneas de transmisión construidas",
     "url": "https://www.arcgis.com/home/item.html?id=244214e5d10b47c1b12bfe9f7666594c",
     "type": "Dataset + Mapa", "year": "2026", "tags": "dataset, map, cc-by-2.0",
     "desc": "Capa GIS de todas las líneas de transmisión construidas en Colombia. Licencia CC-BY-2.0."},
    {"name": "EPM - Conductor primario, media tensión",
     "url": "https://services1.arcgis.com/lHSB5M3vxFT82S7P/ArcGIS/rest/services/P_Redes_Energia_GE3/FeatureServer/13",
     "type": "Dataset + Mapa", "year": "2025", "tags": "dataset, map, distribution",
     "desc": "Redes de media tensión de EPM (Antioquia). ArcGIS FeatureServer."},
    {"name": "EPM - Conductor secundario, baja tensión",
     "url": "https://services1.arcgis.com/lHSB5M3vxFT82S7P/ArcGIS/rest/services/P_Redes_Energia_GE3/FeatureServer/12",
     "type": "Dataset + Mapa", "year": "2025", "tags": "dataset, map, distribution",
     "desc": "Redes de baja tensión de EPM. Cobertura Antioquia."},
    {"name": "EPM - Conductor de transmisión",
     "url": "https://services1.arcgis.com/lHSB5M3vxFT82S7P/ArcGIS/rest/services/P_Redes_Energia_GE3/FeatureServer/17",
     "type": "Dataset + Mapa", "year": "2025", "tags": "dataset, map, transmission",
     "desc": "Líneas de transmisión de EPM en Antioquia. Incluye 230 kV y 500 kV."},
    {"name": "EPM - Subestaciones y transformadores",
     "url": "https://services1.arcgis.com/lHSB5M3vxFT82S7P/ArcGIS/rest/services/P_Redes_Energia_GE3/FeatureServer/14",
     "type": "Dataset + Mapa", "year": "2025", "tags": "dataset, map, substations",
     "desc": "Subestaciones y transformadores de EPM. Datos de capacidad y localización."},
    {"name": "EPM - Postes y torres de transmisión",
     "url": "https://services1.arcgis.com/lHSB5M3vxFT82S7P/ArcGIS/rest/services/P_Redes_Energia_GE3/FeatureServer/7",
     "type": "Dataset + Mapa", "year": "2025", "tags": "dataset, map, towers",
     "desc": "Inventario de postes y torres de transmisión de EPM."},
    {"name": "UPME SIG - Infraestructura eléctrica LATAM",
     "url": "https://sig.upme.gov.co/portal/apps/experiencebuilder/experience/?id=1cb0108edbcb48aca70a838db3db6d89",
     "type": "Mapa interactivo", "year": "2025", "tags": "map, LATAM, interactive",
     "desc": "Mapa interactivo de UPME con infraestructura eléctrica de Latinoamérica."},
    {"name": "XM — Operador del Sistema (estadísticas)",
     "url": "https://www.xm.com.co/",
     "type": "Portal de datos", "year": "2026", "tags": "statistics, operator, real-time",
     "desc": "Operador del SIN. Datos de demanda, generación, precios de bolsa, restricciones."},
    {"name": "CREG — Regulación de capacidad de transmisión",
     "url": "https://www.creg.gov.co/",
     "type": "Regulación", "year": "2025", "tags": "regulation, capacity, allocation",
     "desc": "Resolución 101 094: régimen transitorio de asignación de capacidad de transmisión."},
]

# SIN regions
REGIONS = {
    "La Guajira": {"lat": 11.5, "lon": -72.2, "type": "Hub Eólico", "cap_MW": 19.5,
                   "potential_GW": 21, "color": "#00CC66",
                   "desc": "Mayor recurso eólico de Suramérica (10.5 m/s promedio). 21 GW potencial. Bloqueado por Colectora."},
    "Cesar": {"lat": 10.0, "lon": -73.3, "type": "Nodo de Tránsito", "cap_MW": 450,
              "potential_GW": 2, "color": "#FF9900",
              "desc": "Nodo clave: Colectora termina en La Loma. Generación carbón + térmica."},
    "Atlántico": {"lat": 10.96, "lon": -74.8, "type": "Centro de Demanda", "cap_MW": 800,
                  "potential_GW": 0.5, "color": "#CC0000",
                  "desc": "Barranquilla. Centro de demanda costa Caribe. Importa de Cesar/Antioquia."},
    "Antioquia": {"lat": 6.5, "lon": -75.5, "type": "Hub Hidroeléctrico", "cap_MW": 4200,
                  "potential_GW": 3, "color": "#0066CC",
                  "desc": "Mayor cluster hidro: Hidroituango (2400 MW), San Carlos, Guatapé. Columna vertebral del SIN."},
    "Bogotá": {"lat": 4.6, "lon": -74.1, "type": "Centro de Demanda", "cap_MW": 1800,
               "potential_GW": 0.3, "color": "#CC3366",
               "desc": "Mayor centro de demanda (~25% nacional). Importa de Antioquia/Tolima."},
    "Medellín": {"lat": 6.25, "lon": -75.6, "type": "Centro de Demanda", "cap_MW": 900,
                 "potential_GW": 0.5, "color": "#9933CC",
                 "desc": "Segunda demanda. Cerca de hidro Antioquia pero congestión en temporada seca."},
    "Tolima-Huila": {"lat": 3.8, "lon": -75.2, "type": "Hidro + Tránsito", "cap_MW": 1500,
                     "potential_GW": 1, "color": "#0099CC",
                     "desc": "Hidro Saldaña/Prado. Corredor de tránsito Antioquia-Bogotá."},
    "Valle del Cauca": {"lat": 3.4, "lon": -76.5, "type": "Hidro + Demanda", "cap_MW": 1200,
                        "potential_GW": 1, "color": "#669900",
                       "desc": "Demanda Cali + hidro Salvajina/Alto Anchicayá. Costa Pacífica."},
    "Santander": {"lat": 7.1, "lon": -73.1, "type": "Térmica + Hidro", "cap_MW": 600,
                  "potential_GW": 0.5, "color": "#CC6600",
                  "desc": "Hidro Topocoro + térmica. Conexión entre corredores Caribe y Central."},
}

CORRIDORS = [
    {"name": "Colectora (La Guajira → Cesar)", "voltage": "500 kV", "km": 475,
     "capacity_MW": 1050, "status": "90% completa (Ago 2026)", "congestion": "CRÍTICA",
     "color": "#FF0000", "impact": "Bloquea 21 GW eólico. 7 parques eólicos varados.",
     "delay": "55+ meses. Originalmente 2022, ahora objetivo 2027."},
    {"name": "Cesar → Atlántico", "voltage": "230 kV", "km": 180,
     "capacity_MW": 600, "status": "Operativa (congestionada)", "congestion": "ALTA",
     "color": "#FF6600", "impact": "Limita importaciones Caribe. Se requiere respaldo térmico.",
     "delay": "Expansión planeada pero retrasada."},
    {"name": "Antioquia → Bogotá (500 kV)", "voltage": "500 kV", "km": 450,
     "capacity_MW": 2400, "status": "Operativa", "congestion": "MODERADA",
     "color": "#FFAA00", "impact": "Congestión en horas pico durante temporada seca (El Niño).",
     "delay": "Tercer circuito en estudio."},
    {"name": "Antioquia → Medellín", "voltage": "230 kV", "km": 80,
     "capacity_MW": 800, "status": "Operativa", "congestion": "MODERADA",
     "color": "#FFAA00", "impact": "Congestión durante puesta en marcha de Hidroituango.",
     "delay": "Mejoras locales en curso."},
    {"name": "Tolima → Bogotá", "voltage": "230 kV", "km": 150,
     "capacity_MW": 500, "status": "Operativa", "congestion": "ALTA",
     "color": "#FF6600", "impact": "Cuello de botella sur de Bogotá. Nueva 500 kV planeada.",
     "delay": "Licenciamiento ambiental pendiente."},
    {"name": "Valle → Pacífico", "voltage": "230 kV", "km": 200,
     "capacity_MW": 400, "status": "Operativa", "congestion": "BAJA",
     "color": "#00CC66", "impact": "Demanda limitada costa Pacífica.",
     "delay": "N/A"},
    {"name": "Santander → Caribe", "voltage": "230 kV", "km": 300,
     "capacity_MW": 500, "status": "Operativa", "congestion": "MODERADA",
     "color": "#FFAA00", "impact": "Capacidad de transferencia Norte-Sur limitada.",
     "delay": "Expansión en plan UPME."},
]

def mc(val, label, color="#003399", icon=""):
    return (f"<div style='background:linear-gradient(135deg,{color}12,{color}06);"
            f"padding:12px;border-radius:10px;text-align:center;border-left:4px solid {color};'>"
            f"<h4 style='color:{color};margin:0;font-size:1.05em;'>{icon} {val}</h4>"
            f"<p style='color:#555;margin:2px 0 0;font-size:0.7em;'>{label}</p></div>")

def section_header(title, desc):
    st.markdown(f"""<div style='background:#f0f4f8;padding:12px 16px;border-radius:8px;
                 border-left:5px solid #003399;margin-bottom:10px;'>
                 <b style='color:#003399;font-size:1.05em;'>{title}</b><br/>
                 <span style='color:#555;font-size:0.82em;'>{desc}</span></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  CONGESTION SIMULATOR (HiGHS LP + Weather + BESS + New Lines)
# ═══════════════════════════════════════════════════════════════════════

def _solve_dcopf(gen_specs, link_specs, stor_specs, nodes, demand_arr, T):
    """Single-shot DC-OPF as one linear program, solved with HiGHS (SciPy).

    Decision variables (per hour t):
        gen[k,t]        dispatch of generator k              [0, p_nom * pmax[t]]
        flow[k,t]       flow on link k measured at bus0      [-p_nom, +p_nom]
        chg[k,t]        BESS charge  (power taken from bus)  [0, p_nom]
        dis[k,t]        BESS discharge (power to bus)        [0, p_nom]
        soc[k,t]        BESS state of charge                 [0, max_hours * p_nom]

    Constraints
        nodal balance : gen + inflow*eff - outflow + dis - chg = demand   (per bus, hour)
        soc continuity: soc[t] = soc[t-1] + eff_store*chg[t] - dis[t]/eff_dispatch

    The duals of the nodal-balance rows are the nodal (locational marginal) prices.

    Returns (res, helpers) or (None, None) if the LP is infeasible.
    """
    from scipy.optimize import linprog
    from scipy import sparse

    G, L, S = len(gen_specs), len(link_specs), len(stor_specs)
    NV = (G + L + 3 * S) * T
    g0, l0 = 0, G * T
    c0, d0, s0 = (G + L) * T, (G + L + S) * T, (G + L + 2 * S) * T

    def gi(k, t): return g0 + k * T + t
    def li(k, t): return l0 + k * T + t
    def ci(k, t): return c0 + k * T + t
    def di(k, t): return d0 + k * T + t
    def si(k, t): return s0 + k * T + t

    # ---------- objective ----------
    cost = np.zeros(NV)
    for k, g in enumerate(gen_specs):
        cost[gi(k, 0):gi(k, 0) + T] = g["mc"]
    for k, s in enumerate(stor_specs):
        cost[di(k, 0):di(k, 0) + T] = s["mc"]

    # ---------- bounds ----------
    lo = np.zeros(NV)
    hi = np.zeros(NV)
    for k, g in enumerate(gen_specs):
        hi[gi(k, 0):gi(k, 0) + T] = np.maximum(g["p_nom"] * np.asarray(g["pmax"], dtype=float), 0.0)
    for k, l in enumerate(link_specs):
        # PyPSA's Link default is p_min_pu = 0 -> corridors are directional
        # (bus0 -> bus1).  Keeping them directional also avoids the
        # non-physical "loss in reverse" of a signed single-variable link.
        hi[li(k, 0):li(k, 0) + T] = l["p_nom"]
    for k, s in enumerate(stor_specs):
        hi[ci(k, 0):ci(k, 0) + T] = s["p_nom"]
        hi[di(k, 0):di(k, 0) + T] = s["p_nom"]
        hi[si(k, 0):si(k, 0) + T] = s["max_hours"] * s["p_nom"]

    # ---------- equality constraints ----------
    B = len(nodes)
    nrow = B * T + S * T
    A = sparse.lil_matrix((nrow, NV))
    beq = np.zeros(nrow)
    bix = {b: i for i, b in enumerate(nodes)}

    for k, g in enumerate(gen_specs):
        r = bix[g["bus"]] * T
        for t in range(T):
            A[r + t, gi(k, t)] += 1.0
    for k, l in enumerate(link_specs):
        r0, r1 = bix[l["bus0"]] * T, bix[l["bus1"]] * T
        for t in range(T):
            A[r0 + t, li(k, t)] -= 1.0
            A[r1 + t, li(k, t)] += l["eff"]
    for k, s in enumerate(stor_specs):
        r = bix[s["bus"]] * T
        for t in range(T):
            A[r + t, di(k, t)] += 1.0
            A[r + t, ci(k, t)] -= 1.0
    for b in nodes:
        beq[bix[b] * T: bix[b] * T + T] = demand_arr[b]

    for k, s in enumerate(stor_specs):
        r = B * T + k * T
        es, ed = s["eff_store"], s["eff_dispatch"]
        for t in range(T):
            A[r + t, si(k, t)] = 1.0
            A[r + t, ci(k, t)] = -es
            A[r + t, di(k, t)] = 1.0 / ed
            # state of charge starts empty (matches PyPSA 0.35 dispatch here)
            if t > 0:
                A[r + t, si(k, t - 1)] -= 1.0

    res = linprog(cost, A_eq=A.tocsr(), b_eq=beq,
                  bounds=np.column_stack([lo, hi]), method="highs")
    if not res.success:
        return None, None
    return res, (gi, li, ci, di, si)


def run_congestion_model(guajira_wind_MW=0, colectora_cap_MW=1050,
                          antioquia_hydro_MW=4200, bogota_demand_MW=2000,
                          caribe_demand_MW=1200, medellin_demand_MW=800,
                          gas_price=50, coal_price=40,
                          weather="Normal", bess_guajira_MW=0, bess_bogota_MW=0, bess_caribe_MW=0,
                          colectora2_cap_MW=0, tolima_bog_500_cap_MW=0,
                          n_hours=48, seed=42):
    """7-node Colombian SIN dispatch + congestion model.

    One linear program solved with HiGHS (bundled inside SciPy).  No PyPSA,
    geopandas or shapely involved, so the dependency footprint stays on
    pure-Python wheels and installs cleanly on any interpreter (>=3.9),
    including Python 3.14 on Streamlit Cloud.
    """
    if not HAS_SOLVER:
        return None

    T = int(n_hours)
    snapshots = pd.date_range("2025-01-01", periods=T, freq="h")

    rng = np.random.RandomState(seed)
    hour_of_day = np.arange(T) % 24
    demand_factor = 0.7 + 0.3 * np.sin(np.pi * (hour_of_day - 6) / 12)
    demand_factor = np.clip(demand_factor + rng.normal(0, 0.03, T), 0.5, 1.1)

    # ---- weather ----
    if weather == "El Niño":
        hydro_factor = np.clip(0.35 + rng.normal(0, 0.05, T), 0.15, 0.6)
        wind_cf = np.clip(rng.normal(0.65, 0.15, T), 0.1, 1.0)
        demand_factor = demand_factor * 1.15
    elif weather == "La Niña":
        hydro_factor = np.clip(0.85 + rng.normal(0, 0.05, T), 0.6, 1.0)
        wind_cf = np.clip(rng.normal(0.4, 0.15, T), 0.1, 0.8)
        demand_factor = demand_factor * 0.95
    else:
        hydro_factor = np.clip(0.6 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, T))
                               + rng.normal(0, 0.05, T), 0.3, 1.0)
        wind_cf = np.clip(rng.normal(0.55, 0.2, T), 0.1, 1.0)

    nodes = ["La Guajira", "Cesar", "Caribe", "Antioquia", "Medellin", "Bogota", "Tolima"]
    ones = np.ones(T)

    G = []
    def add_gen(name, bus, p_nom, mc, pmax=1.0):
        G.append({"name": name, "bus": bus, "p_nom": float(p_nom),
                  "pmax": ones * pmax if np.isscalar(pmax) else np.asarray(pmax, dtype=float),
                  "mc": float(mc)})

    if guajira_wind_MW > 0:
        add_gen("Wind_Guajira", "La Guajira", guajira_wind_MW, 5, wind_cf)
    add_gen("Coal_Cesar", "Cesar", 300, coal_price)
    add_gen("Thermal_Cesar", "Cesar", 200, gas_price + 10)
    add_gen("Gas_Caribe", "Caribe", 500, gas_price)
    add_gen("Diesel_Caribe", "Caribe", 200, 120)
    add_gen("Hydro_Antioquia", "Antioquia", antioquia_hydro_MW, 8, hydro_factor)
    add_gen("Gas_Antioquia", "Antioquia", 400, gas_price + 5)
    add_gen("Gas_Medellin", "Medellin", 300, gas_price + 8)
    add_gen("Hydro_Medellin", "Medellin", 200, 10)
    add_gen("Gas_Bogota", "Bogota", 800, gas_price + 15)
    add_gen("Diesel_Bogota", "Bogota", 300, 150)
    add_gen("Hydro_Tolima", "Tolima", 600, 12, hydro_factor)
    add_gen("Gas_Tolima", "Tolima", 300, gas_price + 10)
    for nd in nodes:
        add_gen(f"Unserved_{nd}", nd, 1e4, 10000)

    demand_arr = {b: np.zeros(T) for b in nodes}
    demand_arr["Caribe"] = caribe_demand_MW * demand_factor
    demand_arr["Medellin"] = medellin_demand_MW * demand_factor
    demand_arr["Bogota"] = bogota_demand_MW * demand_factor
    demand_arr["Tolima"] = 400 * demand_factor

    LK = []
    def add_link(name, b0, b1, p_nom, eff):
        LK.append({"name": name, "bus0": b0, "bus1": b1,
                   "p_nom": float(p_nom), "eff": float(eff)})

    add_link("Colectora", "La Guajira", "Cesar", max(colectora_cap_MW, 1e-6), 0.98)
    if colectora2_cap_MW > 0:
        add_link("Colectora_II", "La Guajira", "Cesar", colectora2_cap_MW, 0.98)
    add_link("Cesar_Caribe", "Cesar", "Caribe", 600, 0.97)
    add_link("Ant_Med", "Antioquia", "Medellin", 800, 0.98)
    add_link("Ant_Bog_500", "Antioquia", "Bogota", 2400, 0.97)
    add_link("Tol_Bog", "Tolima", "Bogota", 500, 0.97)
    if tolima_bog_500_cap_MW > 0:
        add_link("Tol_Bog_500", "Tolima", "Bogota", tolima_bog_500_cap_MW, 0.97)
    add_link("Ant_Tol", "Antioquia", "Tolima", 600, 0.97)
    add_link("Cesar_Ant", "Cesar", "Antioquia", 400, 0.96)

    ST = []
    def add_bess(name, bus, p_nom):
        if p_nom and p_nom > 0:
            ST.append({"name": name, "bus": bus, "p_nom": float(p_nom),
                       "max_hours": 4.0, "eff_store": 0.92,
                       "eff_dispatch": 0.92, "mc": 0.5})
    add_bess("BESS_Guajira", "La Guajira", bess_guajira_MW)
    add_bess("BESS_Bogota", "Bogota", bess_bogota_MW)
    add_bess("BESS_Caribe", "Caribe", bess_caribe_MW)

    res, H = _solve_dcopf(G, LK, ST, nodes, demand_arr, T)
    if res is None:
        return None
    gi, li, ci, di, si = H
    x = res.x

    gen_p = pd.DataFrame(
        {g["name"]: np.asarray([x[gi(k, t)] for t in range(T)]) for k, g in enumerate(G)},
        index=snapshots)
    link_p0 = pd.DataFrame(
        {l["name"]: np.asarray([x[li(k, t)] for t in range(T)]) for k, l in enumerate(LK)},
        index=snapshots)

    marg = np.asarray(res.eqlin.marginals)
    nodal_prices = pd.DataFrame(
        {b: marg[bix_t * T:(bix_t * T) + T] for bix_t, b in enumerate(nodes)},
        index=snapshots)

    congestion_data = {}
    for lk in LK:
        nm, b0, b1, cap = lk["name"], lk["bus0"], lk["bus1"], lk["p_nom"]
        flow = link_p0[nm].values
        pdiff = (nodal_prices[b1] - nodal_prices[b0]).abs().values
        cong_h = int(np.sum(pdiff > 1.0))
        congestion_data[nm] = {
            "bus0": b0, "bus1": b1, "capacity_MW": cap,
            "avg_flow_MW": float(np.mean(np.abs(flow))),
            "max_flow_MW": float(np.max(np.abs(flow))),
            "utilization_pct": float(np.mean(np.abs(flow)) / cap * 100) if cap > 0 else 0.0,
            "congestion_hours": cong_h,
            "congestion_pct": cong_h / T * 100,
            "avg_price_diff": float(np.mean(pdiff)),
            "max_price_diff": float(np.max(pdiff)),
            "congestion_cost_EUR": float(np.sum(np.abs(flow) * pdiff)),
        }

    unserved = {nd: float(gen_p[f"Unserved_{nd}"].sum()) for nd in nodes}

    bess_results = {}
    for k, s in enumerate(ST):
        bess_results[s["name"]] = {
            "charge": np.asarray([x[ci(k, t)] for t in range(T)]),
            "discharge": np.asarray([x[di(k, t)] for t in range(T)]),
            "soc": np.asarray([x[si(k, t)] for t in range(T)]),
        }

    return {
        "total_cost": float(res.fun), "gen_p": gen_p, "link_p0": link_p0,
        "nodal_prices": nodal_prices, "congestion": congestion_data,
        "unserved": unserved, "bess": bess_results,
        "n_hours": T, "demand_factor": demand_factor,
        "wind_cf": wind_cf, "hydro_cf": hydro_factor,
    }

# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:14px;
                border-radius:12px;color:white;text-align:center;margin-bottom:10px;'>
    <h2 style='margin:0;font-size:1.15em;'>🇨🇴 SIN Nodal</h2>
    <p style='margin:3px 0;font-size:0.68em;'>Congestión del Sistema Eléctrico Colombiano</p>
    <hr style='border-color:rgba(255,255,255,0.3);margin:5px 0;'/>
    <p style='font-size:0.58em;margin:0;'>
    Datos: OSM / Grid2Poster / Awesome-Grid-Mapping<br/>
    Motor: HiGHS LP (SciPy)<br/>
    Prof. Sergio Rivera — UNAL<br/>
    <b>Concept Test Platform</b>
    </p></div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠 Inicio",
        "🗺️ Mapa de Red",
        "⚡ Arquitectura SIN",
        "🔴 Corredores de Congestión",
        "💰 Precios Nodales",
        "🎮 Simulador de Congestión",
        "📊 Análisis de Escenarios",
        "🗄️ Datos Colombia",
        "📋 Política y Regulación",
    ]
    page = st.radio("Navegar:", pages, key="nav", label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                padding:28px;border-radius:16px;color:white;text-align:center;margin-bottom:16px;'>
    <h1 style='margin:0;font-size:2.1em;'>🇨🇴 Dashboard SIN Nodal</h1>
    <h3 style='margin:5px 0;opacity:0.95;'>Análisis de Congestión — Sistema Interconectado Nacional</h3>
    <hr style='border-color:rgba(255,255,255,0.3);margin:10px auto;width:50%;'/>
    <p style='font-size:0.78em;margin:0;'>
    Análisis interactivo del <b>SIN</b> con datos de OpenStreetMap, UPME, EPM y CREG<br/>
    Modelos de congestión con HiGHS LOPF | Prof. Sergio Rivera — UNAL
    </p></div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for c, (v, l, co, ic) in zip(cols, [
        ("18.4 GW", "Capacidad Instalada", "#003399", "⚡"),
        ("67.5%", "Dominancia Hidro", "#0066CC", "💧"),
        ("21 GW", "Potencial Eólico Guajira", "#00CC66", "💨"),
        ("1,050 MW", "Capacidad Colectora", "#CC0000", "🔌"),
        ("55 meses", "Retraso Promedio Transmisión", "#FF6600", "⏱️"),
    ]):
        c.markdown(mc(v, l, co, ic), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔴 La Crisis de Congestión
        - **La Guajira** tiene los vientos más fuertes de Suramérica (21 GW potencial)
        - Solo **19.5 MW** de capacidad eólica instalada
        - **Colectora** (500 kV, 475 km): retrasada **55+ meses**
        - **Moratoria** de conexiones desde 2024
        - **8.9 GW solar + 2.5 GW eólico** aprobados, **11 GW** pendientes
        - Sin modelar congestión: 60 GW necesarios al 2050
        - Con congestión: solo 50 GW — diferencia de 10 GW
        """)
    with col2:
        st.markdown("""
        ### 📊 Módulos del Dashboard
        | Módulo | Contenido |
        |--------|---------|
        | 🗺️ **Mapa de Red** | Visualización OSM de la red de transmisión |
        | ⚡ **Arquitectura SIN** | Matriz de generación, 9 regiones |
        | 🔴 **Corredores** | 7 corredores críticos con estado y capacidad |
        | 💰 **Precios Nodales** | Diferenciales de precio y costos de restricción |
        | 🎮 **Simulador** | Modelo de despacho 7 nodos con clima, BESS y nuevas líneas |
        | 📊 **Escenarios** | Con/sin Colectora, El Niño/La Niña, BESS, 2050 |
        | 🗄️ **Datos Colombia** | 12 datasets UPME, EPM, CREG, XM |
        | 📋 **Política** | Resolución UPME 358, reformas CREG |
        """)

    st.markdown("---")

    # ─── QUICK IMPACT CALCULATOR ───
    st.subheader("🎯 Calculadora Rápida de Impacto")
    st.markdown("*Ajuste los controles y vea el impacto instantáneo en el sistema:*")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        q_wind = st.slider("💨 Eólico Guajira (MW):", 0, 3000, 0, 100, key="qc_w",
                          help="Capacidad eólica instalada en La Guajira")
    with qc2:
        q_coll = st.slider("🔌 Colectora (MW):", 0, 2000, 1050, 50, key="qc_c",
                          help="Capacidad de la línea Colectora 500 kV")
    with qc3:
        q_bess = st.slider("🔋 BESS Total (MW):", 0, 1000, 0, 50, key="qc_b",
                          help="Almacenamiento en baterías distribuido")
    with qc4:
        q_wx = st.selectbox("🌤️ Clima:", ["Normal", "El Niño", "La Niña"], key="qc_wx",
                           help="El Niño: sequía hidro. La Niña: exceso hidro.")

    # Fast simplified impact calculation
    hydro_cf = {"Normal": 0.6, "El Niño": 0.35, "La Niña": 0.85}[q_wx]
    wind_cf = {"Normal": 0.55, "El Niño": 0.65, "La Niña": 0.40}[q_wx]
    demand_mult = {"Normal": 1.0, "El Niño": 1.15, "La Niña": 0.95}[q_wx]

    wind_gen = q_wind * wind_cf * 8760 / 1000  # GWh/year
    wind_displaced = min(wind_gen, 15000)  # max thermal displacement
    thermal_savings = wind_displaced * 50  # €/MWh avg thermal cost
    co2_reduction = wind_displaced * 0.5  # tons/MWh avg

    coll_benefit = min(q_coll, q_wind) * 0.85 * 8760 / 1000  # GWh evacuated
    bess_benefit = q_bess * 4 * 365 * 30  # €/year (peak shaving value)

    base_cost = 4500  # M€/year baseline congestion cost
    congestion_reduction = min(base_cost * 0.6, (coll_benefit * 0.3 + bess_benefit / 1e6 * 0.2 + wind_displaced * 0.1))

    unmet = max(0, 2000 * demand_mult - 4200 * hydro_cf - q_wind * wind_cf * 0.3 - q_coll * 0.5) * 8760 / 1000

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("⚡ Generación Eólica", f"{wind_gen:,.0f} GWh/año",
               delta=f"+{wind_gen:,.0f}" if q_wind > 0 else "Base")
    rc2.metric("💰 Ahorro Térmico", f"€{thermal_savings/1e6:.1f}M/año",
               delta=f"€{thermal_savings/1e6:.1f}M" if q_wind > 0 else "—")
    rc3.metric("🌱 Reducción CO₂", f"{co2_reduction/1000:.0f} kt/año",
               delta=f"-{co2_reduction/1000:.0f} kt" if q_wind > 0 else "—")
    rc4.metric("🔌 Energía Evacuada", f"{coll_benefit:,.0f} GWh/año",
               delta="Colectora activa" if q_coll > 0 and q_wind > 0 else "—")
    rc5.metric("⚠️ Déficit Energético", f"{unmet:,.0f} GWh/año",
               delta="Crítico" if unmet > 500 else ("Moderado" if unmet > 100 else "OK"),
               delta_color="inverse" if unmet > 500 else ("normal" if unmet > 100 else "normal"))

    # Impact gauge
    impact_score = min(100, max(0, 100 - (unmet / 50) + (wind_gen / 100) + (q_bess / 20)))
    gauge_color = "#00CC66" if impact_score > 60 else ("#FFAA00" if impact_score > 30 else "#CC0000")
    st.markdown(f"""
    <div style='background:{gauge_color}10;padding:16px;border-radius:12px;text-align:center;
                border-left:5px solid {gauge_color};'>
    <b style='font-size:1.5em;color:{gauge_color};'>🎯 Score de Impacto: {impact_score:.0f}/100</b><br/>
    <span style='font-size:0.85em;color:#555;'>
    {"🟢 Excelente — Sistema bien configurado" if impact_score > 60 else "🟡 Moderado — Considere más inversión" if impact_score > 30 else "🔴 Crítico — Se requiere acción urgente"}
    </span></div>""", unsafe_allow_html=True)

    # ─── INTERACTIVITY STATS ───
    st.markdown("---")
    st.subheader("📊 ¿Qué Puede Hacer en Este Dashboard?")
    feat_cols = st.columns(4)
    feat_data = [
        ("🎮 Controles Interactivos", "50+", "Sliders, checkboxes, selectores"),
        ("📊 Gráficos Dinámicos", "25+", "Se actualizan en tiempo real"),
        ("⚡ Modelo de Despacho Real", "7 nodos", "Optimización LOPF con HiGHS"),
        ("📥 Datos Descargables", "CSV", "Precios nodales exportables"),
    ]
    for col, (title, val, desc) in zip(feat_cols, feat_data):
        col.markdown(f"""<div style='background:#f0f4f8;padding:12px;border-radius:10px;text-align:center;
                     border-left:4px solid #003399;'>
        <b style='color:#003399;font-size:1.3em;'>{val}</b><br/>
        <span style='font-size:0.85em;color:#333;'>{title}</span><br/>
        <span style='font-size:0.7em;color:#777;'>{desc}</span></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎓 Tour Rápido — ¿Qué Hay en Cada Hoja?")
    tour_data = [
        ("🗺️ Mapa de Red", "Filtre por voltaje, proyecte demanda, agregue nuevas líneas, vea riesgo de congestión"),
        ("⚡ Arquitectura SIN", "Agregue eólica/solar/BESS/gas/hidro, vea mix energético y capacidad regional cambiar"),
        ("🔴 Corredores", "Ajuste capacidad de cada corredor, vea utilización en tiempo real, antes vs. después"),
        ("💰 Precios Nodales", "Modelo de despacho real, 3 vistas (líneas/heatmap/spread), descargue CSV"),
        ("🎮 Simulador", "17 controles + mix generación + balance waterfall + modo comparación A vs B"),
        ("📊 Escenarios", "4 vistas: proyección, viaje en el tiempo, radar multi-criterio, inversión"),
        ("🗄️ Datos", "Busque y filtre 12 datasets por nombre, tipo y año"),
        ("📋 Política", "Active 8 políticas, vea impacto en congestión/inversión/CO₂"),
    ]
    for sheet, desc in tour_data:
        st.markdown(f"- **{sheet}**: {desc}")

    st.markdown("---")
    st.subheader("📈 Línea de Tiempo")

    # ─── GUIDED CHALLENGES ───
    st.markdown("---")
    st.subheader("🏆 Desafíos Interactivos — ¿Puede Resolver la Crisis Energética?")
    st.markdown("*Use los controles de la Calculadora Rápida arriba y vea si logra superar estos desafíos:*")

    challenges = [
        {"name": "🎯 Desafío 1: Desbloquear La Guajira",
         "desc": "Instale suficiente eólica Y capacidad de Colectora para evacuar al menos 3,000 GWh/año",
         "check": lambda: wind_gen > 3000 and coll_benefit > 2000 and q_wind > 500,
         "hint": "💡 Pista: Necesita >500 MW eólica + Colectora con capacidad suficiente"},
        {"name": "🌵 Desafío 2: Sobrevivir El Niño",
         "desc": "Con clima El Niño, mantenga el déficit energético por debajo de 100 GWh/año",
         "check": lambda: q_wx == "El Niño" and unmet < 100,
         "hint": "💡 Pista: El Niño reduce hidro al 35%. Use BESS y más capacidad para compensar"},
        {"name": "🌱 Desafío 3: Transición Limpia",
         "desc": "Logre un score de impacto >80 con cero déficit y máxima reducción de CO₂",
         "check": lambda: impact_score > 80 and unmet < 10 and co2_reduction > 1000,
         "hint": "💡 Pista: Combine eólica + Colectora + BESS para máximo impacto"},
    ]

    ch_cols = st.columns(3)
    for col, ch in zip(ch_cols, challenges):
        passed = ch["check"]()
        status_color = "#00CC66" if passed else "#999"
        status_icon = "✅ PASADO" if passed else "⏳ Intente"
        col.markdown(f"""<div style='background:{status_color}10;padding:16px;border-radius:12px;
                     border-left:5px solid {status_color};min-height:180px;'>
        <b style='color:{status_color};font-size:1.1em;'>{ch['name']}</b><br/>
        <span style='font-size:1.3em;color:{status_color};'>{status_icon}</span><br/><br/>
        <span style='font-size:0.85em;color:#333;'>{ch['desc']}</span><br/><br/>
        <span style='font-size:0.75em;color:#777;font-style:italic;'>{ch['hint']}</span></div>""",
                    unsafe_allow_html=True)

    # Show current values for reference
    st.markdown(f"""<div style='background:#f0f4f8;padding:10px;border-radius:8px;font-size:0.8em;'>
    <b>📊 Valores actuales:</b> Eólica: {wind_gen:,.0f} GWh | Evacuada: {coll_benefit:,.0f} GWh | 
    Déficit: {unmet:,.0f} GWh | CO₂: -{co2_reduction/1000:.0f} kt | Score: {impact_score:.0f}/100
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tl = pd.DataFrame({
        "Año": [2018, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027],
        "Evento": ["Colectora diseñada (meta: 2022)", "COVID retrasa consultas indígenas",
                   "Colectora retrasada a 2024", "CREG reconoce congestión SIN",
                   "Capacidad SIN casi agotada", "Moratoria de conexiones",
                   "17 nuevos proyectos entran SIN", "Colectora 90%, UPME Res. 358",
                   "Proyectado: Colectora operativa?"],
        "Severidad": [3, 5, 6, 7, 8, 9, 7, 8, 5],
    })
    fig = go.Figure(go.Scatter(x=tl["Año"], y=tl["Severidad"], mode='lines+markers+text',
                               line=dict(color='#CC0000', width=3), marker=dict(size=12),
                               text=tl["Evento"], textposition="top center", textfont=dict(size=8)))
    fig.update_layout(template="plotly_white", height=350, yaxis_title="Severidad (1-10)",
                     xaxis_title="Año", title="Cronología de Congestión SIN Colombia")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  GRID POSTER
# ═══════════════════════════════════════════════════════════════════════
elif page == "🗺️ Mapa de Red":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("🗺️ Red de Transmisión de Colombia — Grid2Poster")
    section_header("Visualización Interactiva basada en OpenStreetMap",
                   "2,517 segmentos de líneas. **Filtre por voltaje**, proyecte demanda y agregue nuevas líneas.")

    # ─── INTERACTIVE VOLTAGE FILTER ───
    st.subheader("🔧 Filtros Interactivos")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        show_low = st.checkbox("⬜ < 115 kV (Distribución)", True, key="map_v1")
        show_mid = st.checkbox("🟦 115–230 kV (Subtransmisión)", True, key="map_v2")
    with fc2:
        show_high = st.checkbox("🟧 230–500 kV (Transmisión)", True, key="map_v3")
        show_vhigh = st.checkbox("🟥 > 500 kV (Extra-alta)", True, key="map_v4")
    with fc3:
        demand_growth = st.slider("📈 Proyección demanda (%):", 0, 50, 0, 5, key="map_dg",
                                  help="Crecimiento de demanda proyectado — afecta necesidad de nuevas líneas")
        new_lines = st.slider("🔌 Nuevas líneas (500 kV):", 0, 5, 0, 1, key="map_nl",
                             help="Líneas de transmisión nuevas de 500 kV")

    # Dynamic voltage distribution based on filters
    voltage_data = {
        "< 115 kV": {"count": 850, "show": show_low, "color": "#666666", "km": 12500},
        "115–230 kV": {"count": 1100, "show": show_mid, "color": "#0066CC", "km": 15800},
        "230–500 kV": {"count": 500, "show": show_high, "color": "#FF9900", "km": 8200},
        "> 500 kV": {"count": 67, "show": show_vhigh, "color": "#CC0000", "km": 1500},
    }

    visible_count = sum(v["count"] for v in voltage_data.values() if v["show"])
    total_km = sum(v["km"] for v in voltage_data.values() if v["show"])

    # Impact of demand growth on infrastructure needs
    lines_needed = int(demand_growth / 10 * 1.5)  # ~1.5 new 500kV lines per 10% demand growth
    congestion_risk = min(100, 15 + demand_growth * 1.5 - new_lines * 8)

    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric("Segmentos Visibles", f"{visible_count:,}", delta=f"de 2,517 totales")
    mk2.metric("Km Totales Visibles", f"{total_km:,} km")
    mk3.metric("Líneas Necesarias", f"{lines_needed} (500 kV)",
               delta=f"{new_lines - lines_needed:+d} vs. planificadas",
               delta_color="normal" if new_lines >= lines_needed else "inverse")
    mk4.metric("Riesgo Congestión", f"{congestion_risk:.0f}%",
               delta="Crítico" if congestion_risk > 70 else ("Moderado" if congestion_risk > 40 else "Bajo"),
               delta_color="inverse" if congestion_risk > 70 else ("normal" if congestion_risk <= 40 else "normal"))

    # Grid map
    col_map, col_charts = st.columns([3, 2])
    with col_map:
        poster_path = os.path.join(DATA_DIR, "colombia_grid_poster.png")
        if os.path.exists(poster_path):
            st.image(poster_path, use_container_width=True,
                     caption=f"Grid2Poster Colombia — {visible_count} segmentos filtrados")
        else:
            st.warning("Poster no encontrado.")

    with col_charts:
        # Dynamic pie chart based on filters
        fig = go.Figure(go.Pie(
            labels=[k for k, v in voltage_data.items() if v["show"]],
            values=[v["count"] for v in voltage_data.values() if v["show"]],
            marker_colors=[v["color"] for v in voltage_data.values() if v["show"]],
            textinfo='label+percent', hole=0.4))
        fig.update_layout(template="plotly_white", height=300,
                         title=f"Distribución por Voltaje ({visible_count:,} segmentos)")
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart of km by voltage
        fig_km = go.Figure(go.Bar(
            x=[k for k, v in voltage_data.items() if v["show"]],
            y=[v["km"] for v in voltage_data.values() if v["show"]],
            marker_color=[v["color"] for v in voltage_data.values() if v["show"]],
            text=[f"{v['km']:,} km" for v in voltage_data.values() if v["show"]],
            textposition="outside"))
        fig_km.update_layout(template="plotly_white", height=250, yaxis_title="Kilómetros",
                            title="Longitud de Líneas por Voltaje")
        st.plotly_chart(fig_km, use_container_width=True)

    # ─── WHAT-IF: New Infrastructure Impact ───
    if new_lines > 0 or demand_growth > 0:
        st.markdown("---")
        st.subheader("🔮 Análisis What-If: Impacto de Nueva Infraestructura")
        wi1, wi2 = st.columns(2)
        with wi1:
            # Capacity evolution
            base_cap = 38000  # km of 500kV + 230kV lines
            years_ahead = list(range(2025, 2036))
            cap_with_growth = [base_cap + (y-2025) * (200 + new_lines * 475) for y in years_ahead]
            cap_needed = [base_cap * (1 + demand_growth/100 * (y-2025)/10) for y in years_ahead]

            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(x=years_ahead, y=cap_with_growth,
                                         name=f"Con {new_lines} líneas nuevas",
                                         line=dict(color='#00CC66', width=3)))
            fig_cap.add_trace(go.Scatter(x=years_ahead, y=cap_needed,
                                         name=f"Necesario ({demand_growth}% crecimiento)",
                                         line=dict(color='#CC0000', width=2, dash='dash')))
            fig_cap.update_layout(template="plotly_white", height=300,
                                 yaxis_title="km equivalentes", title="Evolución de Capacidad de Red")
            st.plotly_chart(fig_cap, use_container_width=True)
        with wi2:
            # Congestion risk gauge
            risk_level = "CRÍTICO" if congestion_risk > 70 else "ALTO" if congestion_risk > 50 else "MODERADO" if congestion_risk > 30 else "BAJO"
            risk_color = "#FF0000" if congestion_risk > 70 else "#FF6600" if congestion_risk > 50 else "#FFAA00" if congestion_risk > 30 else "#00CC66"
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=congestion_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Riesgo de Congestión: {risk_level}"},
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': risk_color},
                    'steps': [{'range': [0, 30], 'color': 'rgba(0,204,102,0.13)'},
                              {'range': [30, 50], 'color': 'rgba(255,170,0,0.13)'},
                              {'range': [50, 70], 'color': 'rgba(255,102,0,0.13)'},
                              {'range': [70, 100], 'color': 'rgba(255,0,0,0.13)'}],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }))
            fig_risk.update_layout(height=300)
            st.plotly_chart(fig_risk, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  SIN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
elif page == "⚡ Arquitectura SIN":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("⚡ Arquitectura SIN — Sistema Eléctrico Colombiano")
    section_header("Sistema Interconectado Nacional (SIN)",
                   "18.4 GW instalados. 67.5% hidro, 27.3% térmico, 5.2% renovable no convencional.")
    # ─── INTERACTIVE CAPACITY BUILDER ───
    st.subheader("🔧 Constructor de Capacidad — ¡Agregue nueva generación y vea el impacto!")
    st.markdown("*Ajuste las capacidades de nueva generación y observe cómo cambia el mix energético en tiempo real:*")

    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
    with bc1:
        add_wind = st.slider("💨 Nueva Eólica (MW):", 0, 5000, 0, 100, key="ab_w",
                            help="Parques eólicos nuevos, principalmente La Guajira")
    with bc2:
        add_solar = st.slider("☀️ Nueva Solar (MW):", 0, 5000, 0, 100, key="ab_s",
                             help="Granjas solares, Cesar/Atlántico/Valle")
    with bc3:
        add_bess = st.slider("🔋 BESS (MW/4h):", 0, 2000, 0, 50, key="ab_b",
                            help="Baterías para gestión de congestión y peak shaving")
    with bc4:
        add_gas = st.slider("🏭 Nuevo Gas (MW):", 0, 2000, 0, 100, key="ab_g",
                           help="Plantas de gas para respaldo")
    with bc5:
        add_hydro = st.slider("💧 Nueva Hidro (MW):", 0, 2000, 0, 100, key="ab_h",
                             help="Expansiones hidroeléctricas")

    # Calculate updated mix
    base_total = 18400  # MW
    base_hydro = 12420; base_thermal = 5023; base_re = 644; base_other = 313
    new_hydro = base_hydro + add_hydro
    new_thermal = base_thermal + add_gas
    new_re = base_re + add_wind + add_solar
    new_other = base_other
    new_total = new_hydro + new_thermal + new_re + new_other + add_bess

    # CO2 impact
    base_co2 = base_thermal * 8760 * 0.45 / 1e6  # Mt/year (avg 0.45 t/MWh thermal)
    new_co2 = new_thermal * 8760 * 0.45 / 1e6 - (add_wind + add_solar) * 8760 * 0.25 * 0.45 / 1e6
    co2_change = new_co2 - base_co2

    # Updated KPIs
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Capacidad Total", f"{new_total/1000:.1f} GW",
               delta=f"+{(new_total-base_total)/1000:.1f} GW" if new_total > base_total else "Base")
    mc2.metric("Hidro %", f"{new_hydro/new_total*100:.1f}%",
               delta=f"{new_hydro/new_total*100-67.5:+.1f}pp" if add_hydro > 0 or add_wind > 0 else "—")
    mc3.metric("Renovable %", f"{(new_hydro+new_re)/new_total*100:.1f}%",
               delta=f"{(new_hydro+new_re)/new_total*100-71:+.1f}pp",
               delta_color="normal" if (new_hydro+new_re)/new_total > 0.71 else "inverse")
    mc4.metric("CO₂ Estimado", f"{new_co2:.1f} Mt/año",
               delta=f"{co2_change:+.1f} Mt",
               delta_color="normal" if co2_change <= 0 else "inverse")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=["Hidro", "Térmico (Gas/Carbón)", "Eólico + Solar", "BESS", "Otro"],
            values=[new_hydro, new_thermal, new_re, add_bess, new_other],
            marker_colors=['#0066CC', '#CC6600', '#00CC66', '#9933CC', '#999999'],
            textinfo='label+percent', hole=0.4))
        fig.update_layout(template="plotly_white", height=320,
                         title=f"Mix de Generación Actualizado ({new_total/1000:.1f} GW)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Updated regional capacity
        updated_caps = {r: REGIONS[r]["cap_MW"] for r in REGIONS}
        if add_wind > 0:
            updated_caps["La Guajira"] += add_wind * 0.7
            updated_caps["Cesar"] += add_wind * 0.3
        if add_solar > 0:
            updated_caps["Cesar"] += add_solar * 0.3
            updated_caps["Atlántico"] += add_solar * 0.3
            updated_caps["Valle del Cauca"] += add_solar * 0.4
        if add_hydro > 0:
            updated_caps["Antioquia"] += add_hydro * 0.6
            updated_caps["Tolima-Huila"] += add_hydro * 0.4

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=list(REGIONS.keys()),
                              y=[REGIONS[r]["cap_MW"] for r in REGIONS],
                              name="Actual", marker_color=[REGIONS[r]["color"] for r in REGIONS], opacity=0.5))
        fig2.add_trace(go.Bar(x=list(updated_caps.keys()),
                              y=[updated_caps[r] - REGIONS[r]["cap_MW"] for r in updated_caps],
                              name="Nueva", marker_color='#00CC66'))
        fig2.update_layout(template="plotly_white", height=320, barmode='stack',
                          yaxis_title="MW", title="Capacidad por Región (Actual + Nueva)")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("🗺️ Detalle Regional")
    tabs = st.tabs(list(REGIONS.keys()))
    for tab, (name, d) in zip(tabs, REGIONS.items()):
        with tab:
            st.markdown(f"""<div style='background:{d['color']}10;padding:16px;border-radius:12px;
                        border-left:5px solid {d['color']};'>
            <b style='color:{d["color"]};font-size:1.2em;'>{name}</b>
            <span style='background:{d["color"]}30;padding:3px 8px;border-radius:4px;
                         font-size:0.8em;margin-left:10px;'>{d["type"]}</span><br/><br/>
            <b>Capacidad:</b> {d['cap_MW']:,} MW | <b>Potencial RE:</b> {d['potential_GW']} GW<br/>
            <b>Coordenadas:</b> {d['lat']}°N, {abs(d['lon'])}°W<br/><br/><i>{d['desc']}</i></div>""",
                        unsafe_allow_html=True)

    # ─── INTERACTIVE NETWORK DIAGRAM ───
    st.markdown("---")
    st.subheader("🕸️ Diagrama de Red Interactivo — Haga click en los nodos")
    st.markdown("*Visualización de la topología SIN. Tamaño = capacidad. Color = tipo de nodo.*")

    # Build network diagram with Plotly
    # Node positions (schematic, not geographic)
    node_positions = {
        "La Guajira": (0.8, 0.9), "Cesar": (0.6, 0.75), "Atlántico": (0.3, 0.8),
        "Antioquia": (0.4, 0.5), "Bogotá": (0.6, 0.35), "Medellín": (0.3, 0.4),
        "Tolima-Huila": (0.5, 0.2), "Valle del Cauca": (0.2, 0.2), "Santander": (0.7, 0.55),
    }

    # Links between nodes
    network_links = [
        ("La Guajira", "Cesar", "Colectora 500kV", 1050),
        ("Cesar", "Atlántico", "230kV", 600),
        ("Antioquia", "Bogotá", "500kV", 2400),
        ("Antioquia", "Medellín", "230kV", 800),
        ("Tolima-Huila", "Bogotá", "230kV", 500),
        ("Antioquia", "Tolima-Huila", "230kV", 600),
        ("Cesar", "Antioquia", "230kV", 400),
        ("Santander", "Atlántico", "230kV", 300),
        ("Valle del Cauca", "Tolima-Huila", "230kV", 200),
    ]

    # Get updated capacities
    node_caps = {r: updated_caps.get(r, REGIONS[r]["cap_MW"]) for r in REGIONS}

    # Create network diagram
    fig_net = go.Figure()

    # Add links
    for src, tgt, label, cap in network_links:
        if src in node_positions and tgt in node_positions:
            x0, y0 = node_positions[src]
            x1, y1 = node_positions[tgt]
            # Line width proportional to capacity
            line_width = max(1, cap / 500)
            fig_net.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                line=dict(color='#CCCCCC', width=line_width),
                hovertext=f"{label} ({cap} MW)", showlegend=False))

    # Add nodes
    for region, (x, y) in node_positions.items():
        cap = node_caps.get(region, REGIONS[region]["cap_MW"])
        color = REGIONS[region]["color"]
        size = max(15, min(50, cap / 100))
        fig_net.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
            text=[f"{region}<br>{cap:,} MW"], textposition="bottom center",
            textfont=dict(size=9),
            hovertext=f"<b>{region}</b><br>Tipo: {REGIONS[region]['type']}<br>"
                     f"Capacidad: {cap:,} MW<br>Potencial RE: {REGIONS[region]['potential_GW']} GW<br>"
                     f"{REGIONS[region]['desc']}",
            showlegend=False))

    fig_net.update_layout(template="plotly_white", height=500,
                         xaxis=dict(visible=False, range=[0, 1]),
                         yaxis=dict(visible=False, range=[0, 1]),
                         title="Topología SIN — Click en nodos para detalles")
    st.plotly_chart(fig_net, use_container_width=True)

    # Node detail selector
    st.subheader("🔍 Explorar Nodo")
    selected_node = st.selectbox("Seleccione un nodo para ver detalles:",
                                list(REGIONS.keys()), key="arch_node_sel")
    node_data = REGIONS[selected_node]
    node_cap = node_caps.get(selected_node, node_data["cap_MW"])

    nd1, nd2, nd3, nd4 = st.columns(4)
    nd1.metric("⚡ Capacidad", f"{node_cap:,} MW",
               delta=f"+{node_cap - node_data['cap_MW']:,} MW nuevo" if node_cap > node_data['cap_MW'] else "Base")
    nd2.metric("🌱 Potencial RE", f"{node_data['potential_GW']} GW")
    nd3.metric("📍 Tipo", node_data["type"])
    nd4.metric("📌 Coordenadas", f"{node_data['lat']}°N, {abs(node_data['lon'])}°W")

    st.info(f"📝 **{selected_node}**: {node_data['desc']}")

    # ─── REGIONAL DEMAND/GENERATION BALANCE ───
    st.markdown("---")
    st.subheader("⚖️ Balance Regional — Ajuste Demanda y Generación por Región")
    st.markdown("*Modifique la demanda y generación de cada región y vea el balance importación/exportación:*")

    region_cols = st.columns(3)
    regional_balance = {}
    for i, (region, data) in enumerate(REGIONS.items()):
        with region_cols[i % 3]:
            st.markdown(f"**{region}**")
            regional_gen = st.slider(f"⚡ Gen {region[:8]} (MW):", 0, 6000,
                                    int(data["cap_MW"]), 50, key=f"rg_{i}",
                                    help=f"Generación instalada en {region}")
            regional_dem = st.slider(f"🏙️ Dem {region[:8]} (MW):", 0, 4000,
                                    int(data["cap_MW"] * 0.6), 50, key=f"rd_{i}",
                                    help=f"Demanda pico en {region}")
            balance = regional_gen - regional_dem
            regional_balance[region] = {"gen": regional_gen, "dem": regional_dem, "balance": balance}

    # Show balance chart
    fig_balance = go.Figure()
    regions_list = list(regional_balance.keys())
    gen_values = [regional_balance[r]["gen"] for r in regions_list]
    dem_values = [regional_balance[r]["dem"] for r in regions_list]
    bal_values = [regional_balance[r]["balance"] for r in regions_list]

    fig_balance.add_trace(go.Bar(x=regions_list, y=gen_values, name="Generación",
                                marker_color='#00CC66'))
    fig_balance.add_trace(go.Bar(x=regions_list, y=dem_values, name="Demanda",
                                marker_color='#CC0000'))
    fig_balance.update_layout(template="plotly_white", height=400, barmode='group',
                             yaxis_title="MW", title="Balance Generación vs Demanda por Región")
    st.plotly_chart(fig_balance, use_container_width=True)

    # Balance summary
    total_gen_regional = sum(regional_balance[r]["gen"] for r in regions_list)
    total_dem_regional = sum(regional_balance[r]["dem"] for r in regions_list)
    net_balance = total_gen_regional - total_dem_regional
    exporters = [r for r in regions_list if regional_balance[r]["balance"] > 0]
    importers = [r for r in regions_list if regional_balance[r]["balance"] < 0]

    rb1, rb2, rb3, rb4 = st.columns(4)
    rb1.metric("⚡ Generación Total", f"{total_gen_regional:,} MW")
    rb2.metric("🏙️ Demanda Total", f"{total_dem_regional:,} MW")
    rb3.metric("⚖️ Balance Neto", f"{net_balance:+,} MW",
               delta="Superávit" if net_balance > 0 else ("Déficit" if net_balance < 0 else "Equilibrio"),
               delta_color="normal" if net_balance >= 0 else "inverse")
    rb4.metric("📊 Exportadores/Importadores", f"{len(exporters)}/{len(importers)}",
               delta=f"{len(exporters)} exportan, {len(importers)} importan")

    # Show exporters/importers
    if exporters or importers:
        exp_imp_cols = st.columns(2)
        with exp_imp_cols[0]:
            st.markdown("**🟢 Regiones Exportadoras (Superávit):**")
            for r in exporters:
                bal = regional_balance[r]["balance"]
                st.markdown(f"- **{r}**: +{bal:,} MW (exporta a la red)")
        with exp_imp_cols[1]:
            st.markdown("**🔴 Regiones Importadoras (Déficit):**")
            for r in importers:
                bal = regional_balance[r]["balance"]
                st.markdown(f"- **{r}**: {bal:,} MW (importa de la red)")

# ═══════════════════════════════════════════════════════════════════════
#  CONGESTION CORRIDORS
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔴 Corredores de Congestión":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("🔴 Corredores de Congestión — Cuellos de Botella")
    section_header("7 Corredores Críticos del SIN",
                   "La congestión impide que generación barata llegue a centros de demanda. "
                   "**Ajuste las capacidades** con los controles y vea el impacto en tiempo real.")

    # ─── INTERACTIVE CORRIDOR CAPACITY ADJUSTMENT ───
    st.subheader("🔧 Ajuste de Capacidad por Corredor")
    st.markdown("*Mueva los sliders para expandir/reducir cada corredor y vea cómo cambia la congestión:*")

    status_colors = {"CRÍTICA": "#FF0000", "ALTA": "#FF6600", "MODERADA": "#FFAA00", "BAJA": "#00CC66"}

    corridor_adjustments = {}
    adj_cols = st.columns(min(len(CORRIDORS), 4))
    for i, c in enumerate(CORRIDORS):
        with adj_cols[i % len(adj_cols)]:
            new_cap = st.slider(
                f"**{c['name'].split('(')[0].strip()}**",
                min_value=0, max_value=c["capacity_MW"] * 3,
                value=c["capacity_MW"], step=50,
                key=f"corr_adj_{i}",
                help=f"Capacidad actual: {c['capacity_MW']} MW. {c['voltage']}, {c['km']} km")
            corridor_adjustments[c["name"]] = {
                "original": c["capacity_MW"],
                "adjusted": new_cap,
                "change_pct": (new_cap - c["capacity_MW"]) / c["capacity_MW"] * 100,
            }

    # Recalculate congestion levels based on adjustments
    # Estimated flow per corridor (based on typical SIN patterns)
    typical_flows = {
        "Colectora (La Guajira → Cesar)": 350,
        "Cesar → Atlántico": 450,
        "Antioquia → Bogotá (500 kV)": 1800,
        "Antioquia → Medellín": 550,
        "Tolima → Bogotá": 420,
        "Valle → Pacífico": 180,
        "Santander → Caribe": 250,
    }

    updated_corridors = []
    for c in CORRIDORS:
        adj = corridor_adjustments.get(c["name"], {"adjusted": c["capacity_MW"]})
        new_cap = adj["adjusted"]
        flow = typical_flows.get(c["name"], c["capacity_MW"] * 0.6)
        util = flow / max(new_cap, 1) * 100

        if util > 90:
            new_cong = "CRÍTICA"; new_color = "#FF0000"
        elif util > 75:
            new_cong = "ALTA"; new_color = "#FF6600"
        elif util > 50:
            new_cong = "MODERADA"; new_color = "#FFAA00"
        else:
            new_cong = "BAJA"; new_color = "#00CC66"

        updated_corridors.append({
            **c, "capacity_MW_adj": new_cap, "utilization": util,
            "congestion_adj": new_cong, "color_adj": new_color
        })

    # Live utilization gauge chart
    st.subheader("📊 Utilización en Tiempo Real")
    fig_gauge = go.Figure()
    for uc in updated_corridors:
        name_short = uc["name"].split("(")[0].strip()[:20]
        fig_gauge.add_trace(go.Bar(
            x=[uc["utilization"]], y=[name_short], orientation='h',
            marker_color=uc["color_adj"],
            text=f'{uc["utilization"]:.0f}% | {uc["capacity_MW_adj"]:,} MW | {uc["congestion_adj"]}',
            textposition="auto", showlegend=False))
        # Capacity limit line
        fig_gauge.add_hline(y=name_short, line_dash="dash", line_color="gray", opacity=0.2)
    fig_gauge.update_layout(template="plotly_white", height=400,
                           xaxis_title="Utilización (%)", xaxis_range=[0, 120],
                           title="Utilización por Corredor (ajustada)")
    # Add threshold lines
    fig_gauge.add_vline(x=50, line_dash="dash", line_color="#FFAA00", opacity=0.5,
                        annotation_text="50%")
    fig_gauge.add_vline(x=75, line_dash="dash", line_color="#FF6600", opacity=0.5,
                        annotation_text="75%")
    fig_gauge.add_vline(x=90, line_dash="dash", line_color="#FF0000", opacity=0.5,
                        annotation_text="90%")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Before/After comparison
    st.subheader("📊 Antes vs Después")
    ba_cols = st.columns(2)
    with ba_cols[0]:
        st.markdown("**Estado Original:**")
        fig_before = go.Figure()
        for c in CORRIDORS:
            fig_before.add_trace(go.Bar(x=[c["capacity_MW"]], y=[c["name"].split("(")[0].strip()[:20]],
                                        orientation='h', marker_color=status_colors.get(c["congestion"], "#999"),
                                        text=f'{c["capacity_MW"]:,} MW', textposition="auto", showlegend=False))
        fig_before.update_layout(template="plotly_white", height=300, xaxis_title="MW",
                                title="Capacidad Original", showlegend=False)
        st.plotly_chart(fig_before, use_container_width=True)
    with ba_cols[1]:
        st.markdown("**Estado Ajustado:**")
        fig_after = go.Figure()
        for uc in updated_corridors:
            fig_after.add_trace(go.Bar(x=[uc["capacity_MW_adj"]],
                                       y=[uc["name"].split("(")[0].strip()[:20]],
                                       orientation='h', marker_color=uc["color_adj"],
                                       text=f'{uc["capacity_MW_adj"]:,} MW', textposition="auto", showlegend=False))
        fig_after.update_layout(template="plotly_white", height=300, xaxis_title="MW",
                               title="Capacidad Ajustada", showlegend=False)
        st.plotly_chart(fig_after, use_container_width=True)

    # Summary impact metrics
    total_cap_change = sum(uc["capacity_MW_adj"] - uc["capacity_MW"] for uc in updated_corridors)
    critical_count = sum(1 for uc in updated_corridors if uc["congestion_adj"] == "CRÍTICA")
    low_count = sum(1 for uc in updated_corridors if uc["congestion_adj"] == "BAJA")
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Cambio Total Capacidad", f"{total_cap_change:+,} MW",
               delta="Expansión" if total_cap_change > 0 else ("Reducción" if total_cap_change < 0 else "Sin cambio"))
    sm2.metric("Corredores Críticos", critical_count,
               delta=f"de {len(CORRIDORS)}", delta_color="inverse" if critical_count > 2 else "normal")
    sm3.metric("Corredores Baja Congestión", low_count,
               delta=f"de {len(CORRIDORS)}", delta_color="normal" if low_count > 3 else "inverse")
    est_investment = max(0, total_cap_change) * 150000  # ~€150k per MW for 500kV
    sm4.metric("Inversión Estimada", f"€{est_investment/1e6:.0f}M" if est_investment > 0 else "—",
               delta="Expansión necesaria" if total_cap_change > 0 else "OK")

    st.markdown("---")

    # ─── BUDGET ALLOCATION GAME ───
    st.subheader("💰 Juego de Asignación de Presupuesto")
    st.markdown("""*Usted tiene un presupuesto de **€3,000M**. Distribuya la inversión entre diferentes 
    proyectos y vea el impacto en congestión, costos y confiabilidad del sistema.*""")

    total_budget = 3000  # M€
    budget_items = [
        {"name": "Colectora Expansión (500 MW)", "cost": 174, "cong_reduction": 15, "icon": "🔌"},
        {"name": "Colectora II Nueva (1 GW)", "cost": 350, "cong_reduction": 25, "icon": "🔌"},
        {"name": "Tolima→Bogotá 500kV (800 MW)", "cost": 280, "cong_reduction": 12, "icon": "🔌"},
        {"name": "BESS La Guajira (400 MW/4h)", "cost": 200, "cong_reduction": 8, "icon": "🔋"},
        {"name": "BESS Bogotá (300 MW/4h)", "cost": 150, "cong_reduction": 6, "icon": "🔋"},
        {"name": "Grid-Enhancing Tech (DLR+FACTS)", "cost": 120, "cong_reduction": 10, "icon": "⚡"},
        {"name": "Demanda Response (500 MW)", "cost": 80, "cong_reduction": 5, "icon": "📉"},
        {"name": "Nueva Solar Cesar (1 GW)", "cost": 400, "cong_reduction": 3, "icon": "☀️"},
    ]

    st.markdown("**Seleccione proyectos a financiar:**")
    selected_projects = []
    proj_cols = st.columns(4)
    for i, proj in enumerate(budget_items):
        with proj_cols[i % 4]:
            is_selected = st.checkbox(
                f"{proj['icon']} {proj['name']}",
                value=(i < 3),
                key=f"budget_proj_{i}",
                help=f"Costo: €{proj['cost']}M | Reduce congestión: {proj['cong_reduction']}%")
            if is_selected:
                selected_projects.append(proj)

    # Calculate budget usage and impact
    spent = sum(p["cost"] for p in selected_projects)
    remaining = total_budget - spent
    cong_reduction = sum(p["cong_reduction"] for p in selected_projects)
    base_cong = 65
    new_cong = max(5, base_cong - cong_reduction)

    # Budget gauge
    budget_pct = spent / total_budget * 100
    budget_color = "#00CC66" if budget_pct <= 80 else ("#FFAA00" if budget_pct <= 100 else "#CC0000")

    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
    bm1.metric("💰 Presupuesto Usado", f"€{spent:,}M",
               delta=f"€{remaining:,}M restante" if remaining >= 0 else f"€{abs(remaining):,}M EXCEDIDO",
               delta_color="normal" if remaining >= 0 else "inverse")
    bm2.metric("📊 Congestión Resultante", f"{new_cong:.0f}%",
               delta=f"-{cong_reduction:.0f} puntos",
               delta_color="normal" if cong_reduction > 0 else "inverse")
    bm3.metric("🔌 MW Nuevos", f"{sum(p['cost'] for p in selected_projects if 'MW' in p['name']):,} MW",
               delta=f"{len(selected_projects)} proyectos")
    bm4.metric("📈 ROI Estimado", f"{cong_reduction * 45 / max(spent, 1) * 100:.0f}%",
               delta="Ahorro/año vs inversión")
    bm5.metric("✅ Estado Presupuesto",
               "OK" if spent <= total_budget else "⚠️ Excedido",
               delta=f"{budget_pct:.0f}% usado",
               delta_color="normal" if budget_pct <= 100 else "inverse")

    # Budget bar chart
    fig_budget = go.Figure()
    fig_budget.add_trace(go.Bar(
        x=[p["name"] for p in budget_items],
        y=[p["cost"] if p in selected_projects else 0 for p in budget_items],
        marker_color=['#0066CC' if p in selected_projects else '#DDDDDD' for p in budget_items],
        text=[f"€{p['cost']}M" if p in selected_projects else "—" for p in budget_items],
        textposition="outside"))
    fig_budget.add_hline(y=total_budget / len(budget_items) * 2, line_dash="dash", line_color="red",
                        annotation_text="Promedio por proyecto")
    fig_budget.update_layout(template="plotly_white", height=350, yaxis_title="Inversión (M€)",
                            title="Distribución de Presupuesto por Proyecto")
    st.plotly_chart(fig_budget, use_container_width=True)

    # Impact summary
    if spent <= total_budget and len(selected_projects) > 0:
        success_msg = f"""
        <div style='background:#d4edda;padding:16px;border-radius:12px;border-left:5px solid #28a745;'>
        <b style='color:#28a745;font-size:1.2em;'>✅ Plan de Inversión Viable</b><br/><br/>
        <b>Inversión Total:</b> €{spent:,}M de €{total_budget:,}M ({budget_pct:.0f}%)<br/>
        <b>Proyectos Financiados:</b> {len(selected_projects)} de {len(budget_items)}<br/>
        <b>Reducción de Congestión:</b> {base_cong}% → {new_cong}% (-{cong_reduction} puntos)<br/>
        <b>Ahorro Anual Estimado:</b> €{cong_reduction * 45:,}M (por reducción de costos de congestión)<br/>
        <b>Payback:</b> {spent / max(cong_reduction * 45, 1):.1f} años
        </div>"""
        st.markdown(success_msg, unsafe_allow_html=True)
    elif spent > total_budget:
        st.error(f"⚠️ **Presupuesto excedido por €{spent - total_budget:,}M**. Desactive proyectos para volver al límite.")

    # ─── COST-BENEFIT ANALYZER ───
    st.markdown("---")
    st.subheader("📈 Analizador de Costo-Beneficio")
    st.markdown("*Ingrese una inversión personalizada y calcule el retorno financiero:*")

    cba_cols = st.columns(4)
    with cba_cols[0]:
        cba_investment = st.number_input("💰 Inversión (M€):", min_value=0, max_value=5000, value=200, step=50, key="cba_inv")
    with cba_cols[1]:
        cba_cong_reduction = st.slider("📉 Reducción congestión (%):", 1, 50, 10, 1, key="cba_cr",
                                       help="Porcentaje de congestión que se elimina con esta inversión")
    with cba_cols[2]:
        cba_lifetime = st.slider("⏱️ Vida útil (años):", 10, 50, 30, 5, key="cba_life")
    with cba_cols[3]:
        cba_discount = st.slider("📊 Tasa descuento (%):", 1, 15, 8, 1, key="cba_disc",
                                help="Tasa de descuento para VPN")

    # Calculate financial metrics
    base_cong_cost = 4500  # M€/year baseline congestion cost
    annual_savings = base_cong_cost * cba_cong_reduction / 100  # M€/year
    total_savings = sum(annual_savings / (1 + cba_discount/100)**t for t in range(1, cba_lifetime + 1))
    npv = total_savings - cba_investment
    payback = cba_investment / max(annual_savings, 0.01)
    irr = (annual_savings / cba_investment * 100) if cba_investment > 0 else 0

    # Simple IRR approximation
    if cba_investment > 0 and annual_savings > 0:
        # Binary search for IRR
        low, high = 0, 100
        for _ in range(50):
            mid = (low + high) / 2
            npv_test = sum(annual_savings / (1 + mid/100)**t for t in range(1, cba_lifetime + 1)) - cba_investment
            if npv_test > 0:
                low = mid
            else:
                high = mid
        irr = (low + high) / 2

    cbr1, cbr2, cbr3, cbr4 = st.columns(4)
    cbr1.metric("💰 VPN", f"€{npv:,.0f}M",
                delta="✅ Rentable" if npv > 0 else "❌ No rentable",
                delta_color="normal" if npv > 0 else "inverse")
    cbr2.metric("⏱️ Payback", f"{payback:.1f} años",
                delta="Rápido" if payback < 10 else ("Moderado" if payback < 20 else "Lento"),
                delta_color="normal" if payback < 10 else "inverse")
    cbr3.metric("📈 TIR", f"{irr:.1f}%",
                delta="Alto" if irr > 15 else ("Moderado" if irr > 8 else "Bajo"),
                delta_color="normal" if irr > 8 else "inverse")
    cbr4.metric("💵 Ahorro Anual", f"€{annual_savings:,.0f}M",
                delta=f"{cba_cong_reduction}% menos congestión")

    # Cash flow chart
    years_range = list(range(cba_lifetime + 1))
    cumulative_cf = [-cba_investment]
    for t in range(1, cba_lifetime + 1):
        cumulative_cf.append(cumulative_cf[-1] + annual_savings)

    fig_cba = go.Figure()
    fig_cba.add_trace(go.Scatter(x=years_range, y=cumulative_cf, mode='lines+markers',
                                name="Flujo Acumulado", line=dict(color='#0066CC', width=3)))
    fig_cba.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Punto de equilibrio")
    fig_cba.add_vline(x=payback, line_dash="dash", line_color="orange", annotation_text=f"Payback: {payback:.1f} años")
    fig_cba.update_layout(template="plotly_white", height=350, xaxis_title="Años", yaxis_title="Flujo acumulado (M€)",
                         title=f"Análisis de Rentabilidad — Inversión €{cba_investment}M")
    st.plotly_chart(fig_cba, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Análisis Detallado por Corredor")
    tabs = st.tabs([c["name"].split("(")[0].strip() for c in CORRIDORS])
    for tab, c in zip(tabs, CORRIDORS):
        with tab:
            st.markdown(f"""<div style='background:{c["color"]}08;padding:16px;border-radius:12px;
                        border-left:5px solid {c["color"]};'>
            <b style='color:{c["color"]};font-size:1.15em;'>{c["name"]}</b>
            <span style='background:{c["color"]}30;padding:3px 8px;border-radius:4px;
                         font-size:0.8em;margin-left:10px;'>{c["congestion"]}</span><br/><br/>
            <table style='width:100%;font-size:0.9em;'>
            <tr><td><b>Voltaje:</b></td><td>{c["voltage"]}</td><td><b>Longitud:</b></td><td>{c["km"]} km</td></tr>
            <tr><td><b>Capacidad:</b></td><td>{c["capacity_MW"]:,} MW</td><td><b>Estado:</b></td><td>{c["status"]}</td></tr>
            </table><br/><b>🔴 Impacto:</b> {c["impact"]}<br/><b>⏱️ Retrasos:</b> {c["delay"]}</div>""",
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  NODAL PRICES
# ═══════════════════════════════════════════════════════════════════════
elif page == "💰 Precios Nodales":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("💰 Análisis de Precios Nodales")
    section_header("Diferenciales de Precio, Heatmap y Costos de Restricción",
                   "Modelo de despacho real. **Ajuste parámetros** y vea el impacto en precios nodo a nodo, hora a hora.")

    if not HAS_SOLVER:
        st.error("SciPy no disponible. Agrega `scipy` a requirements.txt")
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros de Precio")
        np_gas = st.slider("Precio gas (€/MWh):", 20, 100, 50, 5, key="np_gas")
        np_coal = st.slider("Precio carbón (€/MWh):", 15, 80, 40, 5, key="np_coal")
        np_wind = st.slider("Eólico Guajira (MW):", 0, 2000, 0, 100, key="np_wind")
        np_coll_cap = st.slider("Colectora (MW):", 0, 2000, 1050, 50, key="np_coll_cap")
        np_coll2 = st.slider("Colectora II (MW):", 0, 2000, 0, 100, key="np_coll2")
        np_bogota_d = st.slider("Demanda Bogotá (MW):", 500, 3500, 2000, 100, key="np_bogota")
        np_caribe_d = st.slider("Demanda Caribe (MW):", 300, 2500, 1200, 100, key="np_caribe")
        np_wx = st.selectbox("Clima:", ["Normal", "El Niño", "La Niña"], key="np_wx")
        np_hours = st.slider("Horas simulación:", 24, 168, 72, 24, key="np_hours")
        np_bess = st.slider("BESS Guajira (MW):", 0, 500, 0, 25, key="np_bess")
        np_view = st.radio("Vista:", ["📈 Líneas temporales", "🟧 Heatmap", "📊 Spread análisis"], key="np_view")

    with st.spinner("⏳ Ejecutando modelo de despacho para precios nodales..."):
        np_result = run_congestion_model(
            guajira_wind_MW=np_wind, colectora_cap_MW=np_coll_cap,
            antioquia_hydro_MW=4200, bogota_demand_MW=np_bogota_d,
            caribe_demand_MW=np_caribe_d, medellin_demand_MW=800,
            gas_price=np_gas, coal_price=np_coal,
            weather=np_wx, bess_guajira_MW=np_bess,
            bess_bogota_MW=0, bess_caribe_MW=0,
            colectora2_cap_MW=np_coll2, tolima_bog_500_cap_MW=0,
            n_hours=np_hours, seed=42)

    if np_result is None:
        st.error("Error en simulación."); st.stop()

    nodal_prices = np_result['nodal_prices']
    nodes_list = nodal_prices.columns.tolist()
    n_hours_actual = len(nodal_prices)

    # KPIs from the dispatch result
    avg_prices = {node: nodal_prices[node].mean() for node in nodes_list}
    max_spread = max(avg_prices.values()) - min(avg_prices.values())
    total_cong_cost = sum(v['congestion_cost_EUR'] for v in np_result['congestion'].values())
    price_volatility = {node: nodal_prices[node].std() for node in nodes_list}

    nk1, nk2, nk3, nk4, nk5 = st.columns(5)
    nk1.metric("💰 Precio Prom. Sistema", f"€{np.mean(list(avg_prices.values())):.1f}/MWh")
    nk2.metric("📊 Spread Máximo", f"€{max_spread:.1f}/MWh",
               delta="Alto" if max_spread > 30 else ("Moderado" if max_spread > 10 else "Bajo"),
               delta_color="inverse" if max_spread > 30 else "normal")
    nk3.metric("💸 Costo Congestión Total", f"€{total_cong_cost:,.0f}")
    nk4.metric("📈 Nodo Más Caro", f"{max(avg_prices, key=avg_prices.get)}",
               delta=f"€{max(avg_prices.values()):.1f}")
    nk5.metric("📉 Nodo Más Barato", f"{min(avg_prices, key=avg_prices.get)}",
               delta=f"€{min(avg_prices.values()):.1f}")

    # Weather indicator
    wx_emoji = {"Normal": "☀️", "El Niño": "🌵", "La Niña": "🌧️"}
    st.info(f"{wx_emoji.get(np_wx, '☀️')} **{np_wx}** | "
            f"{np_wind} MW eólico | Colectora {np_coll_cap} MW" +
            (f" + II {np_coll2} MW" if np_coll2 > 0 else "") +
            f" | BESS {np_bess} MW")

    # ─── VIEW 1: Time Series ───
    if np_view == "📈 Líneas temporales":
        fig = go.Figure()
        nc = {"La Guajira": "#00CC66", "Cesar": "#FF9900", "Caribe": "#CC0000",
              "Antioquia": "#0066CC", "Medellin": "#9933CC", "Bogota": "#CC3366", "Tolima": "#0099CC"}
        for node in nodes_list:
            prices = nodal_prices[node].values
            fig.add_trace(go.Scatter(
                x=list(range(n_hours_actual)), y=np.clip(prices, 0, 300),
                name=node, line=dict(color=nc.get(node, '#999'), width=1.5)))
        fig.update_layout(template="plotly_white", height=400, xaxis_title="Hora", yaxis_title="€/MWh",
                         title=f"Precios Nodales — {np_wx} ({n_hours_actual} horas)")
        st.plotly_chart(fig, use_container_width=True)

        # Price statistics table
        st.subheader("📊 Estadísticas por Nodo")
        stats_df = pd.DataFrame([{
            "Nodo": node,
            "Promedio (€/MWh)": f"{avg_prices[node]:.1f}",
            "Mínimo": f"{nodal_prices[node].min():.1f}",
            "Máximo": f"{nodal_prices[node].max():.1f}",
            "Volatilidad (σ)": f"{price_volatility[node]:.1f}",
            "Horas > €100": f"{int((nodal_prices[node] > 100).sum())}",
            "Horas < €10": f"{int((nodal_prices[node] < 10).sum())}",
        } for node in nodes_list])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ─── VIEW 2: Heatmap ───
    elif np_view == "🟧 Heatmap":
        st.subheader("🟧 Heatmap de Precios Nodales (Nodo × Hora)")
        st.markdown("*Colores más oscuros = precios más altos. La congestión se ve como diferencias verticales.*")

        # Prepare heatmap data
        z_data = np.clip(nodal_prices.values.T, 0, 200)
        fig_hm = go.Figure(data=go.Heatmap(
            z=z_data, x=list(range(n_hours_actual)), y=nodes_list,
            colorscale='RdYlGn_r',
            colorbar=dict(title="€/MWh", len=0.8),
            hovertemplate='<b>%{y}</b><br>Hora: %{x}<br>Precio: €%{z:.1f}/MWh<extra></extra>'))
        fig_hm.update_layout(template="plotly_white", height=400, xaxis_title="Hora",
                            title="Heatmap de Precios Nodales (LOPF)")
        st.plotly_chart(fig_hm, use_container_width=True)

        # Hourly price range (min-max band)
        fig_range = go.Figure()
        min_prices = nodal_prices.min(axis=1)
        max_prices = nodal_prices.max(axis=1)
        mean_prices = nodal_prices.mean(axis=1)
        fig_range.add_trace(go.Scatter(
            x=list(range(n_hours_actual)), y=max_prices, name="Máximo",
            line=dict(color='rgba(204,0,0,0.5)', width=1), fill=None))
        fig_range.add_trace(go.Scatter(
            x=list(range(n_hours_actual)), y=min_prices, name="Mínimo",
            line=dict(color='rgba(0,102,204,0.5)', width=1), fill='tonexty',
            fillcolor='rgba(0,100,200,0.1)'))
        fig_range.add_trace(go.Scatter(
            x=list(range(n_hours_actual)), y=mean_prices, name="Promedio",
            line=dict(color='#333', width=2)))
        fig_range.update_layout(template="plotly_white", height=300, xaxis_title="Hora", yaxis_title="€/MWh",
                               title="Rango de Precios por Hora (min/max/promedio)")
        st.plotly_chart(fig_range, use_container_width=True)

    # ─── VIEW 3: Spread Analysis ───
    elif np_view == "📊 Spread análisis":
        st.subheader("📊 Análisis de Diferenciales de Precio (Spread)")
        st.markdown("*El spread entre nodos indica congestión. Mayor spread = mayor costo de congestión.*")

        # Spread matrix (node vs node)
        st.markdown("**Matriz de Spread Promedio (€/MWh):**")
        spread_matrix = pd.DataFrame(index=nodes_list, columns=nodes_list, dtype=float)
        for n1 in nodes_list:
            for n2 in nodes_list:
                spread_matrix.loc[n1, n2] = (nodal_prices[n1] - nodal_prices[n2]).abs().mean()

        fig_sm = go.Figure(data=go.Heatmap(
            z=spread_matrix.values, x=nodes_list, y=nodes_list,
            colorscale='Reds',
            colorbar=dict(title="Spread €"),
            hovertemplate='<b>%{y} → %{x}</b><br>Spread: €%{z:.1f}<extra></extra>'))
        fig_sm.update_layout(template="plotly_white", height=400,
                            title="Matriz de Diferenciales de Precio Promedio")
        st.plotly_chart(fig_sm, use_container_width=True)

        # Congestion cost per corridor
        st.markdown("**Costo de Congestión por Corredor:**")
        cong_df = pd.DataFrame([{
            "Corredor": k, "Cap. (MW)": f"{v['capacity_MW']:.0f}",
            "Horas Congestión": f"{v['congestion_hours']:.0f}",
            "ΔPrecio Prom.": f"€{v['avg_price_diff']:.1f}",
            "ΔPrecio Máx.": f"€{v['max_price_diff']:.1f}",
            "Costo Congestión": f"€{v['congestion_cost_EUR']:,.0f}",
        } for k, v in np_result['congestion'].items()])
        st.dataframe(cong_df, use_container_width=True, hide_index=True)

        # Congestion cost bar chart
        fig_cong = go.Figure(go.Bar(
            x=list(np_result['congestion'].keys()),
            y=[v['congestion_cost_EUR'] for v in np_result['congestion'].values()],
            marker_color=[
                '#FF0000' if v['congestion_pct'] > 50 else
                '#FF6600' if v['congestion_pct'] > 20 else
                '#FFAA00' if v['congestion_pct'] > 5 else '#00CC66'
                for v in np_result['congestion'].values()],
            text=[f"€{v['congestion_cost_EUR']:,.0f}" for v in np_result['congestion'].values()],
            textposition="outside"))
        fig_cong.update_layout(template="plotly_white", height=350,
                              yaxis_title="Costo (€)", title="Costo de Congestión por Corredor")
        st.plotly_chart(fig_cong, use_container_width=True)

    st.markdown("---")

    # ─── DEMAND RESPONSE CONTROLS ───
    st.subheader("📉 Respuesta de Demanda — Reduzca consumo y vea el impacto")
    st.markdown("*La respuesta de demanda (DR) reduce picos de precio. Ajuste la reducción en cada nodo:*")

    dr_cols = st.columns(3)
    with dr_cols[0]:
        dr_bogota = st.slider("🏙️ Reducción Bogotá (%):", 0, 30, 0, 5, key="dr_bog",
                             help="Reducción de demanda en Bogotá mediante programas DR")
    with dr_cols[1]:
        dr_caribe = st.slider("🌴 Reducción Caribe (%):", 0, 30, 0, 5, key="dr_car",
                             help="Reducción de demanda en costa Caribe")
    with dr_cols[2]:
        dr_medellin = st.slider("🏔️ Reducción Medellín (%):", 0, 30, 0, 5, key="dr_med",
                               help="Reducción de demanda en Medellín")

    # Recalculate with demand response
    if dr_bogota > 0 or dr_caribe > 0 or dr_medellin > 0:
        # Apply demand response to nodal prices
        dr_factor_bog = 1 - dr_bogota / 100
        dr_factor_car = 1 - dr_caribe / 100
        dr_factor_med = 1 - dr_medellin / 100

        # Price reduction effect (DR flattens peaks)
        price_reduction_bog = (nodal_prices["Bogota"].max() - nodal_prices["Bogota"].mean()) * dr_bogota / 100 * 0.7
        price_reduction_car = (nodal_prices["Caribe"].max() - nodal_prices["Caribe"].mean()) * dr_caribe / 100 * 0.7
        price_reduction_med = (nodal_prices["Medellin"].max() - nodal_prices["Medellin"].mean()) * dr_medellin / 100 * 0.7

        # Calculate savings
        dr_savings = (price_reduction_bog * np_bogota_d + price_reduction_car * np_caribe_d +
                     price_reduction_med * 800) * np_hours / 1000  # k€

        drm1, drm2, drm3, drm4 = st.columns(4)
        drm1.metric("💰 Ahorro por DR", f"€{dr_savings:,.0f}k",
                   delta=f"{np_hours}h simuladas")
        drm2.metric("📉 Reducción Pico Bogotá", f"-€{price_reduction_bog:.1f}/MWh",
                   delta=f"-{dr_bogota}% demanda")
        drm3.metric("📉 Reducción Pico Caribe", f"-€{price_reduction_car:.1f}/MWh",
                   delta=f"-{dr_caribe}% demanda")
        drm4.metric("📉 Reducción Pico Medellín", f"-€{price_reduction_med:.1f}/MWh",
                   delta=f"-{dr_medellin}% demanda")

        # Show price comparison with/without DR
        st.markdown("**Comparación de Precios: Sin DR vs Con DR**")
        fig_dr = go.Figure()
        fig_dr.add_trace(go.Scatter(
            x=list(range(n_hours_actual)), y=np.clip(nodal_prices["Bogota"].values, 0, 300),
            name="Bogotá (sin DR)", line=dict(color='rgba(204,51,102,0.5)', width=1, dash='dash')))
        fig_dr.add_trace(go.Scatter(
            x=list(range(n_hours_actual)),
            y=np.clip(nodal_prices["Bogota"].values * dr_factor_bog, 0, 300),
            name=f"Bogotá (con DR -{dr_bogota}%)", line=dict(color='#CC3366', width=2)))
        fig_dr.add_trace(go.Scatter(
            x=list(range(n_hours_actual)), y=np.clip(nodal_prices["Caribe"].values, 0, 300),
            name="Caribe (sin DR)", line=dict(color='rgba(204,0,0,0.5)', width=1, dash='dash')))
        fig_dr.add_trace(go.Scatter(
            x=list(range(n_hours_actual)),
            y=np.clip(nodal_prices["Caribe"].values * dr_factor_car, 0, 300),
            name=f"Caribe (con DR -{dr_caribe}%)", line=dict(color='#CC0000', width=2)))
        fig_dr.update_layout(template="plotly_white", height=350, xaxis_title="Hora", yaxis_title="€/MWh",
                            title="Impacto de Respuesta de Demanda en Precios Nodales")
        st.plotly_chart(fig_dr, use_container_width=True)

    st.markdown("---")

    # ─── CARBON FOOTPRINT CALCULATOR ───
    st.subheader("🌱 Calculadora de Huella de Carbono")
    st.markdown("*Vea las emisiones de CO₂ del escenario actual y cómo reducirlas:*")

    # Calculate emissions from generation mix
    emissions_factors = {"Gas": 0.45, "Coal": 0.95, "Diesel": 0.75, "Hydro": 0.01, "Wind": 0.01, "Thermal": 0.50}
    total_emissions = 0
    for gen_name in np_result['gen_p'].columns:
        if gen_name.startswith("Unserved_"):
            continue
        carrier = gen_name.split("_")[0]
        gen_mwh = np_result['gen_p'][gen_name].sum()
        ef = emissions_factors.get(carrier, 0.5)
        total_emissions += gen_mwh * ef

    # Annual projection
    annual_hours = 8760
    annual_factor = annual_hours / np_hours
    annual_emissions = total_emissions * annual_factor / 1000  # kt CO2

    # Target comparison
    target_2030 = annual_emissions * 0.6  # 40% reduction target
    target_2050 = annual_emissions * 0.1  # 90% reduction target

    ce1, ce2, ce3, ce4 = st.columns(4)
    ce1.metric("🏭 Emisiones Actuales", f"{annual_emissions:,.0f} kt CO₂/año",
               delta=f"{np_hours}h simuladas")
    ce2.metric("🎯 Meta 2030", f"{target_2030:,.0f} kt",
               delta=f"-{annual_emissions - target_2030:,.0f} kt necesario",
               delta_color="inverse")
    ce3.metric("🎯 Meta 2050", f"{target_2050:,.0f} kt",
               delta=f"-{annual_emissions - target_2050:,.0f} kt necesario",
               delta_color="inverse")
    # Calculate total generation for RE share
    _total_gen_np = sum(np_result['gen_p'][g].sum() for g in np_result['gen_p'].columns
                        if not g.startswith("Unserved_"))
    re_share = sum(np_result['gen_p'][g].sum() for g in np_result['gen_p'].columns
                   if g.split("_")[0] in ["Wind", "Hydro"]) / max(_total_gen_np, 1) * 100
    ce4.metric("🌿 % Renovable", f"{re_share:.1f}%",
               delta="Bueno" if re_share > 70 else ("Moderado" if re_share > 50 else "Bajo"),
               delta_color="normal" if re_share > 70 else "inverse")

    # Emissions breakdown by carrier
    emissions_by_carrier = {}
    for gen_name in np_result['gen_p'].columns:
        if gen_name.startswith("Unserved_"):
            continue
        carrier = gen_name.split("_")[0]
        gen_mwh = np_result['gen_p'][gen_name].sum()
        ef = emissions_factors.get(carrier, 0.5)
        emissions_by_carrier[carrier] = emissions_by_carrier.get(carrier, 0) + gen_mwh * ef * annual_factor / 1000

    fig_emissions = go.Figure(go.Bar(
        x=list(emissions_by_carrier.keys()),
        y=list(emissions_by_carrier.values()),
        marker_color=['#666' if k == 'Coal' else '#FF9900' if k == 'Gas' else '#CC0000' if k == 'Diesel' else '#0066CC' if k == 'Hydro' else '#00CC66' for k in emissions_by_carrier.keys()],
        text=[f"{v:,.0f} kt" for v in emissions_by_carrier.values()],
        textposition="outside"))
    fig_emissions.update_layout(template="plotly_white", height=300, yaxis_title="kt CO₂/año",
                               title="Emisiones por Fuente de Generación (proyección anual)")
    st.plotly_chart(fig_emissions, use_container_width=True)

    st.markdown("---")

    # ─── Download Button ───
    csv_data = nodal_prices.to_csv(index_label="Hora")
    st.download_button("📥 Descargar Precios Nodales (CSV)", data=csv_data,
                       file_name=f"precios_nodales_{np_wx}_{n_hours_actual}h.csv",
                       mime="text/csv", key="np_download")

# ═══════════════════════════════════════════════════════════════════════
#  CONGESTION SIMULATOR (ENHANCED)
# ═══════════════════════════════════════════════════════════════════════
elif page == "🎮 Simulador de Congestión":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("🎮 Simulador de Congestión — Modelo SIN 7 Nodos")
    section_header("HiGHS LP + Clima, BESS y Nuevas Líneas",
                   "Modelo interactivo de 7 nodos. Ajuste generación, demanda, clima, almacenamiento y nuevas líneas de transmisión.")

    if not HAS_SOLVER:
        st.error("SciPy no disponible. Agrega `scipy` a requirements.txt")
        st.stop()

    # ─── ÍNDICE DE CONTENIDO (ayuda a ubicar las animaciones) ───
    st.markdown("""
    <div style='background:linear-gradient(90deg,#003399,#0066CC);color:white;
                padding:14px 18px;border-radius:10px;margin:10px 0 18px 0;'>
        <div style='font-size:1.05em;font-weight:700;margin-bottom:6px;'>🗺️ ¿Qué hay en esta página?</div>
        <div style='font-size:0.9em;line-height:1.7;'>
            1️⃣ Escenarios predefinidos &nbsp;→&nbsp; 2️⃣ Controles (gen / demanda / clima / BESS) &nbsp;→&nbsp;
            3️⃣ Resultados + <b>Diagrama de Sankey</b> &nbsp;→&nbsp;
            4️⃣ Comparador A vs B<br>
            🎬 <b>ANIMACIONES (al final de la página, después de hacer scroll):</b><br>
            &nbsp;&nbsp;&nbsp;🎲 <b>Monte Carlo</b> — botón <i>"▶️ Ejecutar Monte Carlo"</i><br>
            &nbsp;&nbsp;&nbsp;🎬 <b>Revisión Animada de Escenarios</b> — botón <i>"▶️ Generar Animación de Escenarios"</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── PRESET SCENARIOS ───
    st.subheader("🎯 Escenarios Predefinidos — ¡Un click para cargar!")
    preset_cols = st.columns(6)
    presets = {
        "📍 2025 Actual": {"wind": 0, "hydro": 4200, "gas": 50, "coal": 40, "bogota": 2000,
                          "caribe": 1200, "med": 800, "coll": 1050, "coll2": 0, "tol500": 0,
                          "bg": 0, "bb": 0, "bc": 0, "wx": "Normal", "hrs": 48},
        "🔌 2030 Colectora": {"wind": 1500, "hydro": 4500, "gas": 55, "coal": 45, "bogota": 2400,
                             "caribe": 1500, "med": 1000, "coll": 2000, "coll2": 0, "tol500": 500,
                             "bg": 100, "bb": 100, "bc": 100, "wx": "Normal", "hrs": 72},
        "🌵 El Niño Crisis": {"wind": 0, "hydro": 4200, "gas": 80, "coal": 60, "bogota": 2500,
                             "caribe": 1500, "med": 1000, "coll": 1050, "coll2": 0, "tol500": 0,
                             "bg": 0, "bb": 0, "bc": 0, "wx": "El Niño", "hrs": 48},
        "🌧️ La Niña Exceso": {"wind": 500, "hydro": 4200, "gas": 45, "coal": 35, "bogota": 1800,
                              "caribe": 1000, "med": 700, "coll": 1050, "coll2": 0, "tol500": 0,
                              "bg": 0, "bb": 0, "bc": 0, "wx": "La Niña", "hrs": 48},
        "💨 Eólico Masivo": {"wind": 3000, "hydro": 4200, "gas": 50, "coal": 40, "bogota": 2200,
                            "caribe": 1400, "med": 900, "coll": 2000, "coll2": 1000, "tol500": 500,
                            "bg": 200, "bb": 100, "bc": 150, "wx": "Normal", "hrs": 72},
        "🔋 BESS Pesado": {"wind": 1000, "hydro": 4200, "gas": 55, "coal": 45, "bogota": 2200,
                          "caribe": 1300, "med": 900, "coll": 1500, "coll2": 500, "tol500": 300,
                          "bg": 400, "bb": 400, "bc": 300, "wx": "Normal", "hrs": 96},
    }
    selected_preset = None
    for i, (preset_name, preset_data) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(preset_name, key=f"preset_{i}", use_container_width=True,
                        help=f"Cargar escenario: {preset_name}"):
                selected_preset = preset_data
                st.session_state["sim_preset"] = preset_data
                st.rerun()

    # Load preset into session state if selected
    if "sim_preset" in st.session_state:
        preset = st.session_state["sim_preset"]
    else:
        preset = presets["📍 2025 Actual"]

    with st.sidebar:
        st.markdown("### ⚙️ Controles del Simulador")
        st.markdown("**Generación:**")
        sim_wind = st.slider("Eólico Guajira (MW):", 0, 3000, preset["wind"], 100, key="sim_w")
        sim_hydro = st.slider("Hidro Antioquia (MW):", 1000, 6000, preset["hydro"], 200, key="sim_h")
        sim_gas = st.slider("Precio Gas (€/MWh):", 20, 100, preset["gas"], 5, key="sim_g")
        sim_coal = st.slider("Precio Carbón (€/MWh):", 20, 80, preset["coal"], 5, key="sim_co")
        st.markdown("**Demanda:**")
        sim_bogota = st.slider("Demanda Bogotá (MW):", 500, 3500, preset["bogota"], 100, key="sim_b")
        sim_caribe = st.slider("Demanda Caribe (MW):", 300, 2500, preset["caribe"], 100, key="sim_ca")
        sim_med = st.slider("Demanda Medellín (MW):", 200, 1500, preset["med"], 50, key="sim_m")
        st.markdown("**Transmisión:**")
        sim_coll = st.slider("Colectora (MW):", 0, 3000, preset["coll"], 50, key="sim_c")
        sim_coll2 = st.slider("Colectora II (MW):", 0, 2000, preset["coll2"], 100, key="sim_c2")
        sim_tol500 = st.slider("Tolima→Bogotá 500kV (MW):", 0, 1000, preset["tol500"], 50, key="sim_t5")
        st.markdown("**Almacenamiento (BESS):**")
        sim_bess_gj = st.slider("BESS La Guajira (MW):", 0, 500, preset["bg"], 25, key="sim_bg")
        sim_bess_bg = st.slider("BESS Bogotá (MW):", 0, 500, preset["bb"], 25, key="sim_bb")
        sim_bess_cb = st.slider("BESS Caribe (MW):", 0, 500, preset["bc"], 25, key="sim_bc")
        st.markdown("**Clima:**")
        sim_wx = st.selectbox("Escenario climático:", ["Normal", "El Niño", "La Niña"],
                             index=["Normal", "El Niño", "La Niña"].index(preset["wx"]), key="sim_wx")
        sim_hours = st.slider("Horas simulación:", 24, 168, preset["hrs"], 24, key="sim_hr")

    with st.spinner("Ejecutando optimización LOPF (HiGHS)..."):
        result = run_congestion_model(
            guajira_wind_MW=sim_wind, colectora_cap_MW=sim_coll,
            antioquia_hydro_MW=sim_hydro, bogota_demand_MW=sim_bogota,
            caribe_demand_MW=sim_caribe, medellin_demand_MW=sim_med,
            gas_price=sim_gas, coal_price=sim_coal,
            weather=sim_wx,
            bess_guajira_MW=sim_bess_gj, bess_bogota_MW=sim_bess_bg, bess_caribe_MW=sim_bess_cb,
            colectora2_cap_MW=sim_coll2, tolima_bog_500_cap_MW=sim_tol500,
            n_hours=sim_hours, seed=42)

    if result is None:
        st.error("Simulación falló."); st.stop()

    # KPIs
    cols = st.columns(5)
    cols[0].metric("Costo Total Sistema", f"€{result['total_cost']:,.0f}")
    total_unserved = sum(result['unserved'].values())
    cols[1].metric("Energía No Servida", f"{total_unserved:,.0f} MWh",
                   delta="¡Crítico!" if total_unserved > 100 else "OK",
                   delta_color="inverse" if total_unserved > 100 else "normal")
    congested = sum(1 for v in result['congestion'].values() if v['congestion_pct'] > 10)
    cols[2].metric("Corredores Congestionados", f"{congested}", delta=f"de {len(result['congestion'])}")
    avg_util = np.mean([v['utilization_pct'] for v in result['congestion'].values()])
    cols[3].metric("Utilización Promedio", f"{avg_util:.0f}%")
    avg_spread = np.mean([v['avg_price_diff'] for v in result['congestion'].values()])
    cols[4].metric("Spread Precio Promedio", f"€{avg_spread:.1f}/MWh")

    # Weather indicator + contextual tips
    wx_emoji = {"Normal": "☀️", "El Niño": "🌵", "La Niña": "🌧️"}
    st.info(f"{wx_emoji.get(sim_wx, '☀️')} **Escenario: {sim_wx}** — "
            f"Hidro CF promedio: {np.mean(result['hydro_cf']):.2f} | "
            f"Eólico CF promedio: {np.mean(result['wind_cf']):.2f} | "
            f"Factor demanda promedio: {np.mean(result['demand_factor']):.2f}")

    # ─── CONTEXTUAL TIPS (Real-time feedback) ───
    tips = []
    if sim_wind > 500 and sim_coll < sim_wind * 0.8:
        tips.append("⚠️ **Alerta de Curtailment**: Tiene más eólica ({sim_wind} MW) que capacidad de Colectora ({sim_coll} MW). "
                     "Mucha energía eólica se perderá. Considere aumentar Colectora o agregar Colectora II.")
    if sim_wx == "El Niño" and sim_hydro < 3500:
        tips.append("🌵 **Riesgo El Niño**: Con sequía, hidro reducida a {np.mean(result['hydro_cf']):.0%} CF. "
                     "Demanda sube 15%. Considere más BESS o generación de respaldo.")
    if total_unserved > 100:
        tips.append("🔴 **Energía No Servida Crítica**: {total_unserved:,.0f} MWh no servidos. "
                     "Aumente generación local o capacidad de transmisión.")
    if sim_bess_gj > 0 and sim_wind == 0:
        tips.append("💡 **Tip BESS**: Tiene baterías en La Guajira pero sin eólica. "
                     "Las baterías son más útiles cuando hay generación variable que almacenar.")
    if congested > 4:
        tips.append("🔴 **Alta Congestión**: {congested} de {len(result['congestion'])} corredores congestionados. "
                     "Considere agregar nuevas líneas de transmisión o BESS estratégicos.")
    if avg_spread > 30:
        tips.append("💰 **Spread Alto**: Diferencial de precios de €{avg_spread:.0f}/MWh entre nodos. "
                     "Esto indica congestión severa. Inversión en transmisión tiene alto ROI.")
    if sim_wind > 1000 and sim_coll >= 1500 and total_unserved < 10:
        tips.append("✅ **Excelente configuración**: Buena combinación de eólica + transmisión. "
                     "Sistema funciona bien con baja congestión.")

    if tips:
        for tip in tips[:3]:  # Show max 3 tips
            st.warning(tip)

    # Nodal prices
    st.subheader("💰 Precios Nodales en el Tiempo")
    fig = go.Figure()
    nc = {"La Guajira": "#00CC66", "Cesar": "#FF9900", "Caribe": "#CC0000",
          "Antioquia": "#0066CC", "Medellin": "#9933CC", "Bogota": "#CC3366", "Tolima": "#0099CC"}
    for node in result['nodal_prices'].columns:
        prices = result['nodal_prices'][node].values
        fig.add_trace(go.Scatter(x=list(range(len(prices))), y=np.clip(prices, 0, 300),
                                 name=node, line=dict(color=nc.get(node, '#999'), width=1.5)))
    fig.update_layout(template="plotly_white", height=350, xaxis_title="Hora", yaxis_title="€/MWh",
                     title="Precios Nodales (Precios Sombra del Balance de Bus)")
    st.plotly_chart(fig, use_container_width=True)

    # Congestion table
    st.subheader("🔴 Análisis de Congestión por Corredor")
    cong_df = pd.DataFrame([
        {"Corredor": k, "Cap. (MW)": f"{v['capacity_MW']:.0f}",
         "Flujo Prom. (MW)": f"{v['avg_flow_MW']:.0f}", "Utilización": f"{v['utilization_pct']:.1f}%",
         "Horas Congestión": f"{v['congestion_hours']:.0f}",
         "Congestión %": f"{v['congestion_pct']:.1f}%",
         "ΔPrecio Prom. (€/MWh)": f"{v['avg_price_diff']:.1f}",
         "Costo Congestión (€)": f"{v['congestion_cost_EUR']:,.0f}"}
        for k, v in result['congestion'].items()])
    st.dataframe(cong_df, use_container_width=True, hide_index=True)

    # Transmission flows
    st.subheader("🔌 Flujos de Transmisión")
    fig2 = go.Figure()
    for ln in result['link_p0'].columns:
        flow = result['link_p0'][ln].values
        cap = result['congestion'].get(ln, {}).get('capacity_MW', 1000)
        fig2.add_trace(go.Scatter(x=list(range(len(flow))), y=flow, name=ln))
        fig2.add_hline(y=cap, line_dash="dash", line_color="red", opacity=0.3)
    fig2.update_layout(template="plotly_white", height=350, xaxis_title="Hora", yaxis_title="MW",
                      title="Flujos de Potencia en Enlaces de Transmisión")
    st.plotly_chart(fig2, use_container_width=True)

    # BESS results
    if result['bess']:
        st.subheader("🔋 Operación de Almacenamiento (BESS)")
        bess_tabs = st.tabs(list(result['bess'].keys()))
        for tab, (su_name, bess) in zip(bess_tabs, result['bess'].items()):
            with tab:
                fig_b = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                      subplot_titles=["Carga/Descarga (MW)", "Estado de Carga (MWh)"],
                                      row_heights=[0.6, 0.4])
                hrs = list(range(len(bess['charge'])))
                fig_b.add_trace(go.Scatter(x=hrs, y=bess['charge'], name='Carga', fill='tozeroy',
                                           fillcolor='rgba(0,100,200,0.3)', line=dict(color='#0066CC')), row=1, col=1)
                fig_b.add_trace(go.Scatter(x=hrs, y=bess['discharge'], name='Descarga', fill='tozeroy',
                                           fillcolor='rgba(200,100,0,0.3)', line=dict(color='#CC6600')), row=1, col=1)
                fig_b.add_trace(go.Scatter(x=hrs, y=bess['soc'], name='SOC',
                                           line=dict(color='#003399', width=2)), row=2, col=1)
                fig_b.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_b, use_container_width=True)

    if total_unserved > 0:
        st.subheader("⚠️ Energía No Servida por Región")
        uns = pd.DataFrame([{"Región": k, "No Servida (MWh)": f"{v:,.0f}"}
                            for k, v in result['unserved'].items() if v > 0])
        if len(uns) > 0:
            st.dataframe(uns, use_container_width=True, hide_index=True)

    # ─── GENERATION MIX PIE ───
    st.markdown("---")
    st.subheader("⚡ Mix de Generación (Resultado LOPF)")
    gen_p = result['gen_p']
    # Sum generation by carrier
    gen_by_carrier = {}
    for gen_name in gen_p.columns:
        if gen_name.startswith("Unserved_"):
            continue
        # Determine carrier from generator name
        carrier = gen_name.split("_")[0] if "_" in gen_name else gen_name
        if carrier not in gen_by_carrier:
            gen_by_carrier[carrier] = 0
        gen_by_carrier[carrier] += gen_p[gen_name].sum()

    carrier_colors = {"Wind": "#00CC66", "Hydro": "#0066CC", "Gas": "#FF9900", "Coal": "#666666",
                      "Diesel": "#CC0000", "Thermal": "#CC6600"}
    carrier_labels = {"Wind": "💨 Eólica", "Hydro": "💧 Hidro", "Gas": "🏭 Gas", "Coal": "⛏️ Carbón",
                      "Diesel": "🛢️ Diesel", "Thermal": "🏭 Térmica"}

    gen_mix_cols = st.columns(2)
    with gen_mix_cols[0]:
        fig_mix = go.Figure(go.Pie(
            labels=[carrier_labels.get(k, k) for k in gen_by_carrier.keys() if gen_by_carrier[k] > 0],
            values=[v for v in gen_by_carrier.values() if v > 0],
            marker_colors=[carrier_colors.get(k, '#999') for k in gen_by_carrier.keys() if gen_by_carrier[k] > 0],
            textinfo='label+percent', hole=0.4))
        total_gen = sum(gen_by_carrier.values())
        fig_mix.update_layout(template="plotly_white", height=350,
                             title=f"Mix de Generación ({total_gen:,.0f} MWh total)")
        st.plotly_chart(fig_mix, use_container_width=True)

    with gen_mix_cols[1]:
        # Energy balance waterfall
        st.markdown("**Balance Energético:**")
        total_demand = sum([sim_bogota, sim_caribe, sim_med, 400]) * np.mean(result['demand_factor']) * result['n_hours']
        total_gen_actual = sum(gen_by_carrier.values())
        total_losses = total_gen_actual * 0.03  # ~3% transmission losses
        total_storage = sum(b['charge'].sum() for b in result['bess'].values()) if result['bess'] else 0
        total_discharge = sum(b['discharge'].sum() for b in result['bess'].values()) if result['bess'] else 0

        wf_measures = ["Generación Total", "Pérdidas Red", "Carga BESS", "Descarga BESS",
                       "Demanda Cubierta", "No Servida"]
        wf_values = [total_gen_actual, -total_losses, -total_storage, total_discharge,
                     -(total_gen_actual - total_losses - total_storage + total_discharge),
                     -total_unserved]
        wf_colors = ['#00CC66' if v > 0 else '#CC0000' for v in wf_values]

        fig_wf = go.Figure(go.Waterfall(
            x=wf_measures, y=wf_values,
            connector={"line": {"color": "#999"}},
            increasing={"marker": {"color": "#00CC66"}},
            decreasing={"marker": {"color": "#CC0000"}},
            totals={"marker": {"color": "#0066CC"}}))
        fig_wf.update_layout(template="plotly_white", height=350, yaxis_title="MWh",
                            title="Balance Energético (Waterfall)")
        st.plotly_chart(fig_wf, use_container_width=True)

    # ─── ENERGY FLOW SANKEY ───
    st.markdown("---")
    st.subheader("🔀 Flujo de Energía — Diagrama Sankey")
    st.markdown("*Visualice cómo fluye la energía desde generación → transmisión → demanda. Los anchos representan MWh.*")

    # Build Sankey from the dispatch results
    sankey_labels = []
    sankey_source = []
    sankey_target = []
    sankey_value = []
    sankey_colors = []

    # Generation sources
    gen_carriers = {}
    for gen_name in gen_p.columns:
        if gen_name.startswith("Unserved_"):
            continue
        carrier = gen_name.split("_")[0]
        gen_total = gen_p[gen_name].sum()
        if gen_total > 1:
            if carrier not in gen_carriers:
                gen_carriers[carrier] = 0
            gen_carriers[carrier] += gen_total

    carrier_info = {"Wind": ("💨 Eólica", "#00CC66"), "Hydro": ("💧 Hidro", "#0066CC"),
                    "Gas": ("🏭 Gas", "#FF9900"), "Coal": ("⛏️ Carbón", "#666666"),
                    "Diesel": ("🛢️ Diesel", "#CC0000"), "Thermal": ("🏭 Térmica", "#CC6600")}

    # Add generation nodes
    node_offset = 0
    for carrier, total in gen_carriers.items():
        label = carrier_info.get(carrier, (carrier, "#999"))[0]
        color = carrier_info.get(carrier, (carrier, "#999"))[1]
        sankey_labels.append(f"Gen: {label}")
        sankey_source.append(len(sankey_labels) - 1)
        sankey_target.append(len(sankey_labels))  # will be transmission
        sankey_value.append(total)
        sankey_colors.append(color)

    # Transmission node
    sankey_labels.append("🔌 Red de Transmisión")
    total_gen_all = sum(gen_carriers.values())

    # Demand nodes
    demands = {"Bogotá": sim_bogota, "Caribe": sim_caribe, "Medellín": sim_med, "Tolima": 400}
    for demand_name, demand_val in demands.items():
        demand_total = demand_val * np.mean(result['demand_factor']) * result['n_hours']
        sankey_labels.append(f"🏙️ {demand_name}")
        sankey_source.append(len(sankey_labels) - len(demands) - 1)  # transmission
        sankey_target.append(len(sankey_labels) - 1)
        sankey_value.append(demand_total)
        sankey_colors.append("#CC3366")

    if len(sankey_labels) > 2 and len(sankey_source) > 0:
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                     label=sankey_labels, color=["#0066CC"] * len(sankey_labels)),
            link=dict(source=sankey_source, target=sankey_target, value=sankey_value,
                     color=[c for c in sankey_colors[:len(sankey_source)]])
        )])
        fig_sankey.update_layout(template="plotly_white", height=400,
                                title_text="Flujo de Energía: Generación → Transmisión → Demanda",
                                font_size=10)
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ─── COMPARISON MODE ───
    st.markdown("---")
    st.subheader("🔄 Modo Comparación: Ejecute 2 Escenarios")
    st.markdown("*Seleccione un **preset** o configure manualmente y compare lado a lado:*")

    comp_on = st.checkbox("✅ Activar modo comparación", False, key="sim_comp_on")
    if comp_on:
        comp_preset_name = st.selectbox("🎯 Preset Escenario B:", list(presets.keys()), index=2, key="comp_preset")
        comp_preset = presets[comp_preset_name]

        with st.expander("⚙️ O ajuste manualmente los parámetros del Escenario B:", expanded=False):
            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                comp_wind = st.slider("Eólico Guajira B (MW):", 0, 3000, comp_preset["wind"], 100, key="comp_w")
                comp_hydro = st.slider("Hidro Antioquia B (MW):", 1000, 6000, comp_preset["hydro"], 200, key="comp_h")
            with cb2:
                comp_coll = st.slider("Colectora B (MW):", 0, 3000, comp_preset["coll"], 50, key="comp_c")
                comp_coll2 = st.slider("Colectora II B (MW):", 0, 2000, comp_preset["coll2"], 100, key="comp_c2")
            with cb3:
                comp_bess = st.slider("BESS Guajira B (MW):", 0, 500, comp_preset["bg"], 25, key="comp_bg")
                comp_wx = st.selectbox("Clima B:", ["Normal", "El Niño", "La Niña"],
                                      index=["Normal", "El Niño", "La Niña"].index(comp_preset["wx"]), key="comp_wx")

        with st.spinner("⏳ Ejecutando escenario B..."):
            result_b = run_congestion_model(
                guajira_wind_MW=comp_wind, colectora_cap_MW=comp_coll,
                antioquia_hydro_MW=comp_hydro, bogota_demand_MW=sim_bogota,
                caribe_demand_MW=sim_caribe, medellin_demand_MW=sim_med,
                gas_price=sim_gas, coal_price=sim_coal,
                weather=comp_wx, bess_guajira_MW=comp_bess,
                bess_bogota_MW=sim_bess_bg, bess_caribe_MW=sim_bess_cb,
                colectora2_cap_MW=comp_coll2, tolima_bog_500_cap_MW=sim_tol500,
                n_hours=sim_hours, seed=42)

        if result_b:
            # Side-by-side comparison
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown("**🅰️ Escenario A (actual)**")
                cm1a, cm1b, cm1c = st.columns(3)
                cm1a.metric("Costo", f"€{result['total_cost']:,.0f}")
                cm1b.metric("No Servida", f"{total_unserved:,.0f} MWh")
                cm1c.metric("Congestión", f"{congested} corredores")
            with cmp2:
                st.markdown("**🅱️ Escenario B**")
                total_unserved_b = sum(result_b['unserved'].values())
                congested_b = sum(1 for v in result_b['congestion'].values() if v['congestion_pct'] > 10)
                cm2a, cm2b, cm2c = st.columns(3)
                cm2a.metric("Costo", f"€{result_b['total_cost']:,.0f}",
                           delta=f"{result_b['total_cost']-result['total_cost']:+,.0f}",
                           delta_color="normal" if result_b['total_cost'] < result['total_cost'] else "inverse")
                cm2b.metric("No Servida", f"{total_unserved_b:,.0f} MWh",
                           delta=f"{total_unserved_b-total_unserved:+,.0f}",
                           delta_color="normal" if total_unserved_b < total_unserved else "inverse")
                cm2c.metric("Congestión", f"{congested_b} corredores",
                           delta=f"{congested_b-congested:+d}",
                           delta_color="normal" if congested_b < congested else "inverse")

            # Comparison chart
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(x=['Costo (M€)', 'No Servida (MWh)', 'Corredores Congestionados'],
                                     y=[result['total_cost']/1e6, total_unserved, congested],
                                     name="Escenario A", marker_color='#0066CC'))
            fig_cmp.add_trace(go.Bar(x=['Costo (M€)', 'No Servida (MWh)', 'Corredores Congestionados'],
                                     y=[result_b['total_cost']/1e6, total_unserved_b, congested_b],
                                     name="Escenario B", marker_color='#CC6600'))
            fig_cmp.update_layout(template="plotly_white", height=350, barmode='group',
                                 title="Comparación A vs B")
            st.plotly_chart(fig_cmp, use_container_width=True)

    # ─── MONTE CARLO ANIMATION ───
    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(90deg,#006600,#00CC66);color:white;
                padding:14px 18px;border-radius:10px;margin:16px 0 10px 0;'>
        <div style='font-size:1.25em;font-weight:800;'>🎲 Simulación Monte Carlo — Análisis de Incertidumbre</div>
        <div style='font-size:0.88em;margin-top:4px;'>
            Presione <b>"▶️ Ejecutar Monte Carlo"</b> abajo. Luego use los botones
            <b>▶️ Play</b> / <b>⏸️ Pause</b> y el slider bajo el gráfico para ver
            la distribución construirse escenario a escenario.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""*Genera escenarios aleatorios de demanda, viento y precios para cuantificar 
    la incertidumbre en costos y congestión. La animación muestra cómo convergen los resultados 
    a medida que aumentan los escenarios.*""")

    mc_cols = st.columns(3)
    with mc_cols[0]:
        mc_n = st.slider("🔢 Número de escenarios:", 50, 1000, 200, 50, key="mc_n")
    with mc_cols[1]:
        mc_demand_std = st.slider("📊 σ Demanda (%):", 5, 30, 15, 5, key="mc_dstd",
                                  help="Desviación estándar de la demanda como % del valor base")
    with mc_cols[2]:
        mc_wind_std = st.slider("💨 σ Eólico (%):", 5, 40, 20, 5, key="mc_wstd",
                                help="Desviación estándar del factor de capacidad eólico")

    if st.button("▶️ Ejecutar Monte Carlo", key="mc_run", type="primary"):
        rng_mc = np.random.RandomState(123)
        mc_costs = []
        mc_unserved = []
        mc_congested = []
        mc_spreads = []

        progress_bar = st.progress(0, text="Generando escenarios...")
        status_text = st.empty()

        for i in range(mc_n):
            d_noise = rng_mc.normal(0, mc_demand_std / 100)
            w_noise = rng_mc.normal(0, mc_wind_std / 100)
            g_noise = rng_mc.normal(0, 5)

            mc_bog = max(500, sim_bogota * (1 + d_noise))
            mc_car = max(300, sim_caribe * (1 + d_noise))
            mc_med_v = max(200, sim_med * (1 + d_noise))
            mc_wind_v = max(0, sim_wind * (1 + w_noise))
            mc_gas_v = max(20, sim_gas + g_noise)

            try:
                mc_res = run_congestion_model(
                    guajira_wind_MW=mc_wind_v, colectora_cap_MW=sim_coll,
                    antioquia_hydro_MW=sim_hydro, bogota_demand_MW=mc_bog,
                    caribe_demand_MW=mc_car, medellin_demand_MW=mc_med_v,
                    gas_price=mc_gas_v, coal_price=sim_coal,
                    weather=sim_wx,
                    bess_guajira_MW=sim_bess_gj, bess_bogota_MW=sim_bess_bg, bess_caribe_MW=sim_bess_cb,
                    colectora2_cap_MW=sim_coll2, tolima_bog_500_cap_MW=sim_tol500,
                    n_hours=min(sim_hours, 48), seed=i)

                mc_costs.append(mc_res['total_cost'] / 1e6)
                mc_unserved.append(sum(mc_res['unserved'].values()))
                mc_congested.append(sum(1 for v in mc_res['congestion'].values() if v['congestion_pct'] > 10))
                mc_spreads.append(np.mean([v['avg_price_diff'] for v in mc_res['congestion'].values()]))
            except Exception:
                mc_costs.append(np.nan)
                mc_unserved.append(np.nan)
                mc_congested.append(np.nan)
                mc_spreads.append(np.nan)

            if (i + 1) % max(1, mc_n // 20) == 0:
                progress_bar.progress((i + 1) / mc_n, text=f"Escenario {i+1}/{mc_n}")

        progress_bar.progress(1.0, text="✅ Completado")
        status_text.success(f"✅ {mc_n} escenarios generados exitosamente")

        mc_costs = np.array(mc_costs)
        mc_unserved = np.array(mc_unserved)
        mc_congested = np.array(mc_congested)
        mc_spreads = np.array(mc_spreads)
        valid = ~np.isnan(mc_costs)

        # ─── Animated Plotly histograms ───
        st.markdown("### 📊 Resultados — Histogramas Animados")

        # Create animated histogram with Plotly frames
        n_frames = min(20, mc_n // 10)
        frame_sizes = [max(10, int(len(mc_costs[valid]) * (i + 1) / n_frames)) for i in range(n_frames)]

        fig_anim = go.Figure()

        # First frame
        fig_anim.add_trace(go.Histogram(
            x=mc_costs[valid][:frame_sizes[0]], nbinsx=30,
            marker_color='#0066CC', opacity=0.7,
            name="Costo Total (M€)"))
        fig_anim.add_vline(x=np.mean(mc_costs[valid]), line_dash="dash", line_color="red",
                          annotation_text=f"Media: €{np.mean(mc_costs[valid]):.1f}M")

        # Create frames
        frames = []
        for i, fs in enumerate(frame_sizes):
            frames.append(go.Frame(
                data=[go.Histogram(x=mc_costs[valid][:fs], nbinsx=30,
                                   marker_color='#0066CC', opacity=0.7)],
                name=f"frame_{i}"))

        fig_anim.frames = frames
        fig_anim.update_layout(
            template="plotly_white", height=400,
            xaxis_title="Costo Total Sistema (M€)", yaxis_title="Frecuencia",
            title=f"Distribución de Costos — {len(mc_costs[valid])} escenarios Monte Carlo",
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, dict(frame=dict(duration=200, redraw=True),
                                         fromcurrent=True, mode="immediate")]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode="immediate")])
                ],
                direction="left", pad=dict(r=10, t=30), x=0.1, y=0
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                           args=[[f"frame_{i}"], dict(mode="immediate", frame=dict(duration=200, redraw=True))],
                           label=str(frame_sizes[i]))
                       for i in range(n_frames)],
                transition=dict(duration=100), x=0.1, y=0, len=0.8,
                currentvalue=dict(font=dict(size=12), prefix="Escenarios: ", visible=True, xanchor="center")
            )]
        )
        st.plotly_chart(fig_anim, use_container_width=True)

        # ─── 4-panel Monte Carlo dashboard ───
        st.markdown("### 📈 Panel Monte Carlo Completo")
        p1, p2 = st.columns(2)

        with p1:
            # Cost histogram with statistics
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Histogram(x=mc_costs[valid], nbinsx=40,
                                            marker_color='#0066CC', opacity=0.7, name="Costo"))
            mean_c = np.mean(mc_costs[valid])
            std_c = np.std(mc_costs[valid])
            fig_cost.add_vline(x=mean_c, line_dash="dash", line_color="red",
                              annotation_text=f"μ = €{mean_c:.1f}M")
            fig_cost.add_vline(x=mean_c - 1.96*std_c, line_dash="dot", line_color="orange",
                              annotation_text=f"95% CI bajo")
            fig_cost.add_vline(x=mean_c + 1.96*std_c, line_dash="dot", line_color="orange",
                              annotation_text=f"95% CI alto")
            fig_cost.update_layout(template="plotly_white", height=350,
                                  xaxis_title="Costo Total (M€)", yaxis_title="Frecuencia",
                                  title="Distribución de Costo Total del Sistema")
            st.plotly_chart(fig_cost, use_container_width=True)

        with p2:
            # Unserved energy histogram
            fig_uns = go.Figure()
            valid_uns = mc_unserved[~np.isnan(mc_unserved)]
            fig_uns.add_trace(go.Histogram(x=valid_uns, nbinsx=40,
                                           marker_color='#CC0000', opacity=0.7, name="No Servida"))
            fig_uns.add_vline(x=np.mean(valid_uns), line_dash="dash", line_color="red",
                             annotation_text=f"μ = {np.mean(valid_uns):.0f} MWh")
            fig_uns.update_layout(template="plotly_white", height=350,
                                 xaxis_title="Energía No Servida (MWh)", yaxis_title="Frecuencia",
                                 title="Distribución de Energía No Servida")
            st.plotly_chart(fig_uns, use_container_width=True)

        p3, p4 = st.columns(2)
        with p3:
            # Congestion count histogram
            fig_cong_mc = go.Figure()
            valid_cong = mc_congested[~np.isnan(mc_congested)]
            fig_cong_mc.add_trace(go.Histogram(x=valid_cong, nbinsx=max(5, int(np.ptp(valid_cong))+1),
                                               marker_color='#FF6600', opacity=0.7))
            fig_cong_mc.update_layout(template="plotly_white", height=350,
                                     xaxis_title="Corredores Congestionados", yaxis_title="Frecuencia",
                                     title="Distribución de Corredores Congestionados")
            st.plotly_chart(fig_cong_mc, use_container_width=True)

        with p4:
            # Spread histogram
            fig_spread_mc = go.Figure()
            valid_spread = mc_spreads[~np.isnan(mc_spreads)]
            fig_spread_mc.add_trace(go.Histogram(x=valid_spread,
                                                 nbinsx=30, marker_color='#9933CC', opacity=0.7))
            fig_spread_mc.update_layout(template="plotly_white", height=350,
                                       xaxis_title="Spread Precio Promedio (€/MWh)", yaxis_title="Frecuencia",
                                       title="Distribución de Diferencial de Precios")
            st.plotly_chart(fig_spread_mc, use_container_width=True)

        # ─── Convergence plot ───
        st.markdown("### 📉 Convergencia del Valor Esperado")
        running_mean = np.cumsum(mc_costs[valid]) / np.arange(1, len(mc_costs[valid]) + 1)
        running_std = np.array([np.std(mc_costs[valid][:i]) if i > 1 else 0 for i in range(1, len(mc_costs[valid]) + 1)])

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(running_mean) + 1)), y=running_mean,
            mode='lines', name="Media acumulada", line=dict(color='#0066CC', width=3)))
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(running_mean) + 1)), y=running_mean + 1.96 * running_std,
            mode='lines', name="+95% CI", line=dict(color='rgba(0,102,204,0.25)', width=1, dash='dot')))
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(running_mean) + 1)), y=running_mean - 1.96 * running_std,
            mode='lines', name="-95% CI", line=dict(color='rgba(0,102,204,0.25)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(0,102,204,0.1)'))
        fig_conv.add_hline(y=running_mean[-1], line_dash="dash", line_color="red",
                          annotation_text=f"Valor final: €{running_mean[-1]:.2f}M")
        fig_conv.update_layout(template="plotly_white", height=350,
                              xaxis_title="Número de escenarios", yaxis_title="Costo esperado (M€)",
                              title="Convergencia Monte Carlo — Costo Esperado del Sistema")
        st.plotly_chart(fig_conv, use_container_width=True)

        # ─── Summary statistics ───
        st.markdown("### 📊 Resumen Estadístico")
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Media Costo", f"€{np.mean(mc_costs[valid]):.1f}M",
                   delta=f"σ = €{np.std(mc_costs[valid]):.1f}M")
        sm2.metric("Mediana", f"€{np.median(mc_costs[valid]):.1f}M")
        sm3.metric("P95 (Peor caso)", f"€{np.percentile(mc_costs[valid], 95):.1f}M",
                   delta="5% prob. exceder", delta_color="inverse")
        sm4.metric("P5 (Mejor caso)", f"€{np.percentile(mc_costs[valid], 5):.1f}M",
                   delta="5% prob. inferior", delta_color="normal")
        sm5.metric("CVaR₉₅", f"€{np.mean(mc_costs[valid][mc_costs[valid] >= np.percentile(mc_costs[valid], 95)]):.1f}M",
                   delta="Riesgo cola")

    # ─── SCENARIO REVIEW ANIMATION ───
    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(90deg,#CC3366,#FF6699);color:white;
                padding:14px 18px;border-radius:10px;margin:16px 0 10px 0;'>
        <div style='font-size:1.25em;font-weight:800;'>🎬 Revisión Animada de Escenarios</div>
        <div style='font-size:0.88em;margin-top:4px;'>
            Elija el parámetro a barrer y presione <b>"▶️ Generar Animación de Escenarios"</b>.
            El gráfico de burbujas se construye paso a paso — use <b>▶️ Play</b> / <b>⏸️ Pause</b>
            y el slider de escenarios.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("*Visualice cómo cambian las métricas clave al variar un parámetro mientras los demás se mantienen fijos.*")

    anim_param = st.selectbox("🎯 Parámetro a variar:",
                             ["Eólico Guajira", "Colectora", "Demanda Bogotá", "Precio Gas", "BESS Guajira"],
                             key="anim_param")

    if st.button("▶️ Generar Animación de Escenarios", key="anim_run", type="primary"):
        param_ranges = {
            "Eólico Guajira": np.linspace(0, 2000, 11, dtype=int),
            "Colectora": np.linspace(0, 2000, 11, dtype=int),
            "Demanda Bogotá": np.linspace(500, 3500, 11, dtype=int),
            "Precio Gas": np.linspace(20, 100, 11, dtype=int),
            "BESS Guajira": np.linspace(0, 500, 11, dtype=int),
        }
        param_values = param_ranges[anim_param]
        param_keys = {
            "Eólico Guajira": "guajira_wind_MW",
            "Colectora": "colectora_cap_MW",
            "Demanda Bogotá": "bogota_demand_MW",
            "Precio Gas": "gas_price",
            "BESS Guajira": "bess_guajira_MW",
        }

        scan_results = []
        scan_progress = st.progress(0, text="Escaneando parámetro...")

        for idx, val in enumerate(param_values):
            kwargs = {
                "guajira_wind_MW": sim_wind, "colectora_cap_MW": sim_coll,
                "antioquia_hydro_MW": sim_hydro, "bogota_demand_MW": sim_bogota,
                "caribe_demand_MW": sim_caribe, "medellin_demand_MW": sim_med,
                "gas_price": sim_gas, "coal_price": sim_coal,
                "weather": sim_wx,
                "bess_guajira_MW": sim_bess_gj, "bess_bogota_MW": sim_bess_bg, "bess_caribe_MW": sim_bess_cb,
                "colectora2_cap_MW": sim_coll2, "tolima_bog_500_cap_MW": sim_tol500,
                "n_hours": min(sim_hours, 48), "seed": 42,
            }
            kwargs[param_keys[anim_param]] = int(val)

            try:
                res = run_congestion_model(**kwargs)
                scan_results.append({
                    "param": val,
                    "cost": res['total_cost'] / 1e6,
                    "unserved": sum(res['unserved'].values()),
                    "congested": sum(1 for v in res['congestion'].values() if v['congestion_pct'] > 10),
                    "avg_util": np.mean([v['utilization_pct'] for v in res['congestion'].values()]),
                    "avg_spread": np.mean([v['avg_price_diff'] for v in res['congestion'].values()]),
                })
            except Exception:
                pass

            scan_progress.progress((idx + 1) / len(param_values), text=f"Escenario {idx+1}/{len(param_values)}")

        scan_progress.progress(1.0, text="✅ Escaneo completo")

        if scan_results:
            df_scan = pd.DataFrame(scan_results)

            # Animated bubble chart
            fig_scan = go.Figure()

            # Create frames for animation
            frames_scan = []
            for i in range(len(df_scan)):
                frame_data = [
                    go.Scatter(
                        x=df_scan['param'][:i+1], y=df_scan['cost'][:i+1],
                        mode='lines+markers', name="Costo (M€)",
                        line=dict(color='#0066CC', width=3),
                        marker=dict(size=10, color='#0066CC')),
                    go.Scatter(
                        x=df_scan['param'][:i+1], y=df_scan['avg_util'][:i+1],
                        mode='lines+markers', name="Utilización (%)",
                        line=dict(color='#FF6600', width=2, dash='dash'),
                        marker=dict(size=8, color='#FF6600'),
                        yaxis="y2"),
                ]
                frames_scan.append(go.Frame(data=frame_data, name=f"scan_{i}"))

            fig_scan.add_trace(go.Scatter(
                x=df_scan['param'], y=df_scan['cost'],
                mode='lines+markers', name="Costo (M€)",
                line=dict(color='#0066CC', width=3),
                marker=dict(size=10, color='#0066CC')))
            fig_scan.add_trace(go.Scatter(
                x=df_scan['param'], y=df_scan['avg_util'],
                mode='lines+markers', name="Utilización (%)",
                line=dict(color='#FF6600', width=2, dash='dash'),
                marker=dict(size=8, color='#FF6600'),
                yaxis="y2"))

            fig_scan.frames = frames_scan
            fig_scan.update_layout(
                template="plotly_white", height=450,
                xaxis_title=anim_param + " (MW)" if "MW" not in anim_param else anim_param,
                yaxis=dict(title="Costo (M€)", color='#0066CC'),
                yaxis2=dict(title="Utilización (%)", overlaying="y", side="right", color='#FF6600'),
                title=f"Revisión de Escenarios — Variando {anim_param}",
                legend=dict(x=0.01, y=0.99),
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    buttons=[
                        dict(label="▶️ Play", method="animate",
                             args=[None, dict(frame=dict(duration=400, redraw=True),
                                             fromcurrent=True, mode="immediate")]),
                        dict(label="⏸️ Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode="immediate")])
                    ],
                    direction="left", pad=dict(r=10, t=30), x=0.1, y=-0.1
                )],
                sliders=[dict(
                    steps=[dict(method="animate",
                               args=[[f"scan_{i}"], dict(mode="immediate", frame=dict(duration=400, redraw=True))],
                               label=f"{df_scan['param'].iloc[i]}")
                           for i in range(len(df_scan))],
                    transition=dict(duration=200), x=0.1, y=-0.15, len=0.8,
                    currentvalue=dict(font=dict(size=12), prefix=f"{anim_param}: ", visible=True, xanchor="center")
                )]
            )
            st.plotly_chart(fig_scan, use_container_width=True)

            # Multi-metric radar-style comparison
            st.markdown("### 📊 Comparación Multi-Métrica de Escenarios")
            fig_radar_scan = make_subplots(
                rows=2, cols=2,
                subplot_titles=["Costo Total (M€)", "Energía No Servida (MWh)",
                               "Corredores Congestionados", "Utilización Promedio (%)"])

            fig_radar_scan.add_trace(go.Scatter(x=df_scan['param'], y=df_scan['cost'],
                                                mode='lines+markers', line=dict(color='#0066CC', width=3),
                                                marker=dict(size=10)), row=1, col=1)
            fig_radar_scan.add_trace(go.Scatter(x=df_scan['param'], y=df_scan['unserved'],
                                                mode='lines+markers', line=dict(color='#CC0000', width=3),
                                                marker=dict(size=10)), row=1, col=2)
            fig_radar_scan.add_trace(go.Scatter(x=df_scan['param'], y=df_scan['congested'],
                                                mode='lines+markers', line=dict(color='#FF6600', width=3),
                                                marker=dict(size=10)), row=2, col=1)
            fig_radar_scan.add_trace(go.Scatter(x=df_scan['param'], y=df_scan['avg_util'],
                                                mode='lines+markers', line=dict(color='#9933CC', width=3),
                                                marker=dict(size=10)), row=2, col=2)

            for i in range(1, 5):
                r, c = (i-1)//2 + 1, (i-1)%2 + 1
                fig_radar_scan.update_xaxes(title_text=anim_param, row=r, col=c)

            fig_radar_scan.update_layout(template="plotly_white", height=500,
                                        title_text=f"Impacto de {anim_param} en Todas las Métricas",
                                        showlegend=False)
            st.plotly_chart(fig_radar_scan, use_container_width=True)

            # Best/worst scenario summary
            best_idx = df_scan['cost'].idxmin()
            worst_idx = df_scan['cost'].idxmax()
            bs1, bs2 = st.columns(2)
            with bs1:
                st.success(f"🏆 **Mejor escenario**: {anim_param} = {df_scan.loc[best_idx, 'param']:.0f}\n\n"
                          f"Costo: €{df_scan.loc[best_idx, 'cost']:.1f}M | "
                          f"Congestión: {df_scan.loc[best_idx, 'congested']:.0f} corredores")
            with bs2:
                st.error(f"⚠️ **Peor escenario**: {anim_param} = {df_scan.loc[worst_idx, 'param']:.0f}\n\n"
                        f"Costo: €{df_scan.loc[worst_idx, 'cost']:.1f}M | "
                        f"Congestión: {df_scan.loc[worst_idx, 'congested']:.0f} corredores")

# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Análisis de Escenarios":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("📊 Análisis de Escenarios — Colombia 2025–2050")
    section_header("Proyecciones Interactivas con Viaje en el Tiempo",
                   "Comparación de escenarios con/sin Colectora, integración renovable, BESS. "
                   "**Use el slider de año** para ver el estado del sistema en cualquier momento.")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        dgr = st.slider("Crecimiento demanda anual (%):", 1.0, 5.0, 2.5, 0.5, key="sc_dg")
        re30 = st.slider("Meta RE 2030 (% gen):", 10, 50, 25, 5, key="sc_re30")
        re50 = st.slider("Meta RE 2050 (% gen):", 30, 80, 60, 5, key="sc_re50")
        cy = st.slider("Colectora operativa:", 2024, 2032, 2027, 1, key="sc_cy")
        bess_total = st.slider("BESS total instalado 2030 (GW):", 0.0, 5.0, 1.0, 0.5, key="sc_bess")
        st.markdown("### 🎯 Vista")
        sc_view = st.radio("Visualización:", ["📈 Proyección completa", "⏰ Viaje en el tiempo", "🕸️ Radar de escenarios", "💰 Inversión"], key="sc_view")

    years = np.arange(2025, 2051)
    ny = len(years)
    bd = 76.5 * (1 + dgr/100) ** (years - 2025)

    sc1 = np.zeros(ny); sc2 = np.zeros(ny); sc3 = np.zeros(ny); sc4 = np.zeros(ny)
    sc1c = np.zeros(ny); sc2c = np.zeros(ny); sc3c = np.zeros(ny); sc4c = np.zeros(ny)
    # Investment costs per scenario
    sc1_inv = np.zeros(ny); sc2_inv = np.zeros(ny); sc3_inv = np.zeros(ny); sc4_inv = np.zeros(ny)

    for i, y in enumerate(years):
        sc1[i] = 18.4 + 0.5 * (y - 2025)
        sc1c[i] = min(100, 15 + 3 * (y - 2025) * (dgr / 2.5))
        sc1_inv[i] = (y - 2025) * 50  # Status quo: minimal investment
        re_add = max(0, (re30 / 100 * 18.4 * max(0, min(1, (y - 2025) / 5))))
        sc2[i] = 18.4 + 0.8 * (y - 2025) + re_add
        sc2c[i] = max(5, sc1c[i] * 0.4) if y >= cy else sc1c[i]
        sc2_inv[i] = (y - 2025) * 150 + (174 if y >= cy else 0)  # Colectora + moderate expansion
        re_add3 = max(0, (re50 / 100 * 18.4 * max(0, min(1, (y - 2025) / 15))))
        sc3[i] = 18.4 + 1.2 * (y - 2025) + re_add3 * 1.5
        sc3c[i] = max(2, sc1c[i] * 0.2) if y >= cy else sc1c[i]
        sc3_inv[i] = (y - 2025) * 250 + (174 + 350) if y >= cy else (y - 2025) * 100  # Heavy RE + transmission
        bess_effect = max(0, bess_total * min(1, (y - 2025) / 5)) * 2
        sc4[i] = sc3[i] + bess_total * min(1, (y - 2025) / 5)
        sc4c[i] = max(1, sc3c[i] - bess_effect) if y >= cy else sc1c[i]
        sc4_inv[i] = sc3_inv[i] + bess_total * 200 * min(1, (y - 2025) / 5)  # RE + BESS cost

    # ─── VIEW 1: Full Projection ───
    if sc_view == "📈 Proyección completa":
        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "Capacidad Instalada (GW)", "Índice de Congestión (%)",
            "Costo Congestión Anual (M€)", "Demanda (TWh)"])
        for sc, nm, cl in [(sc1, "Status Quo", "#CC0000"), (sc2, "Con Colectora", "#0066CC"),
                            (sc3, "RE Acelerado", "#00CC66"), (sc4, "RE + BESS", "#9933CC")]:
            fig.add_trace(go.Scatter(x=years, y=sc, name=nm, line=dict(color=cl, width=2)), row=1, col=1)
        for sc, nm, cl in [(sc1c, "Status Quo", "#CC0000"), (sc2c, "Con Colectora", "#0066CC"),
                            (sc3c, "RE Acelerado", "#00CC66"), (sc4c, "RE + BESS", "#9933CC")]:
            fig.add_trace(go.Scatter(x=years, y=sc, name=nm, line=dict(color=cl, width=2), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=years, y=sc * 45, name=nm, line=dict(color=cl, width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=bd, name="Demanda", line=dict(color='#333', width=2)), row=2, col=2)
        for r in [1, 2]:
            for c in [1, 2]:
                fig.add_vline(x=cy, line_dash="dash", line_color="orange", annotation_text="Colectora", row=r, col=c)
        fig.update_layout(template="plotly_white", height=600, title="Colombia SIN: Proyecciones 2025–2050")
        st.plotly_chart(fig, use_container_width=True)

    # ─── VIEW 2: Time Travel ───
    elif sc_view == "⏰ Viaje en el tiempo":
        selected_year = st.slider("🗓️ Seleccione año:", 2025, 2050, 2030, 1, key="sc_year")
        yi = selected_year - 2025

        st.markdown(f"### 📊 Estado del Sistema en {selected_year}")

        # Year-specific KPIs
        yk1, yk2, yk3, yk4 = st.columns(4)
        yk1.metric("📅 Año", selected_year)
        yk2.metric("⚡ Demanda Proyectada", f"{bd[yi]:.1f} TWh",
                   delta=f"+{bd[yi]-76.5:.1f} TWh vs 2025")
        yk3.metric("🔌 Colectora", "✅ Operativa" if selected_year >= cy else "⏳ En construcción",
                   delta=f"Desde {cy}")
        yk4.metric("📈 Crecimiento Acumulado", f"+{(1+dgr/100)**(selected_year-2025)*100-100:.0f}%",
                   delta=f"{dgr}% anual")

        st.markdown("---")

        # Comparison cards for selected year
        scenario_data = [
            ("Status Quo", sc1[yi], sc1c[yi], sc1_inv[yi], "#CC0000", "🔴"),
            ("Con Colectora", sc2[yi], sc2c[yi], sc2_inv[yi], "#0066CC", "🔵"),
            ("RE Acelerado", sc3[yi], sc3c[yi], sc3_inv[yi], "#00CC66", "🟢"),
            ("RE + BESS", sc4[yi], sc4c[yi], sc4_inv[yi], "#9933CC", "🟣"),
        ]

        cards = st.columns(4)
        for col, (nm, cap, cong, inv, cl, icon) in zip(cards, scenario_data):
            col.markdown(f"""<div style='background:{cl}08;padding:16px;border-radius:12px;border-left:5px solid {cl};'>
            <b style='color:{cl};font-size:1.1em;'>{icon} {nm}</b><br/><br/>
            <b>Capacidad:</b> {cap:.1f} GW<br/>
            <b>Congestión:</b> {cong:.1f}%<br/>
            <b>Inversión:</b> €{inv:.0f}M<br/>
            <b>Demanda cubierta:</b> {"✅" if cap * 4.5 > bd[yi] else "⚠️"} ({cap*4.5/bd[yi]*100:.0f}%)</div>""",
                        unsafe_allow_html=True)

        # Year-specific charts
        col1, col2 = st.columns(2)
        with col1:
            fig_year_cap = go.Figure(go.Bar(
                x=[s[0] for s in scenario_data],
                y=[s[1] for s in scenario_data],
                marker_color=[s[4] for s in scenario_data],
                text=[f"{s[1]:.1f} GW" for s in scenario_data], textposition="outside"))
            fig_year_cap.update_layout(template="plotly_white", height=350,
                                      yaxis_title="GW", title=f"Capacidad en {selected_year}")
            st.plotly_chart(fig_year_cap, use_container_width=True)
        with col2:
            fig_year_cong = go.Figure(go.Bar(
                x=[s[0] for s in scenario_data],
                y=[s[2] for s in scenario_data],
                marker_color=[
                    '#FF0000' if s[2] > 60 else '#FF6600' if s[2] > 30 else '#FFAA00' if s[2] > 10 else '#00CC66'
                    for s in scenario_data],
                text=[f"{s[2]:.1f}%" for s in scenario_data], textposition="outside"))
            fig_year_cong.update_layout(template="plotly_white", height=350,
                                       yaxis_title="%", title=f"Congestión en {selected_year}")
            st.plotly_chart(fig_year_cong, use_container_width=True)

        # Animated timeline marker
        fig_tl = go.Figure()
        for sc_data, nm, cl in [(sc1c, "Status Quo", "#CC0000"), (sc2c, "Con Colectora", "#0066CC"),
                                (sc3c, "RE Acelerado", "#00CC66"), (sc4c, "RE + BESS", "#9933CC")]:
            fig_tl.add_trace(go.Scatter(x=years, y=sc_data, name=nm, line=dict(color=cl, width=2)))
        fig_tl.add_vline(x=selected_year, line_color="black", line_width=3,
                        annotation_text=f"📍 {selected_year}")
        fig_tl.add_vline(x=cy, line_dash="dash", line_color="orange", annotation_text="Colectora")
        fig_tl.update_layout(template="plotly_white", height=300, yaxis_title="% Congestión",
                            title="Posición temporal — mueva el slider para viajar")
        st.plotly_chart(fig_tl, use_container_width=True)

    # ─── VIEW 3: Radar Chart ───
    elif sc_view == "🕸️ Radar de escenarios":
        st.subheader("🕸️ Comparación Multi-Criterio de Escenarios (2050)")
        st.markdown("*Normalizado 0-100. Mayor = mejor desempeño en esa métrica.*")

        # Normalize metrics for radar (higher is better)
        radar_scenarios = {
            "Status Quo": {
                "Capacidad": min(100, sc1[-1] / 50 * 100),
                "Baja Congestión": max(0, 100 - sc1c[-1]),
                "Bajo Costo": max(0, 100 - sc1[-1] * 45 / 100),
                "RE %": 35,
                "Confiabilidad": max(0, 100 - sc1c[-1] * 0.8),
                "BESS": 0,
            },
            "Con Colectora": {
                "Capacidad": min(100, sc2[-1] / 50 * 100),
                "Baja Congestión": max(0, 100 - sc2c[-1]),
                "Bajo Costo": max(0, 100 - sc2[-1] * 45 / 80),
                "RE %": 50,
                "Confiabilidad": max(0, 100 - sc2c[-1] * 0.8),
                "BESS": 20,
            },
            "RE Acelerado": {
                "Capacidad": min(100, sc3[-1] / 50 * 100),
                "Baja Congestión": max(0, 100 - sc3c[-1]),
                "Bajo Costo": max(0, 100 - sc3[-1] * 45 / 60),
                "RE %": re50,
                "Confiabilidad": max(0, 100 - sc3c[-1] * 0.8),
                "BESS": 30,
            },
            "RE + BESS": {
                "Capacidad": min(100, sc4[-1] / 50 * 100),
                "Baja Congestión": max(0, 100 - sc4c[-1]),
                "Bajo Costo": max(0, 100 - sc4[-1] * 45 / 50),
                "RE %": re50 + 10,
                "Confiabilidad": max(0, 100 - sc4c[-1] * 0.8),
                "BESS": min(100, bess_total * 20),
            },
        }

        categories = list(radar_scenarios["Status Quo"].keys())
        fig_radar = go.Figure()
        colors_radar = ["#CC0000", "#0066CC", "#00CC66", "#9933CC"]
        for (nm, data), cl in zip(radar_scenarios.items(), colors_radar):
            vals = list(data.values()) + [list(data.values())[0]]  # close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=nm, line=dict(color=cl, width=2), fill='toself', opacity=0.3))
        fig_radar.update_layout(
            template="plotly_white", height=500,
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Radar Multi-Criterio de Escenarios (2050)")
        st.plotly_chart(fig_radar, use_container_width=True)

        # Scenario selector for details
        selected_scenario = st.selectbox("Seleccione escenario para detalles:",
                                        list(radar_scenarios.keys()), key="sc_radar_sel")
        sel_data = radar_scenarios[selected_scenario]
        rcols = st.columns(len(categories))
        for col, (cat, val) in zip(rcols, sel_data.items()):
            col.metric(cat, f"{val:.0f}/100")

    # ─── VIEW 4: Investment Breakdown ───
    elif sc_view == "💰 Inversión":
        st.subheader("💰 Desglose de Inversión por Escenario (2025–2050)")

        # Investment breakdown components
        inv_components = {
            "Status Quo": {"Transmisión": 500, "Generación": 800, "BESS": 0, "Otros": 200},
            "Con Colectora": {"Transmisión": 1500, "Generación": 1200, "BESS": 100, "Otros": 300},
            "RE Acelerado": {"Transmisión": 2500, "Generación": 2800, "BESS": 200, "Otros": 400},
            "RE + BESS": {"Transmisión": 2500, "Generación": 2800, "BESS": bess_total * 200 * 4, "Otros": 400},
        }

        fig_inv = go.Figure()
        comp_colors = {"Transmisión": "#0066CC", "Generación": "#00CC66", "BESS": "#9933CC", "Otros": "#999999"}
        for comp, color in comp_colors.items():
            fig_inv.add_trace(go.Bar(
                x=list(inv_components.keys()),
                y=[inv_components[s][comp] for s in inv_components],
                name=comp, marker_color=color))
        fig_inv.update_layout(template="plotly_white", height=400, barmode='stack',
                             yaxis_title="Inversión (M€)", title="Desglose de Inversión por Componente (2025-2050)")
        st.plotly_chart(fig_inv, use_container_width=True)

        # Total investment comparison
        st.markdown("**Inversión Total por Escenario:**")
        ic1, ic2, ic3, ic4 = st.columns(4)
        scenario_colors = ["#CC0000", "#0066CC", "#00CC66", "#9933CC"]
        for col, (nm, data), cl in zip([ic1, ic2, ic3, ic4], inv_components.items(), scenario_colors):
            total = sum(data.values())
            col.markdown(f"""<div style='background:{cl}10;padding:16px;border-radius:12px;border-left:5px solid {cl};'>
            <b style='color:{cl};'>{nm}</b><br/>
            <span style='font-size:1.5em;color:{cl};'>€{total/1000:.1f}B</span><br/>
            <span style='font-size:0.8em;'>Total 2025-2050</span></div>""", unsafe_allow_html=True)

        # ROI analysis
        st.markdown("---")
        st.markdown("**📈 Retorno de Inversión (Ahorro por reducción de congestión):**")
        base_cong_cost = 4500  # M€/year baseline
        roi_data = []
        for nm, (cong_arr, inv_arr) in zip(["Status Quo", "Con Colectora", "RE Acelerado", "RE + BESS"],
                                           [(sc1c, sc1_inv), (sc2c, sc2_inv), (sc3c, sc3_inv), (sc4c, sc4_inv)]):
            total_cong_savings = sum(base_cong_cost * (1 - c/100) for c in cong_arr) - sum(base_cong_cost * (1 - sc1c[i]/100) for i, c in enumerate(sc1c))
            total_inv = inv_arr[-1]
            roi = total_cong_savings / max(total_inv, 1) * 100
            roi_data.append({"Escenario": nm, "Ahorro Congestión (M€)": f"{total_cong_savings:,.0f}",
                            "Inversión Total (M€)": f"{total_inv:,.0f}", "ROI": f"{roi:.0f}%"})
        st.dataframe(pd.DataFrame(roi_data), use_container_width=True, hide_index=True)

    # 2050 comparison cards (always show)
    st.markdown("---")
    st.subheader("📊 Comparación 2050")
    cols = st.columns(4)
    for col, (nm, cap, cong, cl) in zip(cols, [
        ("Status Quo", sc1[-1], sc1c[-1], "#CC0000"), ("Con Colectora", sc2[-1], sc2c[-1], "#0066CC"),
        ("RE Acelerado", sc3[-1], sc3c[-1], "#00CC66"), ("RE + BESS", sc4[-1], sc4c[-1], "#9933CC")]):
        col.markdown(f"""<div style='background:{cl}08;padding:16px;border-radius:12px;border-left:5px solid {cl};'>
        <b style='color:{cl};font-size:1.1em;'>{nm}</b><br/><br/>
        <b>Capacidad 2050:</b> {cap:.1f} GW<br/><b>Congestión:</b> {cong:.1f}%<br/>
        <b>Demanda:</b> {bd[-1]:.1f} TWh</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  COLOMBIAN DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════
elif page == "🗄️ Datos Colombia":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("🗄️ Fuentes de Datos — Infraestructura Eléctrica Colombiana")
    section_header("Catálogo de Datos Abiertos y GIS (Awesome-Electrical-Grid-Mapping)",
                   "Datasets oficiales de UPME, EPM, CREG y XM para análisis del SIN. "
                   "Curados por la iniciativa MapYourGrid / open-energy-transition.")

    st.markdown("""
    ### 🌐 Recursos Clave
    
    El repositorio [Awesome-Electrical-Grid-Mapping](https://github.com/open-energy-transition/Awesome-Electrical-Grid-Mapping) 
    mantiene un catálogo curado de mapas, datasets y recursos de redes eléctricas para (casi) todos los países del mundo.
    """)

    # ─── INTERACTIVE SEARCH/FILTER ───
    st.subheader("🔍 Buscar y Filtrar Datasets")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        search_text = st.text_input("🔎 Buscar por nombre:", "", key="ds_search",
                                    placeholder="Ej: UPME, EPM, transmisión...")
    with fc2:
        filter_type = st.multiselect("📂 Tipo de recurso:",
                                     ["Reporte", "Dataset + Mapa", "Mapa interactivo", "Portal de datos", "Regulación"],
                                     default=[], key="ds_type")
    with fc3:
        filter_year = st.multiselect("📅 Año:",
                                     ["2024", "2025", "2026"],
                                     default=[], key="ds_year")

    # Filter datasets
    filtered_ds = CO_DATASETS
    if search_text:
        filtered_ds = [ds for ds in filtered_ds if search_text.lower() in ds["name"].lower() or search_text.lower() in ds["desc"].lower()]
    if filter_type:
        filtered_ds = [ds for ds in filtered_ds if ds["type"] in filter_type]
    if filter_year:
        filtered_ds = [ds for ds in filtered_ds if ds["year"] in filter_year]

    st.info(f"📊 Mostrando **{len(filtered_ds)}** de {len(CO_DATASETS)} datasets")

    for ds in filtered_ds:
        st.markdown(f"""
        <div style='background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:8px;
                    border-left:4px solid #003399;'>
        <b>📌 {ds['name']}</b>
        <span style='background:#00339920;padding:2px 6px;border-radius:4px;font-size:0.75em;
                     margin-left:8px;'>{ds['type']}</span>
        <span style='background:#00660020;padding:2px 6px;border-radius:4px;font-size:0.75em;
                     margin-left:4px;'>{ds['year']}</span>
        <br/><span style='font-size:0.85em;color:#555;'>{ds['desc']}</span><br/>
        <span style='font-size:0.75em;'>Tags: {ds['tags']}</span><br/>
        <a href="{ds['url']}" target="_blank" style='font-size:0.8em;'>🔗 Abrir recurso</a>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Resumen de Cobertura")
    cov_data = pd.DataFrame({
        "Tipo": ["Líneas de transmisión", "Subestaciones", "Distribución (media tensión)",
                 "Distribución (baja tensión)", "Torres y postes", "Datos de operación"],
        "Fuente": ["UPME (ArcGIS)", "UPME (ArcGIS)", "EPM (ArcGIS)", "EPM (ArcGIS)", "EPM (ArcGIS)", "XM"],
        "Cobertura": ["Nacional (STN+STR)", "Nacional", "Antioquia", "Antioquia", "Antioquia", "Nacional"],
        "Formato": ["GIS Shapefile", "GIS Shapefile", "FeatureServer", "FeatureServer", "FeatureServer", "Portal web"],
        "Acceso": ["CC-BY-2.0", "Público", "Público", "Público", "Público", "Público"],
    })
    st.dataframe(cov_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🗺️ Iniciativas de Mapeo Comunitario")
    st.markdown("""
    - **[MapYourGrid](https://mapyourgrid.org/)** — Iniciativa comunitaria para mapear infraestructura de transmisión en OSM
    - **[Open Infrastructure Map](https://openinframap.org/)** — Visualización global de infraestructura energética en OSM
    - **[GridFinder](https://gridfinder.rdrn.me/)** — Detección automática de redes eléctricas desde imágenes satelitales
    - **[Global Energy Monitor](https://globalenergymonitor.org/)** — Base de datos global de plantas de energía
    
    **Contribuir:** La cobertura de OSM en Colombia puede mejorarse mapeando infraestructura de transmisión directamente.
    El [Starter Kit](https://mapyourgrid.org/starter-kit/) de MapYourGrid proporciona tutoriales para comenzar.
    """)

# ═══════════════════════════════════════════════════════════════════════
#  POLICY & REGULATION
# ═══════════════════════════════════════════════════════════════════════
elif page == "📋 Política y Regulación":
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.header("📋 Marco Político y Regulatorio")
    section_header("UPME, CREG y el Camino hacia la Resolución de Congestión",
                   "Hitos regulatorios, reformas y recomendaciones para la congestión del SIN.")

    st.subheader("📅 Cronología Regulatoria")
    pe = pd.DataFrame({
        "Fecha": ["2022", "2023", "2024", "May 2025", "Nov 2025", "Abr 2026", "May 2026", "Ago 2026"],
        "Evento": ["Meta original Colectora", "CREG reconoce agotamiento SIN", "Moratoria conexiones",
                   "CREG Res. 101 094: régimen transitorio", "17 proyectos entran SIN",
                   "Almacenamiento reconocido como alivio congestión", "UPME Res. 000358: vía rápida",
                   "Colectora 90%; 7 parques sin construir"],
        "Tipo": ["Hito", "Alerta", "Restricción", "Reforma", "Progreso", "Reforma", "Reforma", "Progreso"],
    })
    tc = {"Hito": "#0066CC", "Alerta": "#CC0000", "Restricción": "#FF6600", "Reforma": "#00CC66", "Progreso": "#0099CC"}
    fig = go.Figure()
    for _, row in pe.iterrows():
        fig.add_trace(go.Scatter(x=[row["Fecha"]], y=[1], mode='markers+text',
                                 marker=dict(size=15, color=tc.get(row["Tipo"], "#999")),
                                 text=[f"{row['Fecha']}<br>{row['Evento']}"], textposition="top center",
                                 textfont=dict(size=8), showlegend=False))
    fig.update_layout(template="plotly_white", height=250, xaxis_title="Cronología",
                     yaxis=dict(visible=False), title="Eventos Clave de Política y Regulación")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📜 UPME Resolución 000358 (May 2026)")
        st.markdown("""- Vía rápida para proyectos con obligaciones de energía
        - 5 días hábiles para información inicial
        - 5 días para verificación documental
        - 15 días para información adicional
        - Criterios técnicos actualizados de robustez de red""")
    with c2:
        st.subheader("📜 CREG Resolución 101 094 (2025)")
        st.markdown("""- Régimen transitorio de asignación de capacidad
        - UPME define procedimientos detallados
        - Reconoce agotamiento casi total del SIN
        - 8.9 GW solar + 2.5 GW eólico aprobados
        - 11 GW solicitudes pendientes""")

    st.markdown("---")
    st.subheader("💡 Recomendaciones de Política")
    st.markdown("""
    1. **Acelerar Colectora** — Operación comercial máximo Q2 2027
    2. **Planear Colectora II** — Segunda línea 500 kV (2 GW) desde La Guajira
    3. **Implementar precios nodales (LMP)** — Eliminar gestión formal de congestión
    4. **BESS obligatorios en nodos congestionados** — La Guajira y Bogotá
    5. **Agilizar consultas indígenas** — Corredores pre-aprobados
    6. **Grid-Enhancing Technologies (GETs)** — DLR, FACTS, optimización topológica
    7. **Reporte transparente de congestión** — Costos anuales por corredor (como RTOs en EE.UU.)
    8. **Planeación integrada generación-transmisión** — Co-optimizar con modelos LOPF""")

    # ─── INTERACTIVE POLICY IMPACT SIMULATOR ───
    st.markdown("---")
    st.subheader("🎯 Simulador de Impacto de Políticas")
    st.markdown("*Active cada política y vea el impacto estimado en congestión, inversión y emisiones:*")

    policies = [
        {"name": "Acelerar Colectora", "congestion": -35, "investment": 174, "co2": -800, "icon": "🔌"},
        {"name": "Colectora II (2 GW)", "congestion": -20, "investment": 350, "co2": -1200, "icon": "🔌"},
        {"name": "Precios Nodales (LMP)", "congestion": -15, "investment": 50, "co2": -200, "icon": "💰"},
        {"name": "BESS Obligatorios (500 MW)", "congestion": -10, "investment": 200, "co2": -300, "icon": "🔋"},
        {"name": "Consultas Ágiles", "congestion": -5, "investment": 20, "co2": -100, "icon": "📋"},
        {"name": "GETs (DLR + FACTS)", "congestion": -8, "investment": 80, "co2": -150, "icon": "⚡"},
        {"name": "Reporte Transparente", "congestion": -2, "investment": 5, "co2": 0, "icon": "📊"},
        {"name": "Planeación Integrada", "congestion": -12, "investment": 30, "co2": -400, "icon": "🗺️"},
    ]

    # Policy toggles
    pol_cols = st.columns(4)
    active_policies = []
    for i, pol in enumerate(policies):
        with pol_cols[i % 4]:
            is_active = st.checkbox(
                f"{pol['icon']} {pol['name']}",
                value=(i < 2),  # First two active by default
                key=f"pol_{i}",
                help=f"Congestión: {pol['congestion']:+.0f}% | Inversión: €{pol['investment']}M | CO₂: {pol['co2']:+.0f} kt")
            if is_active:
                active_policies.append(pol)

    # Calculate cumulative impact
    total_cong = sum(p["congestion"] for p in active_policies)
    total_inv = sum(p["investment"] for p in active_policies)
    total_co2 = sum(p["co2"] for p in active_policies)

    base_congestion = 65  # base congestion index
    new_congestion = max(5, base_congestion + total_cong)

    pi1, pi2, pi3, pi4 = st.columns(4)
    pi1.metric("Índice Congestión", f"{new_congestion:.0f}",
               delta=f"{total_cong:+.0f} puntos" if active_policies else "Base",
               delta_color="normal" if total_cong < 0 else "inverse")
    pi2.metric("Inversión Requerida", f"€{total_inv}M",
               delta=f"{len(active_policies)} políticas activas")
    pi3.metric("Reducción CO₂", f"{total_co2:+.0f} kt/año",
               delta="Positivo" if total_co2 < 0 else "Sin impacto",
               delta_color="normal" if total_co2 < 0 else "inverse")
    pi4.metric("Políticas Activas", f"{len(active_policies)}/8",
               delta="Combinación seleccionada")

    # Impact visualization
    fig_pol = go.Figure()
    fig_pol.add_trace(go.Bar(
        x=[p["name"] for p in policies],
        y=[p["congestion"] if p in active_policies else 0 for p in policies],
        marker_color=['#00CC66' if p in active_policies else '#DDDDDD' for p in policies],
        text=[f"{p['congestion']:+.0f}%" if p in active_policies else "Inactivo" for p in policies],
        textposition="outside"))
    fig_pol.update_layout(template="plotly_white", height=350, yaxis_title="Reducción Congestión (%)",
                         title="Impacto por Política (activas en verde)")
    st.plotly_chart(fig_pol, use_container_width=True)

    st.markdown("---")
    st.subheader("📚 Fuentes")
    st.markdown("""
    - UPME Resolución 000358 de 2026 | CREG Resolución 101 094 de 2025
    - SER Colombia — Estudio Precios Nodales (2022)
    - "Modeling Infrastructure Delays and Congestion" (Energies, 2025)
    - "100% renewable electricity in Colombia by 2030" (ScienceDirect, 2025)
    - XM — Operador del Sistema | Grid2Poster | Awesome-Electrical-Grid-Mapping
    - HiGHS: solver LP/MILP de alto rendimiento — highs.dev
    - SciPy: scipy.optimize.linprog ( interfaz a HiGHS ) — scipy.org""")
