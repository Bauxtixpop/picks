import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
from scraper import obtener_tabla_ligamx
from engine import calcular_idr

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Apuestas-Futbol & MLB",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Profesionales
st.markdown("""
<style>
    .match-header { background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%); padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #3b82f6; margin-bottom: 20px; }
    .match-header-mlb { background: linear-gradient(90deg, #7f1d1d 0%, #172554 100%); padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #ef4444; margin-bottom: 20px; }
    .pitcher-box { background: #0f172a; padding: 15px; border-radius: 10px; border: 1px dashed #38bdf8; margin-bottom: 20px; }
    .safe-card { background: #1e3a8a; padding: 15px; border-radius: 10px; border-left: 5px solid #60a5fa; margin-bottom: 10px; }
    .value-card { background: #064e3b; padding: 15px; border-radius: 10px; border-left: 5px solid #10b981; margin-bottom: 10px; }
    .risk-card { background: #7c2d12; padding: 15px; border-radius: 10px; border-left: 5px solid #f97316; margin-bottom: 10px; }
    .parlay-card { background: #4c1d95; padding: 15px; border-radius: 10px; border-left: 5px solid #a78bfa; margin-bottom: 10px; }
    .dream-card { background: linear-gradient(135deg, #831843 0%, #311042 100%); padding: 22px; border-radius: 12px; border: 2px solid #ec4899; margin-top: 20px; box-shadow: 0 4px 20px rgba(236, 72, 153, 0.3); }
    .meta-model-card { background: linear-gradient(135deg, #b91c1c 0%, #450a0a 100%); padding: 20px; border-radius: 12px; border: 2px solid #f87171; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    .method-box { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #475569; margin-bottom: 10px; }
    .stat-box { background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; text-align: center; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# 2. FUNCIONES DE CONVERSIÓN DE MOMIOS
def americano_a_decimal(momio_amer):
    if momio_amer >= 0: return round((momio_amer / 100.0) + 1.0, 2)
    else: return round((100.0 / abs(momio_amer)) + 1.0, 2)

def prob_implicada(decimal_odd):
    return round((1.0 / decimal_odd) * 100.0, 1)

# ==============================================================================
# ====== 🧠 MOTOR GLOBAL DE FÚTBOL (COMPARTIDO LIGA MX & LEAGUES CUP) ==========
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_datos_completos():
    df = obtener_tabla_ligamx()
    df = calcular_idr(df)
    return df

def ejecutar_laboratorio_modelos(local, visita, df):
    # 🛡️ 1. Búsqueda Inteligente (Flexible) para evitar errores de nombres
    def buscar_equipo(nombre, df_tabla):
        if df_tabla.empty or 'Equipo' not in df_tabla.columns: return None
        m = df_tabla[df_tabla['Equipo'].astype(str).str.contains(nombre, case=False, na=False)]
        if not m.empty: return m.iloc[0].to_dict()
        m = df_tabla[df_tabla['Equipo'].astype(str).str.contains(nombre.split()[-1], case=False, na=False)]
        if not m.empty: return m.iloc[0].to_dict()
        m = df_tabla[df_tabla['Equipo'].astype(str).str.contains(nombre.split()[0], case=False, na=False)]
        if not m.empty: return m.iloc[0].to_dict()
        return None

    s_loc = buscar_equipo(local, df)
    if s_loc:
        s_loc['Equipo'] = local
    else:
        s_loc = {"Equipo": local, "PJ": 1, "Pts": 1.5, "GF": 1.5, "GC": 1.5, "xG": 1.4, "xGA": 1.4, "IDR": 50.0, "AttPen_Promedio": 15.0, "Tiros_Promedio": 11.0, "Calidad_Tiro": 0.10}

    s_vis = buscar_equipo(visita, df)
    if s_vis:
        s_vis['Equipo'] = visita
    else:
        s_vis = {"Equipo": visita, "PJ": 1, "Pts": 1.5, "GF": 1.5, "GC": 1.5, "xG": 1.4, "xGA": 1.4, "IDR": 50.0, "AttPen_Promedio": 15.0, "Tiros_Promedio": 11.0, "Calidad_Tiro": 0.10}

    xg_prom = df['xG'].mean() / 10.0 if not df.empty and df['xG'].mean() > 0 else 1.3
    pj_l = max(s_loc.get('PJ', 1), 1)
    pj_v = max(s_vis.get('PJ', 1), 1)
    
    # MÉTODO 1: Distribución de Poisson Ajustada
    atq_l = (s_loc['xG'] / pj_l) / xg_prom
    def_l = (s_loc['xGA'] / pj_l) / xg_prom
    atq_v = (s_vis['xG'] / pj_v) / xg_prom
    def_v = (s_vis['xGA'] / pj_v) / xg_prom
    
    lam_l = xg_prom * atq_l * def_v * 1.15
    lam_v = xg_prom * atq_v * def_l
    
    p1_pois, px_pois, p2_pois = 0.0, 0.0, 0.0
    o25_pois, btts_pois = 0.0, 0.0
    marcadores = []
    
    for gl in range(6):
        for gv in range(6):
            prob = stats.poisson.pmf(gl, lam_l) * stats.poisson.pmf(gv, lam_v)
            marcadores.append({"Marcador": f"{gl} - {gv}", "Prob": prob * 100})
            if gl > gv: p1_pois += prob
            elif gl == gv: px_pois += prob
            else: p2_pois += prob
            if (gl + gv) > 2.5: o25_pois += prob
            if gl > 0 and gv > 0: btts_pois += prob
            
    tot_pois = p1_pois + px_pois + p2_pois
    m1 = [round((p1_pois/tot_pois)*100, 1), round((px_pois/tot_pois)*100, 1), round((p2_pois/tot_pois)*100, 1)]
    marcadores_top = pd.DataFrame(marcadores).sort_values(by="Prob", ascending=False).head(3).to_dict(orient="records")

    # MÉTODO 2: Rating ELO Dinámico
    elo_l = 1500 + (s_loc['Pts'] * 15) + (s_loc['IDR'] * 2) + 35
    elo_v = 1500 + (s_vis['Pts'] * 15) + (s_vis['IDR'] * 2)
    diff_elo = elo_v - elo_l
    prob_elo_l = 1.0 / (1.0 + 10.0 ** (diff_elo / 400.0))
    prob_elo_v = 1.0 - prob_elo_l
    px_elo = max(18.0, 30.0 - (abs(diff_elo) * 0.08))
    rem_elo = 100.0 - px_elo
    m2 = [round(prob_elo_l * rem_elo, 1), round(px_elo, 1), round(prob_elo_v * rem_elo, 1)]

    # MÉTODO 3: Simulación Monte Carlo
    np.random.seed(42)
    sims_l = np.random.poisson(lam_l, 5000)
    sims_v = np.random.poisson(lam_v, 5000)
    wins_l = np.sum(sims_l > sims_v)
    draws = np.sum(sims_l == sims_v)
    wins_v = np.sum(sims_l < sims_v)
    m3 = [round((wins_l/5000)*100, 1), round((draws/5000)*100, 1), round((wins_v/5000)*100, 1)]

    # MÉTODO 4: Índice de Dominio Real (IDR Táctico)
    fuerza_idr_l = max(s_loc['IDR'] + (s_loc['AttPen_Promedio'] * 1.5), 10) * 1.10
    fuerza_idr_v = max(s_vis['IDR'] + (s_vis['AttPen_Promedio'] * 1.5), 10)
    tot_idr = fuerza_idr_l + fuerza_idr_v
    p1_idr = (fuerza_idr_l / tot_idr) * 75.0
    p2_idr = (fuerza_idr_v / tot_idr) * 75.0
    px_idr = 100.0 - (p1_idr + p2_idr)
    m4 = [round(p1_idr, 1), round(px_idr, 1), round(p2_idr, 1)]

    # MÉTODO 5: Forma del Torneo
    ef_l = ((s_loc['Pts']/pj_l) * 25.0) + ((s_loc['GF'] - s_loc['GC']) * 2.0) + 15.0
    ef_v = ((s_vis['Pts']/pj_v) * 25.0) + ((s_vis['GF'] - s_vis['GC']) * 2.0)
    ef_l = max(ef_l, 5.0)
    ef_v = max(ef_v, 5.0)
    tot_ef = ef_l + ef_v
    p1_form = (ef_l / tot_ef) * 72.0
    p2_form = (ef_v / tot_ef) * 72.0
    px_form = 100.0 - (p1_form + p2_form)
    m5 = [round(p1_form, 1), round(px_form, 1), round(p2_form, 1)]

    # 👑 CONSENSO META-MODELO
    cons_1 = round((m1[0]*0.25 + m2[0]*0.15 + m3[0]*0.20 + m4[0]*0.15 + m5[0]*0.25), 1)
    cons_x = round((m1[1]*0.25 + m2[1]*0.15 + m3[1]*0.20 + m4[1]*0.15 + m5[1]*0.25), 1)
    cons_2 = round((m1[2]*0.25 + m2[2]*0.15 + m3[2]*0.20 + m4[2]*0.15 + m5[2]*0.25), 1)

    tiros_l, tiros_v = s_loc.get('Tiros_Promedio', 12.0), s_vis.get('Tiros_Promedio', 10.0)
    att_l, att_v = s_loc.get('AttPen_Promedio', 18.0), s_vis.get('AttPen_Promedio', 14.0)
    corners_total = round((tiros_l * 0.25) + (att_l * 0.08) + (tiros_v * 0.20) + (att_v * 0.06), 1)

    return {
        "Local": local, "Visita": visita,
        "xG_L": round(lam_l, 2), "xG_V": round(lam_v, 2),
        "IDR_L": s_loc['IDR'], "IDR_V": s_vis['IDR'],
        "Prob_1": cons_1, "Prob_X": cons_x, "Prob_2": cons_2,
        "Over_25": round(o25_pois*100, 1), "BTTS_Si": round(btts_pois*100, 1),
        "Corners_Total": corners_total, "Marcadores_Top": marcadores_top,
        "M1_Poisson": m1, "M2_ELO": m2, "M3_MonteCarlo": m3, "M4_IDR": m4, "M5_Forma": m5
    }

# 3. SELECTOR DE DEPORTE (MENÚ LATERAL)
st.sidebar.title("🏆 Centro de Mando")
deporte = st.sidebar.radio("Selecciona tu Motor de Análisis:", ["⚽ Fútbol (Liga MX)", "⚾ Béisbol (MLB)", "🌎 Leagues Cup"])
st.sidebar.markdown("---")
st.sidebar.info("💡 **Sistema Multi-Algoritmo:** Evalúa líneas en vivo usando Poisson, Monte Carlo, ELO y Modelos de Eficiencia.")

# ==============================================================================
# ================= SECCIÓN 1: FÚTBOL (LIGA MX) ================================
# ==============================================================================
if deporte == "⚽ Fútbol (Liga MX)":
    st.title("⚽ Liga MX - Cuantificador Multi-Algoritmo & Value Bets")
    
    with st.spinner("⚡ Conectando con estadísticas y procesando motores matemáticos..."):
        df_ligamx = cargar_datos_completos()

    # --- CONTROL DE JORNADA MANUAL (Evita bloqueos de pago de APIs) ---
    @st.cache_data(ttl=3600*6)
    def obtener_jornada_automatica():
        return [
            "Puebla vs Guadalajara",
            "San Luis vs Tijuana",
            "Juárez vs Pumas",
            "Querétaro vs Tigres",
            "Atlas vs Monterrey",
            "León vs Pachuca",
            "Cruz Azul vs Atlante",
            "América vs Santos",
            "Toluca vs Necaxa"
        ]

    partidos_jornada_default = obtener_jornada_automatica()

    tab_match, tab_lab, tab_jornada, tab_tabla = st.tabs([
        "🏟️ Match Center & Momios en Vivo", 
        "🧪 Laboratorio Multi-Algoritmo",
        "⚡ Parlays de la Jornada", 
        "📊 Ranking IDR"
    ])

    with tab_match:
        if len(partidos_jornada_default) > 0:
            partido_sel = st.selectbox("⚽ Selecciona el enfrentamiento para analizar:", options=partidos_jornada_default, index=0, key="sel_m1")
            eqs = partido_sel.split(" vs ")
            datos = ejecutar_laboratorio_modelos(eqs[0].strip(), eqs[1].strip(), df_ligamx)
            
            if datos:
                st.markdown(f"""
                <div class="match-header">
                    <h2>🏠 {datos['Local']} vs {datos['Visita']} ✈️</h2>
                    <p><b>Proyección de xG:</b> {datos['xG_L']} - {datos['xG_V']} &nbsp;|&nbsp; <b>Dominio IDR:</b> {datos['IDR_L']} vs {datos['IDR_V']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🏦 Ingresa las Cuotas de tu Casa de Apuestas (Formato Americano)")
                c_m1, c_mx, c_m2, c_mo25, c_mu25 = st.columns(5)
                with c_m1: amer_1 = st.number_input(f"Victoria {datos['Local']} (1)", value=-110, step=10, key="m1")
                with c_mx: amer_x = st.number_input("Empate (X)", value=+220, step=10, key="mx")
                with c_m2: amer_2 = st.number_input(f"Victoria {datos['Visita']} (2)", value=+180, step=10, key="m2")
                with c_mo25: amer_o25 = st.number_input("Over 2.5 Goles", value=-115, step=10, key="mo25")
                with c_mu25: amer_u25 = st.number_input("Under 2.5 Goles", value=-115, step=10, key="mu25")
                    
                dec_1, dec_x, dec_2 = americano_a_decimal(amer_1), americano_a_decimal(amer_x), americano_a_decimal(amer_2)
                dec_o25, dec_u25 = americano_a_decimal(amer_o25), americano_a_decimal(amer_u25)
                
                edge_1 = round(datos['Prob_1'] - prob_implicada(dec_1), 1)
                edge_x = round(datos['Prob_X'] - prob_implicada(dec_x), 1)
                edge_2 = round(datos['Prob_2'] - prob_implicada(dec_2), 1)
                edge_o25 = round(datos['Over_25'] - prob_implicada(dec_o25), 1)
                edge_u25 = round((100.0 - datos['Over_25']) - prob_implicada(dec_u25), 1)
                
                edges = {
                    f"Victoria {datos['Local']}": (edge_1, amer_1, dec_1, datos['Prob_1']),
                    "Empate": (edge_x, amer_x, dec_x, datos['Prob_X']),
                    f"Victoria {datos['Visita']}": (edge_2, amer_2, dec_2, datos['Prob_2']),
                    "Over 2.5 Goles": (edge_o25, amer_o25, dec_o25, datos['Over_25']),
                    "Under 2.5 Goles": (edge_u25, amer_u25, dec_u25, round(100.0-datos['Over_25'], 1))
                }
                mejor_val_nombre, (mejor_edge, m_amer_val, m_dec_val, m_prob_val) = max(edges.items(), key=lambda x: x[1][0])

                st.markdown("---")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    pick_seguro = f"1X ({datos['Local']} o Empate)" if datos['Prob_1'] >= datos['Prob_2'] else f"X2 ({datos['Visita']} o Empate)"
                    st.markdown(f'<div class="safe-card"><h3>🛡️ PICK SEGURO (Bajo Riesgo)</h3><p><b>Recomendación:</b> {pick_seguro}</p><p><b>Probabilidad del Modelo:</b> ~{round(datos["Prob_1"]+datos["Prob_X"] if datos["Prob_1"]>=datos["Prob_2"] else datos["Prob_2"]+datos["Prob_X"], 1)}%</p></div>', unsafe_allow_html=True)
                with col_p2:
                    if mejor_edge >= 5.0:
                        st.markdown(f'<div class="value-card"><h3>💎 PICK DE VALOR (Value Bet)</h3><p><b>Recomendación:</b> {mejor_val_nombre} (Momio: {m_amer_val:+} / {m_dec_val})</p><p style="color:#34d399;">🔥 EDGE: +{mejor_edge}%</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>Margen más alto: <b>{mejor_val_nombre}</b> (+{mejor_edge}%).</p></div>', unsafe_allow_html=True)

                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    top_marcador = datos['Marcadores_Top'][0]
                    st.markdown(f'<div class="risk-card"><h3>🔥 PICK RISK</h3><p>Marcador Exacto {top_marcador["Marcador"]} (Prob: {round(top_marcador["Prob"], 1)}%)</p></div>', unsafe_allow_html=True)
                with col_p4:
                    sgp_1 = f"{datos['Local']} 1X" if datos['Prob_1'] >= datos['Prob_2'] else f"{datos['Visita']} X2"
                    sgp_2 = "Over 1.5 Goles" if datos['Over_25'] > 45 else "Under 3.5 Goles"
                    sgp_3 = f"Over 8.5 Córners" if datos['Corners_Total'] > 9.0 else "Under 10.5 Córners"
                    st.markdown(f'<div class="parlay-card"><h3>🎰 PICK PARLAY</h3><p>{sgp_1} + {sgp_2} + {sgp_3}</p></div>', unsafe_allow_html=True)

    with tab_lab:
        st.subheader("🧪 Laboratorio Multi-Algoritmo")
        if len(partidos_jornada_default) > 0:
            part_lab = st.selectbox("🔬 Selecciona el partido:", options=partidos_jornada_default, key="sel_lab")
            eqs_lab = part_lab.split(" vs ")
            d_lab = ejecutar_laboratorio_modelos(eqs_lab[0].strip(), eqs_lab[1].strip(), df_ligamx)
            if d_lab:
                st.markdown(f'<div class="meta-model-card"><h3>👑 CONSENSO DEFINITIVO</h3><h1>{d_lab["Local"]}: {d_lab["Prob_1"]}% | X: {d_lab["Prob_X"]}% | {d_lab["Visita"]}: {d_lab["Prob_2"]}%</h1></div>', unsafe_allow_html=True)
                tabla_comparativa = {
                    "Modelo": ["Poisson", "ELO", "Monte Carlo", "IDR", "Forma"],
                    f"🏠 {d_lab['Local']}": [f"{d_lab['M1_Poisson'][0]}%", f"{d_lab['M2_ELO'][0]}%", f"{d_lab['M3_MonteCarlo'][0]}%", f"{d_lab['M4_IDR'][0]}%", f"{d_lab['M5_Forma'][0]}%"],
                    "🤝 Empate": [f"{d_lab['M1_Poisson'][1]}%", f"{d_lab['M2_ELO'][1]}%", f"{d_lab['M3_MonteCarlo'][1]}%", f"{d_lab['M4_IDR'][1]}%", f"{d_lab['M5_Forma'][1]}%"],
                    f"✈️ {d_lab['Visita']}": [f"{d_lab['M1_Poisson'][2]}%", f"{d_lab['M2_ELO'][2]}%", f"{d_lab['M3_MonteCarlo'][2]}%", f"{d_lab['M4_IDR'][2]}%", f"{d_lab['M5_Forma'][2]}%"]
                }
                st.dataframe(pd.DataFrame(tabla_comparativa), use_container_width=True)

    with tab_jornada:
        st.subheader("⚡ Boletos Combinados de la Jornada")
        jornada_data = [ejecutar_laboratorio_modelos(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_ligamx) for p in partidos_jornada_default]
        jornada_data = [d for d in jornada_data if d is not None]
            
        if len(jornada_data) > 0:
            col_j1, col_j2 = st.columns(2)
            with col_j1:
                picks_seg, cuota_tot_dec = "", 1.0
                for d in sorted(jornada_data, key=lambda x: abs(x['IDR_L']-x['IDR_V']), reverse=True)[:4]:
                    pick = f"1X ({d['Local']})" if d['IDR_L'] >= d['IDR_V'] else f"X2 ({d['Visita']})"
                    picks_seg += f"⚽ <b>{d['Local']} vs {d['Visita']}:</b> {pick}<br>"
                    cuota_tot_dec *= 1.25
                mom_amer_seg = int((cuota_tot_dec - 1.0) * 100) if cuota_tot_dec >= 2.0 else int(-100 / (cuota_tot_dec - 1.0))
                st.markdown(f'<div class="safe-card"><h3>🛡️ Parlay Seguro</h3>{picks_seg}<hr><h4>🎟️ Momio: {mom_amer_seg:+}</h4></div>', unsafe_allow_html=True)
            with col_j2:
                picks_gol, cuota_tot_gol = "", 1.0
                for d in sorted(jornada_data, key=lambda x: x['Over_25'], reverse=True)[:3]:
                    pick = "Over 2.5" if d['Over_25'] > 55 else "Ambos Anotan SÍ"
                    picks_gol += f"💥 <b>{d['Local']} vs {d['Visita']}:</b> {pick}<br>"
                    cuota_tot_gol *= 1.75
                mom_amer_gol = int((cuota_tot_gol - 1.0) * 100) if cuota_tot_gol >= 2.0 else int(-100 / (cuota_tot_gol - 1.0))
                st.markdown(f'<div class="value-card"><h3>💎 Parlay Goles</h3>{picks_gol}<hr><h4>🎟️ Momio: {mom_amer_gol:+}</h4></div>', unsafe_allow_html=True)

    with tab_tabla:
        st.subheader("📈 Ranking de Dominio Real (IDR)")
        df_show = df_ligamx[['Equipo', 'Pts', 'GF', 'GC', 'xG', 'IDR']].sort_values(by='IDR', ascending=False).reset_index(drop=True)
        df_show.index += 1
        st.dataframe(df_show.style.background_gradient(subset=['IDR'], cmap='viridis'), use_container_width=True)

# ==============================================================================
# ================= SECCIÓN 2: BÉISBOL (MLB) ===================================
# ==============================================================================
elif deporte == "⚾ Béisbol (MLB)":
    st.title("⚾ Proyector Cuantitativo & Moneyball - MLB")
    
    @st.cache_data(ttl=3600)
    def cargar_estadisticas_mlb():
        url_standings = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&hydrate=team"
        try:
            res = requests.get(url_standings, timeout=10)
            data = res.json()
            equipos_data = []
            for record in data.get("records", []):
                for team_rec in record.get("teamRecords", []):
                    nombre = team_rec["team"]["name"]
                    g, p = team_rec["wins"], team_rec["losses"]
                    rs, ra = team_rec.get("runsScored", 0), team_rec.get("runsAllowed", 0)
                    racha_str = team_rec.get("streak", {}).get("streakCode", "W1")
                    pitagorica = ((rs ** 1.83) / ((rs ** 1.83) + (ra ** 1.83))) * 100 if rs > 0 else 50.0
                    equipos_data.append({
                        "Equipo": nombre, "G": g, "P": p, "RS_prom": round(rs/(g+p) if (g+p)>0 else 4.5, 2),
                        "RA_prom": round(ra/(g+p) if (g+p)>0 else 4.5, 2), "ERA": round((ra/(g+p) if (g+p)>0 else 4.5) * 0.92, 2),
                        "Pitagorica": round(pitagorica, 2), "Racha": racha_str
                    })
            return pd.DataFrame(equipos_data)
        except: return pd.DataFrame()

    with st.spinner("⚾ Conectando a MLB..."):
        df_mlb = cargar_estadisticas_mlb()

    @st.cache_data(ttl=3600*3)
    def obtener_partidos_mlb(fecha_str):
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha_str}"
        try:
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                juegos = []
                for date_info in res.json().get("dates", []):
                    for game in date_info.get("games", []):
                        away, home = game["teams"]["away"]["team"]["name"], game["teams"]["home"]["team"]["name"]
                        juegos.append(f"{home} vs {away}")
                if juegos: return juegos
        except: pass
        return ["New York Yankees vs Los Angeles Dodgers", "Philadelphia Phillies vs Baltimore Orioles"]

    fecha_sel = st.date_input("📅 Selecciona Jornada MLB:", value=date(2026, 7, 20))
    partidos_mlb = obtener_partidos_mlb(fecha_sel.strftime("%Y-%m-%d"))

    def motor_mlb_360(local, visita, df, era_sp_l=None, era_sp_v=None, linea_ou=8.5):
        local_clean, visita_clean = local.split(" (Juego")[0].strip(), visita.split(" (Juego")[0].strip()
        sl = df[df['Equipo'].str.contains(local_clean.split()[-1], case=False, na=False)].iloc[0].to_dict() if not df.empty else {"Pitagorica": 50, "ERA": 4, "Racha": "W"}
        sv = df[df['Equipo'].str.contains(visita_clean.split()[-1], case=False, na=False)].iloc[0].to_dict() if not df.empty else {"Pitagorica": 50, "ERA": 4, "Racha": "W"}
        if era_sp_l is None: era_sp_l = sl['ERA']
        if era_sp_v is None: era_sp_v = sv['ERA']
        
        def_l, def_v = (era_sp_l * 0.65) + (sl['ERA'] * 0.35), (era_sp_v * 0.65) + (sv['ERA'] * 0.35)
        prob_l = sl['Pitagorica'] + (def_v - def_l)*4.0
        prob_v = sv['Pitagorica'] - (def_v - def_l)*4.0
        total = prob_l + prob_v
        p1, p2 = round((prob_l/total)*100, 1), round((prob_v/total)*100, 1) if total > 0 else (50.0, 50.0)
        
        return {"Local": local_clean, "Visita": visita_clean, "Prob_1": p1, "Prob_2": p2, "ER_L": round((def_v*1.05)*(p1/50), 1), "ER_V": round((def_l*1.05)*(p2/50), 1), "NRFI": max(10, min(90, 50 + (8-(era_sp_l+era_sp_v))*4)), "Over_Line": max(10, min(90, 50 + ((round((def_v*1.05)*(p1/50), 1)+round((def_l*1.05)*(p2/50), 1))-linea_ou)*5))}

    tab_mlb1, tab_mlb3 = st.tabs(["🏟️ Match Center (MLB)", "⚡ Parlays MLB"])
    with tab_mlb1:
        part_mlb_sel = st.selectbox("⚾ Selecciona Juego MLB:", partidos_mlb)
        eqs_m = part_mlb_sel.split(" vs ")
        c1, c2, c3 = st.columns(3)
        with c1: sp_l = st.number_input("ERA Abridor Local", value=4.0)
        with c2: sp_v = st.number_input("ERA Abridor Visita", value=4.0)
        with c3: ou_l = st.number_input("Línea O/U", value=8.5)
        dm = motor_mlb_360(eqs_m[0].strip(), eqs_m[1].strip(), df_mlb, sp_l, sp_v, ou_l)
        st.markdown(f'<div class="match-header-mlb"><h2>{dm["Local"]} vs {dm["Visita"]}</h2><p>Prob. ML: {dm["Prob_1"]}% vs {dm["Prob_2"]}% | Total Proyectado: {dm["ER_L"] + dm["ER_V"]}</p></div>', unsafe_allow_html=True)
    
    with tab_mlb3:
        st.subheader("⚡ Picks Automáticos MLB")
        data_j_mlb = [motor_mlb_360(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_mlb) for p in partidos_mlb]
        for x in sorted(data_j_mlb, key=lambda d: max(d['Prob_1'], d['Prob_2']), reverse=True)[:3]:
            fav = x['Local'] if x['Prob_1'] > x['Prob_2'] else x['Visita']
            st.markdown(f'<div class="safe-card">⚾ <b>{x["Local"]} vs {x["Visita"]}</b>: {fav} Gana</div>', unsafe_allow_html=True)

# ==============================================================================
# ================= SECCIÓN 3: LEAGUES CUP (PARLAYS & PICKS) ===================
# ==============================================================================
elif deporte == "🌎 Leagues Cup":
    st.title("🌎 Leagues Cup - Máquina de Parlays y Picks")
    st.write("Análisis masivo cruzando métricas de la MLS (FBref) y la Liga MX.")

    @st.cache_data(ttl=3600*24)
    def obtener_tabla_mls():
        url = "https://fbref.com/es/comps/22/Estadisticas-de-Major-League-Soccer"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            res = requests.get(url, headers=headers, timeout=12)
            if res.status_code != 200: return pd.DataFrame()
            tablas = pd.read_html(res.text)
            df_mls = pd.DataFrame()
            for t in tablas:
                if isinstance(t.columns, pd.MultiIndex): t.columns = t.columns.droplevel(0)
                if 'Equipo' in t.columns and 'Pts' in t.columns:
                    df_mls = pd.concat([df_mls, t], ignore_index=True)
            if df_mls.empty: return pd.DataFrame()
            df_mls = df_mls.dropna(subset=['Equipo'])
            df_mls = df_mls[df_mls['Equipo'] != 'Equipo']
            for c in ['PJ', 'Pts', 'GF', 'GC', 'xG', 'xGA']:
                if c in df_mls.columns: df_mls[c] = pd.to_numeric(df_mls[c], errors='coerce').fillna(1.0)
            df_mls['AttPen_Promedio'] = 16.0
            df_mls['Tiros_Promedio'] = 12.0
            df_mls['Calidad_Tiro'] = 0.11
            return df_mls
        except Exception as e: return pd.DataFrame()

    with st.spinner("⚡ Extrayendo métricas de MLS en FBref y fusionando con Liga MX..."):
        df_ligamx = cargar_datos_completos() # AHORA SÍ CONOCE LA FUNCIÓN
        df_mls = obtener_tabla_mls()
        if not df_mls.empty:
            df_mls = calcular_idr(df_mls)
            df_leagues_cup = pd.concat([df_ligamx, df_mls], ignore_index=True)
        else:
            st.error("⚠️ FBref bloqueó la conexión temporalmente. Usando datos de respaldo.")
            df_leagues_cup = df_ligamx.copy()

    @st.cache_data(ttl=3600*6)
    def obtener_jornada_leagues_cup():
        return [
            "Inter Miami vs Tigres",
            "LAFC vs Monterrey",
            "Columbus Crew vs América",
            "Seattle Sounders vs Pumas",
            "Orlando City vs Cruz Azul"
        ]
    
    partidos_lc = obtener_jornada_leagues_cup()
    
    # AHORA SÍ CONOCE LA FUNCIÓN GLOBAL DE MODELOS
    datos_lc = [ejecutar_laboratorio_modelos(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_leagues_cup) for p in partidos_lc]
    datos_lc = [d for d in datos_lc if d is not None]

    tab_singles, tab_dobles, tab_sgp, tab_jornada = st.tabs(["🎯 Picks Solos", "🛡️ Dobles (1X)", "🎰 Same Game Parlay", "📅 El Acumulador"])

    with tab_singles:
        c1, c2 = st.columns(2)
        picks_seguros = sorted(datos_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)
        if picks_seguros:
            t1 = picks_seguros[0]
            with c1: st.markdown(f'<div class="safe-card"><h3>🛡️ Pick Directo / DNB</h3><p><b>{t1["Local"] if t1["Prob_1"]>t1["Prob_2"] else t1["Visita"]}</b></p><p>Prob: {max(t1["Prob_1"], t1["Prob_2"])}%</p></div>', unsafe_allow_html=True)
        picks_goles = sorted(datos_lc, key=lambda x: x['Over_25'], reverse=True)
        if picks_goles:
            t2 = picks_goles[0]
            with c2: st.markdown(f'<div class="safe-card"><h3>⚽ Totales Conservadores</h3><p><b>{t2["Local"]} vs {t2["Visita"]}: Over 1.5</b></p><p>Prob (+2.5): {t2["Over_25"]}%</p></div>', unsafe_allow_html=True)

    with tab_dobles:
        picks_dobles, c_doble = "", 1.0
        for d in sorted(datos_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)[:3]:
            picks_dobles += f"⚽ <b>{d['Local'] if d['Prob_1']>d['Prob_2'] else d['Visita']} o Empate</b><br>"
            c_doble *= 1.25
        st.markdown(f'<div class="parlay-card"><h3>🛡️ Doble Oportunidad</h3>{picks_dobles}<hr><h4>🎟️ Momio Est: {(int((c_doble-1)*100) if c_doble>=2 else int(-100/(c_doble-1))):+}</h4></div>', unsafe_allow_html=True)

    with tab_sgp:
        if partidos_lc:
            p_sgp = st.selectbox("Selecciona partido SGP:", options=partidos_lc, key="sel_lc")
            loc_s, vis_s = p_sgp.split(" vs ")
            d_sgp = next((i for i in datos_lc if i["Local"] == loc_s.strip() and i["Visita"] == vis_s.strip()), None)
            if d_sgp:
                fav_sgp = d_sgp['Local'] if d_sgp['Prob_1'] > d_sgp['Prob_2'] else d_sgp['Visita']
                st.markdown(f'<div class="match-header"><h2>{d_sgp["Local"]} vs {d_sgp["Visita"]}</h2></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="risk-card"><h3>🔥 SGP Recomendado</h3><p>1. {fav_sgp} o Empate<br>2. Over 1.5 Goles<br>3. Over 7.5 Córners</p></div>', unsafe_allow_html=True)

    with tab_jornada:
        picks_j, c_j = "", 1.0
        for d in sorted(datos_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)[:4]:
            picks_j += f"🔥 <b>{d['Local'] if d['Prob_1']>d['Prob_2'] else d['Visita']} a Ganar</b><br>"
            c_j *= 1.85
        st.markdown(f'<div class="value-card"><h3>💰 Acumulador</h3>{picks_j}<hr><h4>🎟️ Momio: {(int((c_j-1)*100) if c_j>=2 else int(-100/(c_j-1))):+}</h4></div>', unsafe_allow_html=True)
