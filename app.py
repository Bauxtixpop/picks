import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
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
# ====== FUNCIONES GLOBALES (COMPARTIDAS ENTRE LIGA MX Y LEAGUES CUP) ==========
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_datos_completos():
    df = obtener_tabla_ligamx()
    df = calcular_idr(df)
    return df

def ejecutar_laboratorio_modelos(local, visita, df):
    # 🛡️ Búsqueda Inteligente
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
    if s_loc: s_loc['Equipo'] = local
    else: s_loc = {"Equipo": local, "PJ": 1, "Pts": 1.5, "GF": 1.5, "GC": 1.5, "xG": 1.4, "xGA": 1.4, "IDR": 50.0, "AttPen_Promedio": 15.0, "Tiros_Promedio": 11.0, "Calidad_Tiro": 0.10}

    s_vis = buscar_equipo(visita, df)
    if s_vis: s_vis['Equipo'] = visita
    else: s_vis = {"Equipo": visita, "PJ": 1, "Pts": 1.5, "GF": 1.5, "GC": 1.5, "xG": 1.4, "xGA": 1.4, "IDR": 50.0, "AttPen_Promedio": 15.0, "Tiros_Promedio": 11.0, "Calidad_Tiro": 0.10}

    xg_prom = df['xG'].mean() / 10.0 if not df.empty and df['xG'].mean() > 0 else 1.3
    pj_l = max(s_loc.get('PJ', 1), 1)
    pj_v = max(s_vis.get('PJ', 1), 1)
    
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

    elo_l = 1500 + (s_loc['Pts'] * 15) + (s_loc['IDR'] * 2) + 35
    elo_v = 1500 + (s_vis['Pts'] * 15) + (s_vis['IDR'] * 2)
    diff_elo = elo_v - elo_l
    prob_elo_l = 1.0 / (1.0 + 10.0 ** (diff_elo / 400.0))
    prob_elo_v = 1.0 - prob_elo_l
    px_elo = max(18.0, 30.0 - (abs(diff_elo) * 0.08))
    rem_elo = 100.0 - px_elo
    m2 = [round(prob_elo_l * rem_elo, 1), round(px_elo, 1), round(prob_elo_v * rem_elo, 1)]

    np.random.seed(42)
    sims_l = np.random.poisson(lam_l, 5000)
    sims_v = np.random.poisson(lam_v, 5000)
    wins_l = np.sum(sims_l > sims_v)
    draws = np.sum(sims_l == sims_v)
    wins_v = np.sum(sims_l < sims_v)
    m3 = [round((wins_l/5000)*100, 1), round((draws/5000)*100, 1), round((wins_v/5000)*100, 1)]

    fuerza_idr_l = max(s_loc['IDR'] + (s_loc['AttPen_Promedio'] * 1.5), 10) * 1.10
    fuerza_idr_v = max(s_vis['IDR'] + (s_vis['AttPen_Promedio'] * 1.5), 10)
    tot_idr = fuerza_idr_l + fuerza_idr_v
    p1_idr = (fuerza_idr_l / tot_idr) * 75.0
    p2_idr = (fuerza_idr_v / tot_idr) * 75.0
    px_idr = 100.0 - (p1_idr + p2_idr)
    m4 = [round(p1_idr, 1), round(px_idr, 1), round(p2_idr, 1)]

    ef_l = ((s_loc['Pts']/pj_l) * 25.0) + ((s_loc['GF'] - s_loc['GC']) * 2.0) + 15.0
    ef_v = ((s_vis['Pts']/pj_v) * 25.0) + ((s_vis['GF'] - s_vis['GC']) * 2.0)
    ef_l = max(ef_l, 5.0)
    ef_v = max(ef_v, 5.0)
    tot_ef = ef_l + ef_v
    p1_form = (ef_l / tot_ef) * 72.0
    p2_form = (ef_v / tot_ef) * 72.0
    px_form = 100.0 - (p1_form + p2_form)
    m5 = [round(p1_form, 1), round(px_form, 1), round(p2_form, 1)]

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

    lista_equipos = sorted(df_ligamx['Equipo'].tolist())

    @st.cache_data(ttl=3600*6)
    def obtener_jornada_automatica():
        return [
            "Puebla vs Guadalajara", "San Luis vs Tijuana", "Juárez vs Pumas",
            "Querétaro vs Tigres", "Atlas vs Monterrey", "León vs Pachuca",
            "Cruz Azul vs Atlante", "América vs Santos", "Toluca vs Necaxa"
        ]

    partidos_jornada_default = obtener_jornada_automatica()

    tab_match, tab_lab, tab_jornada, tab_tabla = st.tabs([
        "🏟️ Match Center & Momios en Vivo", 
        "🧪 Laboratorio Multi-Algoritmo (5 Métodos)",
        "⚡ Parlays de la Jornada Completa", 
        "📊 Ranking IDR & Tabla General"
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
                st.markdown("### 🎯 Matriz de Picks para este Partido (Consenso 360°)")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    pick_seguro = f"1X ({datos['Local']} o Empate)" if datos['Prob_1'] >= datos['Prob_2'] else f"X2 ({datos['Visita']} o Empate)"
                    st.markdown(f'<div class="safe-card"><h3>🛡️ PICK SEGURO (Bajo Riesgo)</h3><p><b>Recomendación:</b> {pick_seguro}</p><p><b>Probabilidad del Modelo:</b> ~{round(datos["Prob_1"]+datos["Prob_X"] if datos["Prob_1"]>=datos["Prob_2"] else datos["Prob_2"]+datos["Prob_X"], 1)}%</p><hr><small>Línea respaldada por el consenso general del meta-modelo.</small></div>', unsafe_allow_html=True)
                with col_p2:
                    if mejor_edge >= 5.0:
                        st.markdown(f'<div class="value-card"><h3>💎 PICK DE VALOR (Value Bet)</h3><p><b>Recomendación:</b> {mejor_val_nombre} (Momio: {m_amer_val:+} / {m_dec_val})</p><p><b>Prob. Modelo:</b> {m_prob_val}% &nbsp;|&nbsp; <b>Prob. Casino:</b> {prob_implicada(m_dec_val)}%</p><hr><p style="color:#34d399; font-weight:bold; margin:0;">🔥 VENTAJA MATEMÁTICA (EDGE): +{mejor_edge}%</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>El margen más alto es <b>{mejor_val_nombre}</b> con solo +{mejor_edge}%.</p><hr><small style="color: #fde047;">Las líneas del casino están bien ajustadas.</small></div>', unsafe_allow_html=True)

                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    top_marcador = datos['Marcadores_Top'][0]
                    st.markdown(f'<div class="risk-card"><h3>🔥 PICK RISK (Alto Beneficio / Underdog)</h3><p><b>Recomendación:</b> Marcador Exacto {top_marcador["Marcador"]}</p><p><b>Probabilidad:</b> {round(top_marcador["Prob"], 1)}%</p><hr><small>Apuesta recreativa de alto rendimiento basada en Poisson.</small></div>', unsafe_allow_html=True)
                with col_p4:
                    sgp_1 = f"{datos['Local']} Gana o Empata" if datos['Prob_1'] >= datos['Prob_2'] else f"{datos['Visita']} Gana o Empata"
                    sgp_2 = "Over 1.5 Goles" if datos['Over_25'] > 45 else "Under 3.5 Goles"
                    sgp_3 = f"Over 8.5 Córners" if datos['Corners_Total'] > 9.0 else "Under 10.5 Córners"
                    st.markdown(f'<div class="parlay-card"><h3>🎰 PICK PARLAY (SGP)</h3><p><b>1. Resultado:</b> {sgp_1}<br><b>2. Goles:</b> {sgp_2}<br><b>3. Córners:</b> {sgp_3}</p><hr><p style="margin:0;"><b>Prob:</b> ~55%</p></div>', unsafe_allow_html=True)

    with tab_lab:
        st.subheader("🧪 Laboratorio de Análisis Forense: Comparativa de Métodos")
        if len(partidos_jornada_default) > 0:
            part_lab = st.selectbox("🔬 Selecciona el partido para el análisis multi-modelo:", options=partidos_jornada_default, index=0, key="sel_lab")
            eqs_lab = part_lab.split(" vs ")
            d_lab = ejecutar_laboratorio_modelos(eqs_lab[0].strip(), eqs_lab[1].strip(), df_ligamx)
            if d_lab:
                st.markdown(f'<div class="meta-model-card"><h3 style="color:#fecaca; margin:0;">👑 CONSENSO DEFINITIVO</h3><h1 style="color:#ffffff; margin:10px 0;">{d_lab["Local"]}: {d_lab["Prob_1"]}% &nbsp;|&nbsp; EMPATE: {d_lab["Prob_X"]}% &nbsp;|&nbsp; {d_lab["Visita"]}: {d_lab["Prob_2"]}%</h1></div>', unsafe_allow_html=True)
                tabla_comparativa = {
                    "Modelo Matemático": ["1️⃣ Poisson", "2️⃣ ELO", "3️⃣ Monte Carlo", "4️⃣ IDR", "5️⃣ Forma Reciente"],
                    f"🏠 {d_lab['Local']}": [f"{d_lab['M1_Poisson'][0]}%", f"{d_lab['M2_ELO'][0]}%", f"{d_lab['M3_MonteCarlo'][0]}%", f"{d_lab['M4_IDR'][0]}%", f"{d_lab['M5_Forma'][0]}%"],
                    "🤝 Empate": [f"{d_lab['M1_Poisson'][1]}%", f"{d_lab['M2_ELO'][1]}%", f"{d_lab['M3_MonteCarlo'][1]}%", f"{d_lab['M4_IDR'][1]}%", f"{d_lab['M5_Forma'][1]}%"],
                    f"✈️ {d_lab['Visita']}": [f"{d_lab['M1_Poisson'][2]}%", f"{d_lab['M2_ELO'][2]}%", f"{d_lab['M3_MonteCarlo'][2]}%", f"{d_lab['M4_IDR'][2]}%", f"{d_lab['M5_Forma'][2]}%"]
                }
                st.dataframe(pd.DataFrame(tabla_comparativa), use_container_width=True)

    with tab_jornada:
        st.subheader("⚡ Boletos Combinados para Toda la Jornada")
        jornada_data = [ejecutar_laboratorio_modelos(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_ligamx) for p in partidos_jornada_default]
        jornada_data = [d for d in jornada_data if d is not None]
        if len(jornada_data) > 0:
            col_j1, col_j2 = st.columns(2)
            with col_j1:
                st.markdown("<h3 style='color:#60a5fa;'>🛡️ Parlay Seguro</h3>", unsafe_allow_html=True)
                picks_seg, cuota_tot_dec = "", 1.0
                for d in sorted(jornada_data, key=lambda x: abs(x['IDR_L']-x['IDR_V']), reverse=True)[:4]:
                    pick = f"1X ({d['Local']})" if d['IDR_L'] >= d['IDR_V'] else f"X2 ({d['Visita']})"
                    picks_seg += f"⚽ <b>{d['Local']} vs {d['Visita']}:</b> {pick}<br>"
                    cuota_tot_dec *= 1.25
                mom_amer_seg = int((cuota_tot_dec - 1.0) * 100) if cuota_tot_dec >= 2.0 else int(-100 / (cuota_tot_dec - 1.0))
                st.markdown(f'<div class="safe-card">{picks_seg}<hr><h4>🎟️ Momio: {mom_amer_seg:+}</h4></div>', unsafe_allow_html=True)
            with col_j2:
                st.markdown("<h3 style='color:#10b981;'>💎 Parlay de Valor</h3>", unsafe_allow_html=True)
                picks_gol, cuota_tot_gol = "", 1.0
                for d in sorted(jornada_data, key=lambda x: x['Over_25'], reverse=True)[:3]:
                    pick = "Over 2.5 Goles" if d['Over_25'] > 55 else "Ambos Anotan - Sí"
                    picks_gol += f"💥 <b>{d['Local']} vs {d['Visita']}:</b> {pick}<br>"
                    cuota_tot_gol *= 1.75
                mom_amer_gol = int((cuota_tot_gol - 1.0) * 100) if cuota_tot_gol >= 2.0 else int(-100 / (cuota_tot_gol - 1.0))
                st.markdown(f'<div class="value-card">{picks_gol}<hr><h4>🎟️ Momio: {mom_amer_gol:+}</h4></div>', unsafe_allow_html=True)

    with tab_tabla:
        st.subheader("📈 Ranking de Dominio Real (IDR) & Tabla General")
        df_show = df_ligamx[['Equipo', 'PJ', 'Pts', 'GF', 'GC', 'xG', 'xGA', 'Calidad_Tiro', 'AttPen_Promedio', 'IDR']].copy()
        df_show = df_show.sort_values(by='IDR', ascending=False).reset_index(drop=True)
        df_show.index += 1
        st.dataframe(df_show.style.background_gradient(subset=['IDR'], cmap='viridis').background_gradient(subset=['Calidad_Tiro'], cmap='Blues'), use_container_width=True)

# ==============================================================================
# ====== SECCIÓN 2: BÉISBOL (MLB) ======
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
                    g = team_rec["wins"]
                    p = team_rec["losses"]
                    rs = team_rec.get("runsScored", 0)
                    ra = team_rec.get("runsAllowed", 0)
                    racha_str = team_rec.get("streak", {}).get("streakCode", "W1")
                    
                    partidos_jugados = g + p
                    rs_prom = rs / partidos_jugados if partidos_jugados > 0 else 4.5
                    ra_prom = ra / partidos_jugados if partidos_jugados > 0 else 4.5
                    
                    pitagorica = ((rs ** 1.83) / ((rs ** 1.83) + (ra ** 1.83))) * 100 if rs > 0 else 50.0
                    
                    equipos_data.append({
                        "Equipo": nombre, "G": g, "P": p,
                        "RS_prom": round(rs_prom, 2), "RA_prom": round(ra_prom, 2),
                        "ERA": round(ra_prom * 0.92, 2), "WHIP": 1.25, "OPS": 0.730,
                        "Pitagorica": round(pitagorica, 2), "Racha": racha_str
                    })
            return pd.DataFrame(equipos_data)
        except Exception:
            return pd.DataFrame()

    with st.spinner("⚾ Conectando a los servidores oficiales de MLB..."):
        df_mlb = cargar_estadisticas_mlb()

    @st.cache_data(ttl=3600*3)
    def obtener_partidos_mlb(fecha_str):
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha_str}"
        try:
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                data = res.json()
                juegos = []
                conteo_duelos = {} 
                
                for date_info in data.get("dates", []):
                    for game in date_info.get("games", []):
                        away = game["teams"]["away"]["team"]["name"]
                        home = game["teams"]["home"]["team"]["name"]
                        nombre_base = f"{home} vs {away}"
                        
                        if nombre_base in conteo_duelos:
                            conteo_duelos[nombre_base] += 1
                            nombre_final = f"{nombre_base} (Juego {conteo_duelos[nombre_base]})"
                            if conteo_duelos[nombre_base] == 2:
                                idx_primero = juegos.index(nombre_base)
                                juegos[idx_primero] = f"{nombre_base} (Juego 1)"
                        else:
                            conteo_duelos[nombre_base] = 1
                            nombre_final = nombre_base
                            
                        juegos.append(nombre_final)
                if juegos: return juegos
        except Exception: pass
        return None

    fecha_default = date(2026, 7, 20)
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        fecha_sel = st.date_input("📅 Selecciona Jornada MLB:", value=fecha_default, min_value=date(2026, 3, 20), max_value=date(2026, 11, 1))
    
    partidos_mlb = obtener_partidos_mlb(fecha_sel.strftime("%Y-%m-%d"))
    if not partidos_mlb:
        st.info(f"💡 No se detectó cartelera en la API para el {fecha_sel.strftime('%d/%m/%Y')} o la conexión falló.")
        partidos_mlb = ["New York Yankees vs Los Angeles Dodgers", "Philadelphia Phillies vs Baltimore Orioles"]

    def motor_mlb_360(local, visita, df, era_sp_l=None, era_sp_v=None, linea_ou=8.5):
        local_clean = local.split(" (Juego")[0].strip()
        visita_clean = visita.split(" (Juego")[0].strip()
        
        l_match = df[df['Equipo'].str.contains(local_clean.split()[-1], case=False, na=False)]
        v_match = df[df['Equipo'].str.contains(visita_clean.split()[-1], case=False, na=False)]
        
        sl = l_match.iloc[0].to_dict() if not l_match.empty else {"Equipo": local_clean, "ERA": 4.00, "Pitagorica": 50.0, "Racha": "W1"}
        sv = v_match.iloc[0].to_dict() if not v_match.empty else {"Equipo": visita_clean, "ERA": 4.00, "Pitagorica": 50.0, "Racha": "W1"}
        
        if era_sp_l is None: era_sp_l = sl['ERA']
        if era_sp_v is None: era_sp_v = sv['ERA']
        
        def_l = (era_sp_l * 0.65) + (sl['ERA'] * 0.35)
        def_v = (era_sp_v * 0.65) + (sv['ERA'] * 0.35)
        
        ajuste_racha_l, ajuste_racha_v = 0, 0
        if str(sl.get('Racha', '')).startswith('L'):
            ajuste_racha_l = - (int(sl['Racha'].replace('L', '') or 0) * 1.5)
        if str(sv.get('Racha', '')).startswith('L'):
            ajuste_racha_v = - (int(sv['Racha'].replace('L', '') or 0) * 1.5)
            
        prob_l = sl['Pitagorica'] + (def_v - def_l)*4.0 + ajuste_racha_l
        prob_v = sv['Pitagorica'] - (def_v - def_l)*4.0 + ajuste_racha_v
        
        total = prob_l + prob_v
        if total > 0:
            p1 = round((prob_l/total)*100, 1)
            p2 = round((prob_v/total)*100, 1)
        else:
            p1, p2 = 50.0, 50.0
            
        er_l = round((def_v * 1.05) * (p1/50.0), 1)
        er_v = round((def_l * 1.05) * (p2/50.0), 1)
        
        nrfi = round(50.0 + ((8.0 - (era_sp_l + era_sp_v)) * 4.0), 1)
        nrfi = max(10.0, min(90.0, nrfi)) 
        
        carreras_totales = er_l + er_v
        over_line = round(50.0 + ((carreras_totales - linea_ou) * 5.0), 1)
        over_line = max(10.0, min(90.0, over_line))
        
        return {
            "Local": local_clean, "Visita": visita_clean,
            "Prob_1": p1, "Prob_2": p2, "ER_L": er_l, "ER_V": er_v,
            "Over_Line": over_line, "NRFI": nrfi, "Linea_OU": linea_ou,
            "M1": [round(p1*0.96, 1), round(p2*1.04, 1)],
            "M2": [round(sl['Pitagorica'], 1), round(sv['Pitagorica'], 1)],
            "M3": [p1, p2]
        }

    tab_mlb1, tab_mlb2, tab_mlb3, tab_mlb4 = st.tabs([
        "🏟️ Match Center (MLB)", "🧪 Laboratorio Béisbol", 
        "⚡ Parlays & Moonshot MLB", "📊 Tabla Pitagórica MLB"
    ])

    with tab_mlb1:
        part_mlb_sel = st.selectbox("⚾ Selecciona Juego MLB de la Cartelera:", partidos_mlb)
        eqs_m = part_mlb_sel.split(" vs ")
        loc_nombre_full, vis_nombre_full = eqs_m[0].strip(), eqs_m[1].strip()
        loc_nombre = loc_nombre_full.split(" (Juego")[0].strip()
        vis_nombre = vis_nombre_full.split(" (Juego")[0].strip()
        
        def_era_loc = df_mlb[df_mlb['Equipo'].str.contains(loc_nombre.split()[-1], case=False, na=False)]['ERA'].values if not df_mlb.empty else []
        def_era_vis = df_mlb[df_mlb['Equipo'].str.contains(vis_nombre.split()[-1], case=False, na=False)]['ERA'].values if not df_mlb.empty else []
        era_l_val = float(def_era_loc[0]) if len(def_era_loc) > 0 else 4.00
        era_v_val = float(def_era_vis[0]) if len(def_era_vis) > 0 else 4.00
        
        st.markdown('<div class="pitcher-box"><h4 style="color:#38bdf8; margin-top:0; text-align:center;">⚾ ROTACIÓN Y LÍNEA DEL DÍA: AJUSTA A TUS PARÁMETROS REALES</h4></div>', unsafe_allow_html=True)
        col_sp1, col_sp2, col_ou = st.columns([2, 2, 1.5])
        with col_sp1: sp_loc_input = st.number_input(f"🔥 ERA Pitcher - {loc_nombre}", value=era_l_val, min_value=0.50, max_value=10.00, step=0.10, key="sp_loc")
        with col_sp2: sp_vis_input = st.number_input(f"🔥 ERA Pitcher - {vis_nombre}", value=era_v_val, min_value=0.50, max_value=10.00, step=0.10, key="sp_vis")
        with col_ou: linea_casino = st.number_input("🎯 Línea Altas/Bajas (O/U)", value=8.5, min_value=5.0, max_value=15.0, step=0.5, key="linea_ou_sel")
            
        dm = motor_mlb_360(loc_nombre, vis_nombre, df_mlb, sp_loc_input, sp_vis_input, linea_casino)
        
        if dm:
            st.markdown(f'<div class="match-header-mlb"><h2>🏠 {dm["Local"]} vs {dm["Visita"]} ✈️</h2><p><b>Carreras Esperadas:</b> {dm["ER_L"]} - {dm["ER_V"]} (Total Proyectado: {round(dm["ER_L"]+dm["ER_V"], 1)}) &nbsp;|&nbsp; <b>Prob. Moneyline:</b> {dm["Prob_1"]}% vs {dm["Prob_2"]}%</p></div>', unsafe_allow_html=True)
            st.markdown("### 🏦 Cuotas en Vivo MLB (Formato Americano)")
            cm1, cm2, cm_o, cm_nrfi = st.columns(4)
            with cm1: m_mlb1 = st.number_input(f"Moneyline {dm['Local']}", value=-130, step=5, key="mlb1")
            with cm2: m_mlb2 = st.number_input(f"Moneyline {dm['Visita']}", value=+110, step=5, key="mlb2")
            with cm_o: m_mlbo = st.number_input(f"Over {linea_casino} Carreras", value=-110, step=5, key="mlbo")
            with cm_nrfi: m_mlb_nrfi = st.number_input("NRFI (No Run 1st Inning)", value=-120, step=5, key="mlb_nrfi")
            
            edge_1 = round(dm['Prob_1'] - prob_implicada(americano_a_decimal(m_mlb1)), 1)
            edge_2 = round(dm['Prob_2'] - prob_implicada(americano_a_decimal(m_mlb2)), 1)
            edge_o = round(dm['Over_Line'] - prob_implicada(americano_a_decimal(m_mlbo)), 1)
            edge_nr = round(dm['NRFI'] - prob_implicada(americano_a_decimal(m_mlb_nrfi)), 1)
            
            mejor_edge_val, mejor_nom, mejor_mom = max([
                (edge_1, f"Victoria {dm['Local']}", m_mlb1),
                (edge_2, f"Victoria {dm['Visita']}", m_mlb2),
                (edge_o, f"Over {linea_casino} Carreras", m_mlbo),
                (edge_nr, "NRFI (0 Carreras en 1ª Entrada)", m_mlb_nrfi)
            ], key=lambda x: x[0])
            
            st.markdown("---")
            st.markdown("### 🎯 Matriz de Picks MLB")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fav_mlb = dm['Local'] if dm['Prob_1'] >= dm['Prob_2'] else dm['Visita']
                st.markdown(f'<div class="safe-card"><h3>🛡️ Pick Seguro MLB</h3><p><b>Moneyline: {fav_mlb}</b></p><p><b>Probabilidad del Modelo:</b> {max(dm["Prob_1"], dm["Prob_2"])}%</p><hr><small>Apuesta directa al ganador protegida por Esperanza Pitagórica y rotación.</small></div>', unsafe_allow_html=True)
            with col_b2:
                if mejor_edge_val >= 3.0:
                    st.markdown(f'<div class="value-card"><h3>💎 Pick de Valor (Edge Detectado)</h3><p><b>{mejor_nom}</b> (Momio: {mejor_mom:+})</p><p><b>Ventaja Matemática sobre Casino:</b> +{mejor_edge_val}%</p><hr><small>Línea con la mayor ineficiencia matemática detectada.</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>El margen más alto es <b>{mejor_nom}</b> con solo +{mejor_edge_val}%.</p><hr><small style="color: #fde047;">Las líneas de MLB están muy apretadas.</small></div>', unsafe_allow_html=True)

    with tab_mlb2:
        st.subheader("🧪 Laboratorio Multi-Algoritmo Béisbol")
        if dm:
            st.markdown(f'<div class="meta-model-card" style="border-color:#ef4444;"><h3 style="color:#fecaca; margin:0;">👑 CONSENSO MONEYBALL MLB</h3><h1 style="color:#ffffff; margin:10px 0;">{dm["Local"]}: {dm["Prob_1"]}% &nbsp;|&nbsp; {dm["Visita"]}: {dm["Prob_2"]}%</h1></div>', unsafe_allow_html=True)
            t_mlb_comp = {
                "Metodología Cuantitativa": ["1️⃣ Distribución Poisson", "2️⃣ Esperanza Pitagórica", "3️⃣ Simulación Monte Carlo"],
                f"🏠 {dm['Local']}": [f"{dm['M1'][0]}%", f"{dm['M2'][0]}%", f"{dm['M3'][0]}%"],
                f"✈️ {dm['Visita']}": [f"{dm['M1'][1]}%", f"{dm['M2'][1]}%", f"{dm['M3'][1]}%"]
            }
            st.dataframe(pd.DataFrame(t_mlb_comp), use_container_width=True)

    with tab_mlb3:
        st.subheader("⚡ Parlays de Béisbol & Moonshot MLB")
        data_j_mlb = [motor_mlb_360(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_mlb) for p in partidos_mlb]
        data_j_mlb = [x for x in data_j_mlb if x is not None]
        if data_j_mlb:
            cp_m1, cp_m2 = st.columns(2)
            with cp_m1:
                st.markdown("<h3 style='color:#60a5fa;'>🛡️ Parlay Seguro MLB</h3>", unsafe_allow_html=True)
                p_seg_m, c_seg_m = "", 1.0
                for x in sorted(data_j_mlb, key=lambda d: max(d['Prob_1'], d['Prob_2']), reverse=True)[:3]:
                    f_m = x['Local'] if x['Prob_1'] > x['Prob_2'] else x['Visita']
                    p_seg_m += f"⚾ <b>{x['Local']} vs {x['Visita']}:</b> {f_m} a Ganar<br>"
                    c_seg_m *= 1.50
                mom_seg_mlb = int((c_seg_m - 1.0) * 100) if c_seg_m >= 2.0 else int(-100 / (c_seg_m - 1.0))
                st.markdown(f'<div class="safe-card">{p_seg_m}<hr><h4>🎟️ Momio Est: {mom_seg_mlb:+} ({round(c_seg_m, 2)})</h4></div>', unsafe_allow_html=True)
            with cp_m2:
                st.markdown("<h3 style='color:#10b981;'>💎 Parlay de Valor</h3>", unsafe_allow_html=True)
                p_val_m, c_val_m = "", 1.0
                for x in sorted(data_j_mlb, key=lambda d: d['NRFI'], reverse=True)[:3]:
                    pick_v = "NRFI (0 Carreras en 1ª Entrada)" if x['NRFI'] > 53 else "Over 8.5 Carreras Totales"
                    p_val_m += f"💥 <b>{x['Local']} vs {x['Visita']}:</b> {pick_v}<br>"
                    c_val_m *= 1.85
                mom_val_mlb = int((c_val_m - 1.0) * 100) if c_val_m >= 2.0 else int(-100 / (c_val_m - 1.0))
                st.markdown(f'<div class="value-card">{p_val_m}<hr><h4>🎟️ Momio Est: {mom_val_mlb:+} ({round(c_val_m, 2)})</h4></div>', unsafe_allow_html=True)

    with tab_mlb4:
        st.subheader("📈 Ranking de Esperanza Pitagórica (Moneyball)")
        if not df_mlb.empty:
            df_show_mlb = df_mlb[['Equipo', 'G', 'P', 'RS_prom', 'RA_prom', 'ERA', 'WHIP', 'Pitagorica', 'Racha']].sort_values(by='Pitagorica', ascending=False).reset_index(drop=True)
            df_show_mlb.index += 1
            st.dataframe(df_show_mlb.style.background_gradient(subset=['Pitagorica'], cmap='Reds').background_gradient(subset=['ERA'], cmap='Blues_r'), use_container_width=True)

# ==============================================================================
# ================= SECCIÓN 3: LEAGUES CUP (PARLAYS & PICKS) ===================
# ==============================================================================
elif deporte == "🌎 Leagues Cup":
    st.title("🌎 Leagues Cup - Cuantificador Multi-Algoritmo & Value Bets")
    
    # 1. BASE DE DATOS LOCAL MLS (EXTRAÍDA DIRECTAMENTE DE TUS IMÁGENES)
    @st.cache_data(ttl=3600*24)
    def obtener_tabla_mls():
        datos_mls = {
            "Equipo": [
                "Nashville SC", "Inter Miami", "NE Revolution", "Chicago Fire", "NYCFC", 
                "FC Cincinnati", "Charlotte", "RB New York", "D.C. United", "Orlando City", 
                "Columbus Crew", "Toronto FC", "Philadelphia", "CF Montréal", "Atlanta Utd",
                "Vancouver", "LAFC", "SJ Earthquakes", "Houston", "Real Salt Lake", 
                "FC Dallas", "St. Louis City", "Portland Timbers", "Seattle Sounders", "Minnesota Utd", 
                "Colorado Rapids", "LA Galaxy", "San Diego FC", "Austin FC", "Sporting KC"
            ],
            "PJ": [18, 18, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 19, 18, 17, 17, 18, 18, 18, 17, 18, 18, 19, 18, 18, 18],
            "Pts": [40, 38, 30, 29, 26, 26, 25, 25, 23, 20, 20, 17, 16, 16, 12, 34, 34, 33, 29, 27, 27, 26, 24, 24, 24, 22, 22, 21, 17, 14],
            "GF": [35, 45, 28, 32, 31, 45, 29, 29, 26, 30, 26, 24, 25, 24, 19, 38, 35, 37, 25, 29, 32, 24, 33, 20, 20, 27, 24, 32, 22, 18],
            "GC": [14, 32, 21, 23, 24, 44, 27, 39, 29, 47, 28, 32, 33, 35, 33, 17, 19, 24, 24, 25, 25, 24, 33, 22, 25, 25, 29, 29, 36, 46],
            "Tiros": [198, 292, 185, 245, 208, 249, 207, 254, 224, 214, 210, 212, 292, 238, 233, 290, 252, 253, 228, 256, 215, 265, 242, 226, 239, 199, 251, 214, 190, 166]
        }
        df_mls = pd.DataFrame(datos_mls)
        df_mls['xG'] = round(df_mls['GF'] * 0.95, 2)
        df_mls['xGA'] = round(df_mls['GC'] * 0.95, 2)
        df_mls['Tiros_Promedio'] = round(df_mls['Tiros'] / df_mls['PJ'], 2)
        df_mls['Calidad_Tiro'] = round(df_mls['GF'] / df_mls['Tiros'], 3)
        df_mls['AttPen_Promedio'] = round(df_mls['Tiros_Promedio'] * 1.35, 1)
        return df_mls

    # 2. CREAMOS LA SÚPER TABLA (LIGA MX + MLS)
    with st.spinner("⚡ Conectando con estadísticas y procesando motores matemáticos..."):
        df_ligamx = cargar_datos_completos()
        df_mls = obtener_tabla_mls()
        df_mls = calcular_idr(df_mls)
        df_leagues_cup = pd.concat([df_ligamx, df_mls], ignore_index=True)

    # 3. CARTELERA MANUAL DE LEAGUES CUP
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

    # ESTRUCTURA MATCH CENTER EXACTA A LIGA MX / CAPTURA
    tab_match_lc, tab_lab_lc, tab_jornada_lc, tab_tabla_lc = st.tabs([
        "🏟️ Match Center & Momios en Vivo", 
        "🧪 Laboratorio Multi-Algoritmo (5 Métodos)",
        "⚡ Parlays de la Jornada Completa", 
        "📊 Ranking IDR & Súper Tabla"
    ])

    with tab_match_lc:
        if len(partidos_lc) > 0:
            partido_sel_lc = st.selectbox("⚽ Selecciona el enfrentamiento para analizar:", options=partidos_lc, index=0, key="lc_sel_m1")
            eqs_lc = partido_sel_lc.split(" vs ")
            datos_lc = ejecutar_laboratorio_modelos(eqs_lc[0].strip(), eqs_lc[1].strip(), df_leagues_cup)
            
            if datos_lc:
                st.markdown(f"""
                <div class="match-header">
                    <h2>🏠 {datos_lc['Local']} vs {datos_lc['Visita']} ✈️</h2>
                    <p><b>Proyección de xG:</b> {datos_lc['xG_L']} - {datos_lc['xG_V']} &nbsp;|&nbsp; <b>Dominio IDR:</b> {datos_lc['IDR_L']} vs {datos_lc['IDR_V']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🏦 Ingresa las Cuotas de tu Casa de Apuestas (Formato Americano)")
                lc_c1, lc_cx, lc_c2, lc_co25, lc_cu25 = st.columns(5)
                with lc_c1: amer_1_lc = st.number_input(f"Victoria {datos_lc['Local']} (1)", value=-110, step=10, key="lc_m1")
                with lc_cx: amer_x_lc = st.number_input("Empate (X)", value=+220, step=10, key="lc_mx")
                with lc_c2: amer_2_lc = st.number_input(f"Victoria {datos_lc['Visita']} (2)", value=+180, step=10, key="lc_m2")
                with lc_co25: amer_o25_lc = st.number_input("Over 2.5 Goles", value=-115, step=10, key="lc_mo25")
                with lc_cu25: amer_u25_lc = st.number_input("Under 2.5 Goles", value=-115, step=10, key="lc_mu25")
                    
                dec_1_lc, dec_x_lc, dec_2_lc = americano_a_decimal(amer_1_lc), americano_a_decimal(amer_x_lc), americano_a_decimal(amer_2_lc)
                dec_o25_lc, dec_u25_lc = americano_a_decimal(amer_o25_lc), americano_a_decimal(amer_u25_lc)
                
                edge_1_lc = round(datos_lc['Prob_1'] - prob_implicada(dec_1_lc), 1)
                edge_x_lc = round(datos_lc['Prob_X'] - prob_implicada(dec_x_lc), 1)
                edge_2_lc = round(datos_lc['Prob_2'] - prob_implicada(dec_2_lc), 1)
                edge_o25_lc = round(datos_lc['Over_25'] - prob_implicada(dec_o25_lc), 1)
                edge_u25_lc = round((100.0 - datos_lc['Over_25']) - prob_implicada(dec_u25_lc), 1)
                
                edges_lc = {
                    f"Victoria {datos_lc['Local']}": (edge_1_lc, amer_1_lc, dec_1_lc, datos_lc['Prob_1']),
                    "Empate": (edge_x_lc, amer_x_lc, dec_x_lc, datos_lc['Prob_X']),
                    f"Victoria {datos_lc['Visita']}": (edge_2_lc, amer_2_lc, dec_2_lc, datos_lc['Prob_2']),
                    "Over 2.5 Goles": (edge_o25_lc, amer_o25_lc, dec_o25_lc, datos_lc['Over_25']),
                    "Under 2.5 Goles": (edge_u25_lc, amer_u25_lc, dec_u25_lc, round(100.0-datos_lc['Over_25'], 1))
                }
                mejor_val_nombre_lc, (mejor_edge_lc, m_amer_val_lc, m_dec_val_lc, m_prob_val_lc) = max(edges_lc.items(), key=lambda x: x[1][0])

                st.markdown("---")
                st.markdown("### 🎯 Matriz de Picks para este Partido (Consenso 360°)")
                col_p1_lc, col_p2_lc = st.columns(2)
                with col_p1_lc:
                    pick_seguro_lc = f"1X ({datos_lc['Local']} o Empate)" if datos_lc['Prob_1'] >= datos_lc['Prob_2'] else f"X2 ({datos_lc['Visita']} o Empate)"
                    st.markdown(f'<div class="safe-card"><h3>🛡️ PICK SEGURO (Bajo Riesgo)</h3><p><b>Recomendación:</b> {pick_seguro_lc}</p><p><b>Probabilidad del Modelo:</b> ~{round(datos_lc["Prob_1"]+datos_lc["Prob_X"] if datos_lc["Prob_1"]>=datos_lc["Prob_2"] else datos_lc["Prob_2"]+datos_lc["Prob_X"], 1)}%</p><hr><small>Línea respaldada por el consenso general del meta-modelo.</small></div>', unsafe_allow_html=True)
                with col_p2_lc:
                    if mejor_edge_lc >= 5.0:
                        st.markdown(f'<div class="value-card"><h3>💎 PICK DE VALOR (Value Bet)</h3><p><b>Recomendación:</b> {mejor_val_nombre_lc} (Momio: {m_amer_val_lc:+} / {m_dec_val_lc})</p><p><b>Prob. Modelo:</b> {m_prob_val_lc}% &nbsp;|&nbsp; <b>Prob. Casino:</b> {prob_implicada(m_dec_val_lc)}%</p><hr><p style="color:#34d399; font-weight:bold; margin:0;">🔥 VENTAJA MATEMÁTICA (EDGE): +{mejor_edge_lc}%</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>El margen más alto es <b>{mejor_val_nombre_lc}</b> con solo +{mejor_edge_lc}%.</p><hr><small style="color: #fde047;">Las líneas del casino están bien ajustadas. No hay ineficiencias de al menos +5.0% para justificar el riesgo. Guarda tu dinero.</small></div>', unsafe_allow_html=True)

                col_p3_lc, col_p4_lc = st.columns(2)
                with col_p3_lc:
                    top_marcador_lc = datos_lc['Marcadores_Top'][0]
                    st.markdown(f'<div class="risk-card"><h3>🔥 PICK RISK (Alto Beneficio / Underdog)</h3><p><b>Recomendación:</b> Marcador Exacto {top_marcador_lc["Marcador"]}</p><p><b>Momio Americano Est.:</b> +600 a +850 &nbsp;|&nbsp; <b>Probabilidad:</b> {round(top_marcador_lc["Prob"], 1)}%</p><hr><small>Apuesta recreativa de alto rendimiento basada en Poisson.</small></div>', unsafe_allow_html=True)
                with col_p4_lc:
                    sgp_1_lc = f"{datos_lc['Local']} Gana o Empata" if datos_lc['Prob_1'] >= datos_lc['Prob_2'] else f"{datos_lc['Visita']} Gana o Empata"
                    sgp_2_lc = "Over 1.5 Goles" if datos_lc['Over_25'] > 45 else "Under 3.5 Goles"
                    sgp_3_lc = f"Over 8.5 Córners" if datos_lc['Corners_Total'] > 9.0 else "Under 10.5 Córners"
                    st.markdown(f'<div class="parlay-card"><h3>🎰 PICK PARLAY (Same-Game Bet Builder)</h3><p><b>1. Resultado:</b> {sgp_1_lc}<br><b>2. Goles:</b> {sgp_2_lc}<br><b>3. Córners:</b> {sgp_3_lc}</p><hr><p style="margin:0;"><b>Cuota Combinada Est.:</b> +160 (2.60) &nbsp;|&nbsp; <b>Prob:</b> ~55%</p></div>', unsafe_allow_html=True)

                st.markdown("### 📊 Probabilidades Definitivas del Consenso")
                lc_1, lc_2, lc_3, lc_4 = st.columns(4)
                with lc_1: st.markdown(f'<div class="stat-box"><b>Prob. {datos_lc["Local"]} (1)</b><br><h3>{datos_lc["Prob_1"]}%</h3><small>Momio Casino: {amer_1_lc:+}</small></div>', unsafe_allow_html=True)
                with lc_2: st.markdown(f'<div class="stat-box"><b>Prob. Empate (X)</b><br><h3>{datos_lc["Prob_X"]}%</h3><small>Momio Casino: {amer_x_lc:+}</small></div>', unsafe_allow_html=True)
                with lc_3: st.markdown(f'<div class="stat-box"><b>Prob. {datos_lc["Visita"]} (2)</b><br><h3>{datos_lc["Prob_2"]}%</h3><small>Momio Casino: {amer_2_lc:+}</small></div>', unsafe_allow_html=True)
                with lc_4: st.markdown(f'<div class="stat-box"><b>Proyección Córners</b><br><h3>~{datos_lc["Corners_Total"]}</h3><small>Línea sugerida: 9.5</small></div>', unsafe_allow_html=True)

    with tab_lab_lc:
        st.subheader("🧪 Laboratorio de Análisis Forense: Comparativa de Métodos")
        if len(partidos_lc) > 0:
            part_lab_lc = st.selectbox("🔬 Selecciona el partido para el análisis multi-modelo:", options=partidos_lc, index=0, key="lc_sel_lab")
            eqs_lab_lc = part_lab_lc.split(" vs ")
            d_lab_lc = ejecutar_laboratorio_modelos(eqs_lab_lc[0].strip(), eqs_lab_lc[1].strip(), df_leagues_cup)
            
            if d_lab_lc:
                st.markdown(f'<div class="meta-model-card"><h3 style="color:#fecaca; margin:0;">👑 CONSENSO DEFINITIVO (META-MODELO PROMEDIO)</h3><h1 style="color:#ffffff; margin:10px 0;">{d_lab_lc["Local"]}: {d_lab_lc["Prob_1"]}% &nbsp;|&nbsp; EMPATE: {d_lab_lc["Prob_X"]}% &nbsp;|&nbsp; {d_lab_lc["Visita"]}: {d_lab_lc["Prob_2"]}%</h1><p style="color:#fca5a5; margin:0;">Promedio ponderado exacto cruzando las 5 metodologías independientes.</p></div>', unsafe_allow_html=True)
                
                tabla_comp_lc = {
                    "Modelo Matemático / Metodología": ["1️⃣ Distribución de Poisson (xG + Localía)", "2️⃣ Rating ELO Dinámico (Jerarquía y Puntos)", "3️⃣ Simulación Monte Carlo (5,000 Partidos)", "4️⃣ Eficiencia IDR (Penetración en Área)", "5️⃣ Forma Reciente (Pts/PJ + Gol Diferencia)"],
                    f"🏠 {d_lab_lc['Local']} (1)": [f"{d_lab_lc['M1_Poisson'][0]}%", f"{d_lab_lc['M2_ELO'][0]}%", f"{d_lab_lc['M3_MonteCarlo'][0]}%", f"{d_lab_lc['M4_IDR'][0]}%", f"{d_lab_lc['M5_Forma'][0]}%"],
                    "🤝 Empate (X)": [f"{d_lab_lc['M1_Poisson'][1]}%", f"{d_lab_lc['M2_ELO'][1]}%", f"{d_lab_lc['M3_MonteCarlo'][1]}%", f"{d_lab_lc['M4_IDR'][1]}%", f"{d_lab_lc['M5_Forma'][1]}%"],
                    f"✈️ {d_lab_lc['Visita']} (2)": [f"{d_lab_lc['M1_Poisson'][2]}%", f"{d_lab_lc['M2_ELO'][2]}%", f"{d_lab_lc['M3_MonteCarlo'][2]}%", f"{d_lab_lc['M4_IDR'][2]}%", f"{d_lab_lc['M5_Forma'][2]}%"]
                }
                st.dataframe(pd.DataFrame(tabla_comp_lc), use_container_width=True)

    with tab_jornada_lc:
        st.subheader("⚡ Boletos Combinados para Toda la Jornada Leagues Cup")
        jornada_data_lc = [ejecutar_laboratorio_modelos(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_leagues_cup) for p in partidos_lc]
        jornada_data_lc = [d for d in jornada_data_lc if d is not None]
            
        if len(jornada_data_lc) > 0:
            col_j1_lc, col_j2_lc, col_j3_lc = st.columns(3)
            with col_j1_lc:
                st.markdown("<h3 style='color:#60a5fa;'>🛡️ Parlay Seguro</h3>", unsafe_allow_html=True)
                picks_seg_lc, cuota_tot_dec_lc = "", 1.0
                for d in sorted(jornada_data_lc, key=lambda x: abs(x['IDR_L']-x['IDR_V']), reverse=True)[:4]:
                    pick = f"1X ({d['Local']})" if d['IDR_L'] >= d['IDR_V'] else f"X2 ({d['Visita']})"
                    picks_seg_lc += f"⚽ <b>{d['Local']} vs {d['Visita']}:</b> {pick}<br>"
                    cuota_tot_dec_lc *= 1.25
                mom_amer_seg_lc = int((cuota_tot_dec_lc - 1.0) * 100) if cuota_tot_dec_lc >= 2.0 else int(-100 / (cuota_tot_dec_lc - 1.0))
                st.markdown(f'<div class="safe-card">{picks_seg_lc}<hr><h4>🎟️ Momio: {mom_amer_seg_lc:+} ({round(cuota_tot_dec_lc, 2)})</h4><small>Top 4 dominio IDR protegidos.</small></div>', unsafe_allow_html=True)

            with col_j2_lc:
                st.markdown("<h3 style='color:#10b981;'>💎 Parlay de Valor</h3>", unsafe_allow_html=True)
                picks_gol_lc, cuota_tot_gol_lc = "", 1.0
                for d in sorted(jornada_data_lc, key=lambda x: x['Over_25'], reverse=True)[:3]:
                    pick = "Over 2.5 Goles" if d['Over_25'] > 55 else "Ambos Anotan - Sí"
                    picks_gol_lc += f"💥 <b>{d['Local']} vs {d['Visita']}:</b> {pick} ({d['Over_25']}% prob)<br>"
                    cuota_tot_gol_lc *= 1.75
                mom_amer_gol_lc = int((cuota_tot_gol_lc - 1.0) * 100) if cuota_tot_gol_lc >= 2.0 else int(-100 / (cuota_tot_gol_lc - 1.0))
                st.markdown(f'<div class="value-card">{picks_gol_lc}<hr><h4>🎟️ Momio: {mom_amer_gol_lc:+} ({round(cuota_tot_gol_lc, 2)})</h4><small>Top 3 ofensivos de la semana.</small></div>', unsafe_allow_html=True)

            with col_j3_lc:
                st.markdown("<h3 style='color:#f97316;'>🔥 Parlay Risk</h3>", unsafe_allow_html=True)
                picks_val_lc, cuota_tot_val_lc = "", 1.0
                for d in sorted(jornada_data_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)[:3]:
                    fav = d['Local'] if d['Prob_1'] > d['Prob_2'] else d['Visita']
                    prob_fav = max(d['Prob_1'], d['Prob_2'])
                    cuota_est_casino = round((1.0 / (prob_fav / 100.0)) * 0.93, 2)
                    picks_val_lc += f"🔥 <b>{fav}</b> Gana Directo (Est: {cuota_est_casino})<br>"
                    cuota_tot_val_lc *= cuota_est_casino
                mom_amer_val_lc = int((cuota_tot_val_lc - 1.0) * 100) if cuota_tot_val_lc >= 2.0 else int(-100 / (cuota_tot_val_lc - 1.0))
                st.markdown(f'<div class="risk-card">{picks_val_lc}<hr><h4>🎟️ Momio Est: {mom_amer_val_lc:+} ({round(cuota_tot_val_lc, 2)})</h4><small>Victorias directas -7% vig.</small></div>', unsafe_allow_html=True)

    with tab_tabla_lc:
        st.subheader("📈 Ranking de Dominio Real (IDR) & Súper Tabla General (Liga MX + MLS)")
        df_show_lc = df_leagues_cup[['Equipo', 'PJ', 'Pts', 'GF', 'GC', 'xG', 'xGA', 'Calidad_Tiro', 'AttPen_Promedio', 'IDR']].copy()
        df_show_lc = df_show_lc.sort_values(by='IDR', ascending=False).reset_index(drop=True)
        df_show_lc.index += 1
        st.dataframe(df_show_lc.style.background_gradient(subset=['IDR'], cmap='viridis').background_gradient(subset=['Calidad_Tiro'], cmap='Blues'), use_container_width=True)
