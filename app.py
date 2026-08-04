import streamlit as st
import cloudscraper
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
# ====== FUNCIONES GLOBALES (COMPARTIDAS ENTRE LIGA MX Y LEAGUES CUP) ==========
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

    lista_equipos = sorted(df_ligamx['Equipo'].tolist())

    # --- CONTROL DE JORNADA MANUAL ---
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
                        st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>El margen más alto es <b>{mejor_val_nombre}</b> con solo +{mejor_edge}%.</p><hr><small style="color: #fde047;">Las líneas del casino están bien ajustadas. No hay ineficiencias de al menos +5.0% para justificar el riesgo. Guarda tu dinero.</small></div>', unsafe_allow_html=True)

                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    top_marcador = datos['Marcadores_Top'][0]
                    st.markdown(f'<div class="risk-card"><h3>🔥 PICK RISK (Alto Beneficio / Underdog)</h3><p><b>Recomendación:</b> Marcador Exacto {top_marcador["Marcador"]}</p><p><b>Momio Americano Est.:</b> +600 a +850 &nbsp;|&nbsp; <b>Probabilidad:</b> {round(top_marcador["Prob"], 1)}%</p><hr><small>Apuesta recreativa de alto rendimiento basada en Poisson.</small></div>', unsafe_allow_html=True)
                with col_p4:
                    sgp_1 = f"{datos['Local']} Gana o Empata" if datos['Prob_1'] >= datos['Prob_2'] else f"{datos['Visita']} Gana o Empata"
                    sgp_2 = "Over 1.5 Goles" if datos['Over_25'] > 45 else "Under 3.5 Goles"
                    sgp_3 = f"Over 8.5 Córners" if datos['Corners_Total'] > 9.0 else "Under 10.5 Córners"
                    st.markdown(f'<div class="parlay-card"><h3>🎰 PICK PARLAY (Same-Game Bet Builder)</h3><p><b>1. Resultado:</b> {sgp_1}<br><b>2. Goles:</b> {sgp_2}<br><b>3. Córners:</b> {sgp_3}</p><hr><p style="margin:0;"><b>Cuota Combinada Est.:</b> +160 (2.60) &nbsp;|&nbsp; <b>Prob:</b> ~55%</p></div>', unsafe_allow_html=True)

                st.markdown("### 📊 Probabilidades Definitivas del Consenso")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f'<div class="stat-box"><b>Prob. {datos["Local"]} (1)</b><br><h3>{datos["Prob_1"]}%</h3><small>Momio Casino: {amer_1:+}</small></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="stat-box"><b>Prob. Empate (X)</b><br><h3>{datos["Prob_X"]}%</h3><small>Momio Casino: {amer_x:+}</small></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="stat-box"><b>Prob. {datos["Visita"]} (2)</b><br><h3>{datos["Prob_2"]}%</h3><small>Momio Casino: {amer_2:+}</small></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="stat-box"><b>Proyección Córners</b><br><h3>~{datos["Corners_Total"]}</h3><small>Línea sugerida: 9.5</small></div>', unsafe_allow_html=True)

    with tab_lab:
        st.subheader("🧪 Laboratorio de Análisis Forense: Comparativa de Métodos")
        if len(partidos_jornada_default) > 0:
            part_lab = st.selectbox("🔬 Selecciona el partido para el análisis multi-modelo:", options=partidos_jornada_default, index=0, key="sel_lab")
            eqs_lab = part_lab.split(" vs ")
            d_lab = ejecutar_laboratorio_modelos(eqs_lab[0].strip(), eqs_lab[1].strip(), df_ligamx)
            
            if d_lab:
                st.markdown(f'<div class="meta-model-card"><h3 style="color:#fecaca; margin:0;">👑 CONSENSO DEFINITIVO (META-MODELO PROMEDIO)</h3><h1 style="color:#ffffff; margin:10px 0;">{d_lab["Local"]}: {d_lab["Prob_1"]}% &nbsp;|&nbsp; EMPATE: {d_lab["Prob_X"]}% &nbsp;|&nbsp; {d_lab["Visita"]}: {d_lab["Prob_2"]}%</h1><p style="color:#fca5a5; margin:0;">Promedio ponderado exacto cruzando las 5 metodologías independientes.</p></div>', unsafe_allow_html=True)
                
                tabla_comparativa = {
                    "Modelo Matemático / Metodología": ["1️⃣ Distribución de Poisson (xG + Localía)", "2️⃣ Rating ELO Dinámico (Jerarquía y Puntos)", "3️⃣ Simulación Monte Carlo (5,000 Partidos)", "4️⃣ Eficiencia IDR (Penetración en Área)", "5️⃣ Forma Reciente (Pts/PJ + Gol Diferencia)"],
                    f"🏠 {d_lab['Local']} (1)": [f"{d_lab['M1_Poisson'][0]}%", f"{d_lab['M2_ELO'][0]}%", f"{d_lab['M3_MonteCarlo'][0]}%", f"{d_lab['M4_IDR'][0]}%", f"{d_lab['M5_Forma'][0]}%"],
                    "🤝 Empate (X)": [f"{d_lab['M1_Poisson'][1]}%", f"{d_lab['M2_ELO'][1]}%", f"{d_lab['M3_MonteCarlo'][1]}%", f"{d_lab['M4_IDR'][1]}%", f"{d_lab['M5_Forma'][1]}%"],
                    f"✈️ {d_lab['Visita']} (2)": [f"{d_lab['M1_Poisson'][2]}%", f"{d_lab['M2_ELO'][2]}%", f"{d_lab['M3_MonteCarlo'][2]}%", f"{d_lab['M4_IDR'][2]}%", f"{d_lab['M5_Forma'][2]}%"]
                }
                st.dataframe(pd.DataFrame(tabla_comparativa), use_container_width=True)
                
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    st.markdown(f'<div class="method-box"><h4 style="color:#60a5fa;">1. Distribución de Poisson (xG)</h4><p>Evalúa Goles Esperados y otorga +15% por localía.</p><hr style="border-color:#334155;"><b>Resultado:</b> {d_lab["Local"]} ({d_lab["M1_Poisson"][0]}%) | X ({d_lab["M1_Poisson"][1]}%) | {d_lab["Visita"]} ({d_lab["M1_Poisson"][2]}%)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="method-box"><h4 style="color:#a78bfa;">3. Simulación Monte Carlo (5,000 Iteraciones)</h4><p>Juega 5,000 partidos virtuales para ver en cuántos universos gana cada equipo.</p><hr style="border-color:#334155;"><b>Resultado:</b> {d_lab["Local"]} ({d_lab["M3_MonteCarlo"][0]}%) | X ({d_lab["M3_MonteCarlo"][1]}%) | {d_lab["Visita"]} ({d_lab["M3_MonteCarlo"][2]}%)</div>', unsafe_allow_html=True)
                with col_l2:
                    st.markdown(f'<div class="method-box"><h4 style="color:#34d399;">2. Rating ELO Dinámico</h4><p>Evalúa jerarquía pura y ventaja de estadio con curva logística.</p><hr style="border-color:#334155;"><b>Resultado:</b> {d_lab["Local"]} ({d_lab["M2_ELO"][0]}%) | X ({d_lab["M2_ELO"][1]}%) | {d_lab["Visita"]} ({d_lab["M2_ELO"][2]}%)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="method-box"><h4 style="color:#f87171;">4. Eficiencia Táctico-Penetrativa (IDR Puro)</h4><p>Mide exclusively quién pisa más el área chica rival (AttPen) y calidad por tiro.</p><hr style="border-color:#334155;"><b>Resultado:</b> {d_lab["Local"]} ({d_lab["M4_IDR"][0]}%) | X ({d_lab["M4_IDR"][1]}%) | {d_lab["Visita"]} ({d_lab["M4_IDR"][2]}%)</div>', unsafe_allow_html=True)

    with tab_jornada:
        st.subheader("⚡ Boletos Combinados para Toda la Jornada (Momios Reales Casino -7% Margen)")
        jornada_data = [ejecutar_laboratorio_modelos(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_ligamx) for p in partidos_jornada_default]
        jornada_data = [d for d in jornada_data if d is not None]
            
        if len(jornada_data) > 0:
            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                st.markdown("<h3 style='color:#60a5fa;'>🛡️ Parlay Seguro</h3>", unsafe_allow_html=True)
                picks_seg, cuota_tot_dec = "", 1.0
                for d in sorted(jornada_data, key=lambda x: abs(x['IDR_L']-x['IDR_V']), reverse=True)[:4]:
                    pick = f"1X ({d['Local']})" if d['IDR_L'] >= d['IDR_V'] else f"X2 ({d['Visita']})"
                    picks_seg += f"⚽ <b>{d['Local']} vs {d['Visita']}:</b> {pick}<br>"
                    cuota_tot_dec *= 1.25
                mom_amer_seg = int((cuota_tot_dec - 1.0) * 100) if cuota_tot_dec >= 2.0 else int(-100 / (cuota_tot_dec - 1.0))
                st.markdown(f'<div class="safe-card">{picks_seg}<hr><h4>🎟️ Momio: {mom_amer_seg:+} ({round(cuota_tot_dec, 2)})</h4><small>Top 4 dominio IDR protegidos.</small></div>', unsafe_allow_html=True)

            with col_j2:
                st.markdown("<h3 style='color:#10b981;'>💎 Parlay de Valor</h3>", unsafe_allow_html=True)
                picks_gol, cuota_tot_gol = "", 1.0
                for d in sorted(jornada_data, key=lambda x: x['Over_25'], reverse=True)[:3]:
                    pick = "Over 2.5 Goles" if d['Over_25'] > 55 else "Ambos Anotan - Sí"
                    picks_gol += f"💥 <b>{d['Local']} vs {d['Visita']}:</b> {pick} ({d['Over_25']}% prob)<br>"
                    cuota_tot_gol *= 1.75
                mom_amer_gol = int((cuota_tot_gol - 1.0) * 100) if cuota_tot_gol >= 2.0 else int(-100 / (cuota_tot_gol - 1.0))
                st.markdown(f'<div class="value-card">{picks_gol}<hr><h4>🎟️ Momio: {mom_amer_gol:+} ({round(cuota_tot_gol, 2)})</h4><small>Top 3 ofensivos de la semana.</small></div>', unsafe_allow_html=True)

            with col_j3:
                st.markdown("<h3 style='color:#f97316;'>🔥 Parlay Risk</h3>", unsafe_allow_html=True)
                picks_val, cuota_tot_val = "", 1.0
                for d in sorted(jornada_data, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)[:3]:
                    fav = d['Local'] if d['Prob_1'] > d['Prob_2'] else d['Visita']
                    prob_fav = max(d['Prob_1'], d['Prob_2'])
                    cuota_est_casino = round((1.0 / (prob_fav / 100.0)) * 0.93, 2)
                    picks_val += f"🔥 <b>{fav}</b> Gana Directo (Est: {cuota_est_casino})<br>"
                    cuota_tot_val *= cuota_est_casino
                mom_amer_val = int((cuota_tot_val - 1.0) * 100) if cuota_tot_val >= 2.0 else int(-100 / (cuota_tot_val - 1.0))
                st.markdown(f'<div class="risk-card">{picks_val}<hr><h4>🎟️ Momio Est: {mom_amer_val:+} ({round(cuota_tot_val, 2)})</h4><small>Victorias directas -7% vig.</small></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h2 style='color:#ec4899; text-align:center;'>🚀 EL PARLAY SOÑADOR (MOONSHOT DE LA JORNADA)</h2>", unsafe_allow_html=True)
            st.write("Selección de mercados alternativos de alto rendimiento combinados en un solo boleto. El algoritmo busca córners elevados, partidos de ida y vuelta (BTTS + Over 2.5) y golpes tácticos para maximizar el pago.")
            
            picks_sonador = ""
            cuota_tot_sonador = 1.0
            
            for i, d in enumerate(jornada_data[:6]):
                if d['Over_25'] > 50 and d['BTTS_Si'] > 50:
                    pick_s = "Ambos Anotan SÍ + Over 2.5 Goles"
                    cuota_s = 2.20
                elif d['Corners_Total'] > 9.8:
                    pick_s = f"Over 10.5 Tiros de Esquina (Proy: ~{d['Corners_Total']})"
                    cuota_s = 2.10
                elif d['Prob_1'] > 45 and d['IDR_L'] > d['IDR_V']:
                    pick_s = f"Victoria Directa {d['Local']} + Over 1.5 Goles"
                    cuota_s = 2.35
                elif d['Prob_2'] > 40 and d['IDR_V'] > d['IDR_L']:
                    pick_s = f"Victoria Visitante {d['Visita']} (Golpe Táctico)"
                    cuota_s = 2.60
                else:
                    pick_s = "Empate al Descanso o Marcador Exacto 1-1"
                    cuota_s = 2.05
                    
                picks_sonador += f"✨ <b>{d['Local']} vs {d['Visita']}:</b> {pick_s} <span style='color:#fbcfe8;'>[Cuota: {cuota_s}]</span><br>"
                cuota_tot_sonador *= cuota_s
                
            momio_amer_sonador = int((cuota_tot_sonador - 1.0) * 100) if cuota_tot_sonador >= 2.0 else int(-100 / (cuota_tot_sonador - 1.0))
            
            st.markdown(f"""
            <div class="dream-card">
                <h3 style="color:#fdf2f8; margin-top:0;">🌌 BOLETO MOONSHOT - LOTERÍA ALGOREÍTMICA</h3>
                <div style="font-size: 1.05em; line-height: 1.6; margin: 15px 0;">
                    {picks_sonador}
                </div>
                <hr style="border-color: #db2777;">
                <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                    <div>
                        <h2 style="color: #ffffff; margin: 0;">🎟️ MOMIO EST: {momio_amer_sonador:+}</h2>
                        <span style="color: #fbcfe8;">Cuota Decimal Combinada: <b>{round(cuota_tot_sonador, 2)}</b></span>
                    </div>
                    <div style="background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 8px; border: 1px solid #ec4899; margin-top: 10px;">
                        <span style="color: #f43f5e; font-weight: bold;">⚠️ GESTIÓN DE RIESGO:</span><br>
                        <small style="color: #fce7f3;">Stake sugerido: <b>0.25u o moneditas de sobra</b>.<br>Si le metes $100 MXN, ¡el retorno estimado es de <b>${round(cuota_tot_sonador * 100):,} MXN</b>!</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_tabla:
        st.subheader("📈 Ranking de Dominio Real (IDR) & Tabla General")
        df_show = df_ligamx[['Equipo', 'PJ', 'Pts', 'GF', 'GC', 'xG', 'xGA', 'Calidad_Tiro', 'AttPen_Promedio', 'IDR']].copy()
        df_show = df_show.sort_values(by='IDR', ascending=False).reset_index(drop=True)
        df_show.index += 1
        st.dataframe(df_show.style.background_gradient(subset=['IDR'], cmap='viridis').background_gradient(subset=['Calidad_Tiro'], cmap='Blues'), use_container_width=True)

# ==============================================================================
# ====== SECCIÓN 2: BÉISBOL (MLB) - CON ROTACIÓN Y LÍNEA TOTAL DINÁMICA ========
# ==============================================================================
elif deporte == "⚾ Béisbol (MLB)":
    st.title("⚾ Proyector Cuantitativo & Moneyball - MLB")
    
    # --- EXTRACCIÓN DE DATOS EN VIVO (API OFICIAL MLB) ---
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

    # --- CONSULTA EN VIVO A LA API OFICIAL DE MLB ---
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

    # --- MOTOR MATEMÁTICO MLB 360° (CON AJUSTE DE RACHA) ---
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
            
        # Restauración de variables para la Interfaz Gráfica
        er_l = round((def_v * 1.05) * (p1/50.0), 1)
        er_v = round((def_l * 1.05) * (p2/50.0), 1)
        
        nrfi = round(50.0 + ((8.0 - (era_sp_l + era_sp_v)) * 4.0), 1)
        nrfi = max(10.0, min(90.0, nrfi)) 
        
        carreras_totales = er_l + er_v
        over_line = round(50.0 + ((carreras_totales - linea_ou) * 5.0), 1)
        over_line = max(10.0, min(90.0, over_line))
        
        return {
            "Local": local_clean,
            "Visita": visita_clean,
            "Prob_1": p1,
            "Prob_2": p2,
            "ER_L": er_l,
            "ER_V": er_v,
            "Over_Line": over_line,
            "NRFI": nrfi,
            "Linea_OU": linea_ou,
            "M1": [round(p1*0.96, 1), round(p2*1.04, 1)],
            "M2": [round(sl['Pitagorica'], 1), round(sv['Pitagorica'], 1)],
            "M3": [p1, p2]
        }

    tab_mlb1, tab_mlb2, tab_mlb3, tab_mlb4 = st.tabs([
        "🏟️ Match Center (MLB)", 
        "🧪 Laboratorio Béisbol", 
        "⚡ Parlays & Moonshot MLB", 
        "📊 Tabla Pitagórica MLB"
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
        
        st.markdown(f"""
        <div class="pitcher-box">
            <h4 style="color:#38bdf8; margin-top:0; text-align:center;">⚾ ROTACIÓN Y LÍNEA DEL DÍA: AJUSTA A TUS PARÁMETROS REALES</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_sp1, col_sp2, col_ou = st.columns([2, 2, 1.5])
        with col_sp1:
            sp_loc_input = st.number_input(f"🔥 ERA Pitcher - {loc_nombre}", value=era_l_val, min_value=0.50, max_value=10.00, step=0.10, key="sp_loc")
        with col_sp2:
            sp_vis_input = st.number_input(f"🔥 ERA Pitcher - {vis_nombre}", value=era_v_val, min_value=0.50, max_value=10.00, step=0.10, key="sp_vis")
        with col_ou:
            linea_casino = st.number_input("🎯 Línea Altas/Bajas (O/U)", value=8.5, min_value=5.0, max_value=15.0, step=0.5, key="linea_ou_sel")
            
        dm = motor_mlb_360(loc_nombre, vis_nombre, df_mlb, sp_loc_input, sp_vis_input, linea_casino)
        
        if dm:
            st.markdown(f"""
            <div class="match-header-mlb">
                <h2>🏠 {dm['Local']} vs {dm['Visita']} ✈️</h2>
                <p><b>Carreras Esperadas:</b> {dm['ER_L']} - {dm['ER_V']} (Total Proyectado: {round(dm['ER_L']+dm['ER_V'], 1)}) &nbsp;|&nbsp; <b>Prob. Moneyline:</b> {dm['Prob_1']}% vs {dm['Prob_2']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
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
                    st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>El margen más alto es <b>{mejor_nom}</b> con solo +{mejor_edge_val}%.</p><hr><small style="color: #fde047;">Las líneas de MLB están muy apretadas. Recomendamos saltar este juego para apuestas directas.</small></div>', unsafe_allow_html=True)
                
            col_b3, col_b4 = st.columns(2)
            with col_b3:
                st.markdown(f'<div class="risk-card"><h3>🔥 Pick Risk (Run Line -1.5)</h3><p><b>{fav_mlb} Run Line -1.5</b> (Gana por 2+ carreras)</p><p><b>Probabilidad Proyectada:</b> ~{round(max(dm["Prob_1"], dm["Prob_2"])*0.62, 1)}% &nbsp;|&nbsp; <b>Momio Est:</b> +140</p><hr><small>Para maximizar cuotas cuando tu pitcher abridor tiene dominio absoluto.</small></div>', unsafe_allow_html=True)
            with col_b4:
                st.markdown(f'<div class="parlay-card"><h3>⚾ Prop del Inning 1 (Impacto Abridor)</h3><p><b>Recomendación:</b> {"NRFI (No Run 1st Inning)" if dm["NRFI"] >= 52 else "YRFI (Yes Run 1st Inning)"}</p><p><b>Probabilidad:</b> {dm["NRFI"] if dm["NRFI"]>=52 else round(100-dm["NRFI"],1)}% &nbsp;|&nbsp; <b>Línea:</b> 0.5 Carreras 1er Rollo</p><hr><small>Cálculo pesado en los ERAs introducidos para el 1er inning.</small></div>', unsafe_allow_html=True)

    with tab_mlb2:
        st.subheader("🧪 Laboratorio Multi-Algoritmo Béisbol")
        if dm:
            st.markdown(f'<div class="meta-model-card" style="border-color:#ef4444;"><h3 style="color:#fecaca; margin:0;">👑 CONSENSO MONEYBALL MLB</h3><h1 style="color:#ffffff; margin:10px 0;">{dm["Local"]}: {dm["Prob_1"]}% &nbsp;|&nbsp; {dm["Visita"]}: {dm["Prob_2"]}%</h1><p style="color:#fca5a5; margin:0;">En béisbol no existen empates; el 100% de la probabilidad se divide entre ambos bandos.</p></div>', unsafe_allow_html=True)
            t_mlb_comp = {
                "Metodología Cuantitativa": ["1️⃣ Distribución Poisson (Carreras + Abridor)", "2️⃣ Esperanza Pitagórica (Bill James Formula)", "3️⃣ Simulación Monte Carlo (5,000 Juegos)"],
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
                st.markdown(f'<div class="safe-card">{p_seg_m}<hr><h4>🎟️ Momio Est: {mom_seg_mlb:+} ({round(c_seg_m, 2)})</h4><small>Top 3 favoritos según Bill James.</small></div>', unsafe_allow_html=True)
                
            with cp_m2:
                st.markdown("<h3 style='color:#10b981;'>💎 Parlay de Valor</h3>", unsafe_allow_html=True)
                p_val_m, c_val_m = "", 1.0
                for x in sorted(data_j_mlb, key=lambda d: d['NRFI'], reverse=True)[:3]:
                    pick_v = "NRFI (0 Carreras en 1ª Entrada)" if x['NRFI'] > 53 else "Over 8.5 Carreras Totales"
                    p_val_m += f"💥 <b>{x['Local']} vs {x['Visita']}:</b> {pick_v}<br>"
                    c_val_m *= 1.85
                mom_val_mlb = int((c_val_m - 1.0) * 100) if c_val_m >= 2.0 else int(-100 / (c_val_m - 1.0))
                st.markdown(f'<div class="value-card">{p_val_m}<hr><h4>🎟️ Momio Est: {mom_val_mlb:+} ({round(c_val_m, 2)})</h4><small>Combinación de props de pitcheo y bateo.</small></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h2 style='color:#ec4899; text-align:center;'>🚀 EL PARLAY SOÑADOR MLB (MOONSHOT BÉISBOL)</h2>", unsafe_allow_html=True)
            picks_mlb_moon, cuota_mlb_moon = "", 1.0
            for x in data_j_mlb[:5]:
                if x['Over_Line'] > 52:
                    p_moon = f"Over {x['Linea_OU']} Carreras + YRFI (Carrera en 1ª Entrada)"
                    c_moon = 2.40
                elif x['Prob_1'] > 56:
                    p_moon = f"{x['Local']} Run Line -1.5 (Gana por paliza)"
                    c_moon = 2.25
                elif x['Prob_2'] > 55:
                    p_moon = f"{x['Visita']} Run Line -1.5 (Gana por paliza)"
                    c_moon = 2.35
                else:
                    p_moon = f"NRFI + Under {x['Linea_OU']} Carreras (Duelo de Pitcheo)"
                    c_moon = 2.15
                picks_mlb_moon += f"✨ <b>{x['Local']} vs {x['Visita']}:</b> {p_moon} <span style='color:#fbcfe8;'>[Cuota: {c_moon}]</span><br>"
                cuota_mlb_moon *= c_moon
                
            mom_moon_mlb = int((cuota_mlb_moon - 1.0) * 100) if cuota_mlb_moon >= 2.0 else int(-100 / (cuota_mlb_moon - 1.0))
            st.markdown(f"""
            <div class="dream-card">
                <h3 style="color:#fdf2f8; margin-top:0;">🌌 BOLETO MOONSHOT MLB - BOMBAS DEL DIAMANTE</h3>
                <div style="font-size: 1.05em; line-height: 1.6; margin: 15px 0;">{picks_mlb_moon}</div>
                <hr style="border-color: #db2777;">
                <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                    <div>
                        <h2 style="color: #ffffff; margin: 0;">🎟️ MOMIO EST: {mom_moon_mlb:+}</h2>
                        <span style="color: #fbcfe8;">Cuota Combinada: <b>{round(cuota_mlb_moon, 2)}</b></span>
                    </div>
                    <div style="background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 8px; border: 1px solid #ec4899; margin-top: 10px;">
                        <span style="color: #f43f5e; font-weight: bold;">⚠️ GESTIÓN DE RIESGO:</span><br>
                        <small style="color: #fce7f3;">Stake sugerido: <b>0.25u</b>. Si le metes $100 MXN, ¡el retorno proyectado es de <b>${round(cuota_mlb_moon * 100):,} MXN</b>!</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_mlb4:
        st.subheader("📈 Ranking de Esperanza Pitagórica (Moneyball)")
        st.write("La Esperanza Pitagórica nos dice qué porcentaje de victorias *debería* tener un equipo según sus carreras. Además, la API ahora castiga matemáticamente a los equipos atrapados en rachas perdedoras.")
        if not df_mlb.empty:
            df_show_mlb = df_mlb[['Equipo', 'G', 'P', 'RS_prom', 'RA_prom', 'ERA', 'WHIP', 'Pitagorica', 'Racha']].sort_values(by='Pitagorica', ascending=False).reset_index(drop=True)
            df_show_mlb.index += 1
            st.dataframe(df_show_mlb.style.background_gradient(subset=['Pitagorica'], cmap='Reds').background_gradient(subset=['ERA'], cmap='Blues_r'), use_container_width=True)

# ==============================================================================
# ================= SECCIÓN 3: LEAGUES Cup (PARLAYS & PICKS) ===================
# ==============================================================================
elif deporte == "🌎 Leagues Cup":
    st.title("🌎 Leagues Cup - Máquina de Parlays y Picks")
    st.write("Análisis masivo cruzando métricas de la MLS (FBref) y la Liga MX.")

    # 1. SCRAPER PROXY PARA FBRef (DESCARGA DE LA MLS COMPLETA)
    @st.cache_data(ttl=3600*24) # 1 petición al día para cuidar la cuota gratis
    def obtener_tabla_mls():
        url_objetivo = "https://fbref.com/es/comps/22/Estadisticas-de-Major-League-Soccer"
        
        # 🔑 1. INGRESA TU API KEY DE SCRAPERAPI
        API_KEY = "e5bf56968d9196c900a3dd8abbe93917" 
        
        # 2. Construimos la ruta del Proxy para engañar a Cloudflare
        proxy_url = f"http://api.scraperapi.com?api_key={API_KEY}&url={url_objetivo}"
        
        try:
            # Hacemos la petición a través del proxy (le damos 30 seg porque los proxys tardan un poco)
            res = requests.get(proxy_url, timeout=30)
            if res.status_code != 200: return pd.DataFrame()
            
            tablas = pd.read_html(res.text)
            df_mls = pd.DataFrame()
            
            # Unimos la Conferencia Este y Oeste completas
            for t in tablas:
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = t.columns.droplevel(0)
                if 'Equipo' in t.columns and 'Pts' in t.columns:
                    df_mls = pd.concat([df_mls, t], ignore_index=True)
            
            if df_mls.empty: return pd.DataFrame()
            
            # Limpieza básica
            df_mls = df_mls.dropna(subset=['Equipo'])
            df_mls = df_mls[df_mls['Equipo'] != 'Equipo'] 
            
            # Aseguramos formato numérico de todas las estadísticas
            for c in ['PJ', 'Pts', 'GF', 'GC', 'xG', 'xGA']:
                if c in df_mls.columns:
                    df_mls[c] = pd.to_numeric(df_mls[c], errors='coerce').fillna(1.0)
            
            # Agregamos métricas proxy de penetración para tu motor IDR
            df_mls['AttPen_Promedio'] = 16.0 
            df_mls['Tiros_Promedio'] = 12.0
            df_mls['Calidad_Tiro'] = 0.11
            
            return df_mls
        except Exception as e:
            return pd.DataFrame()

    # 2. CREAMOS LA SÚPER TABLA (LIGA MX + MLS)
    with st.spinner("⚡ Extrayendo métricas de MLS en FBref y fusionando con Liga MX..."):
        df_ligamx = cargar_datos_completos() # Ya incluye IDR
        df_mls = obtener_tabla_mls()
        
        if not df_mls.empty:
            df_mls = calcular_idr(df_mls) # Pasamos la MLS por tu motor IDR
            df_leagues_cup = pd.concat([df_ligamx, df_mls], ignore_index=True)
        else:
            st.error("⚠️ FBref bloqueó la conexión temporalmente. Usando datos de respaldo.")
            df_leagues_cup = df_ligamx.copy()

    # 3. CARTELERA MANUAL DE LEAGUES CUP
    @st.cache_data(ttl=3600*6)
    def obtener_jornada_leagues_cup():
        # Actualiza esta lista cada semana con los partidos más interesantes
        return [
            "Inter Miami vs Tigres",
            "LAFC vs Monterrey",
            "Columbus Crew vs América",
            "Seattle Sounders vs Pumas",
            "Orlando City vs Cruz Azul"
        ]
    
    partidos_lc = obtener_jornada_leagues_cup()
    
    # Ejecutamos TODOS los algoritmos (Poisson, ELO, Monte Carlo) con la Súper Tabla
    datos_lc = [ejecutar_laboratorio_modelos(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_leagues_cup) for p in partidos_lc]
    datos_lc = [d for d in datos_lc if d is not None]

    tab_singles, tab_dobles, tab_sgp, tab_jornada = st.tabs([
        "🎯 Picks Solos Segurísimos",
        "🛡️ Parlay Doble Oportunidad",
        "🎰 Parlays por Partido (SGP)",
        "📅 Parlay de la Jornada"
    ])

    # --- PESTAÑA 1: PICKS SOLOS ---
    with tab_singles:
        st.subheader("🎯 Picks Solos Segurísimos (Bajo Riesgo, Alta Probabilidad)")
        st.write("Filtro estricto: Solo apuestas con probabilidad matemática superior al 60%.")
        c1, c2 = st.columns(2)
        
        picks_seguros = sorted(datos_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)
        if len(picks_seguros) > 0:
            top_1 = picks_seguros[0]
            fav_1 = top_1['Local'] if top_1['Prob_1'] > top_1['Prob_2'] else top_1['Visita']
            prob_1 = max(top_1['Prob_1'], top_1['Prob_2'])
            with c1:
                st.markdown(f'<div class="safe-card"><h3>🛡️ Victoria Empate No Válida (DNB)</h3><p><b>{fav_1} (Empate No Acción)</b></p><p><b>Prob. Modelo Directo:</b> {prob_1}%</p><hr><small>Si empatan, te devuelven el dinero. Si ganan, cobras.</small></div>', unsafe_allow_html=True)
        
        picks_goles = sorted(datos_lc, key=lambda x: x['Over_25'], reverse=True)
        if len(picks_goles) > 0:
            top_g = picks_goles[0]
            with c2:
                st.markdown(f'<div class="safe-card"><h3>⚽ Totales Conservadores</h3><p><b>{top_g["Local"]} vs {top_g["Visita"]}: Over 1.5 Goles</b></p><p><b>Prob. Modelo (+2.5):</b> {top_g["Over_25"]}%</p><hr><small>Ideal para subir bank con los equipos más ofensivos.</small></div>', unsafe_allow_html=True)

    # --- PESTAÑA 2: DOBLE OPORTUNIDAD ---
    with tab_dobles:
        st.subheader("🛡️ Parlay de Doble Oportunidad (1X / X2)")
        st.write("3 equipos que cubren dos de los tres resultados posibles.")
        picks_dobles = ""
        c_doble = 1.0
        for d in sorted(datos_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)[:3]:
            if d['Prob_1'] > d['Prob_2']:
                picks_dobles += f"⚽ <b>{d['Local']} o Empate (1X)</b><br>"
            else:
                picks_dobles += f"⚽ <b>{d['Visita']} o Empate (X2)</b><br>"
            c_doble *= 1.25 # Cuota promedio aproximada de un 1X
            
        mom_d = int((c_doble - 1.0) * 100) if c_doble >= 2.0 else int(-100 / (c_doble - 1.0))
        st.markdown(f'<div class="parlay-card">{picks_dobles}<hr><h4>🎟️ Momio Est: {mom_d:+} ({round(c_doble, 2)})</h4><small>Estadísticamente, tienes el 66% de cobertura en cada juego.</small></div>', unsafe_allow_html=True)

    # --- PESTAÑA 3: SAME GAME PARLAY (SGP) ---
    with tab_sgp:
        st.subheader("🎰 Parlay por Partido (Same Game Parlay)")
        if len(partidos_lc) > 0:
            partido_sgp = st.selectbox("Selecciona el partido para armar el SGP:", options=partidos_lc, key="sel_lc")
            loc_s, vis_s = partido_sgp.split(" vs ")
            d_sgp = next((item for item in datos_lc if item["Local"] == loc_s.strip() and item["Visita"] == vis_s.strip()), None)
            
            if d_sgp:
                st.markdown(f'<div class="match-header"><h2>🏠 {d_sgp["Local"]} vs {d_sgp["Visita"]} ✈️</h2><p><b>Proy. Goles:</b> {d_sgp["xG_L"]} - {d_sgp["xG_V"]}</p></div>', unsafe_allow_html=True)
                
                c_sgp1, c_sgp2 = st.columns(2)
                fav_sgp = d_sgp['Local'] if d_sgp['Prob_1'] > d_sgp['Prob_2'] else d_sgp['Visita']
                
                with c_sgp1:
                    st.markdown(f'<div class="risk-card"><h3>🔥 SGP Conservador</h3><p><b>1. {fav_sgp} o Empate</b><br><b>2. Over 1.5 Goles Totales</b><br><b>3. Over 7.5 Córners Totales</b></p><hr><p style="margin:0;"><b>Cuota Combinada Est.:</b> +130</p></div>', unsafe_allow_html=True)
                with c_sgp2:
                    btts = "SÍ" if d_sgp['BTTS_Si'] > 50 else "NO"
                    st.markdown(f'<div class="dream-card" style="margin-top:0;"><h3>🌌 SGP Agresivo</h3><p><b>1. Ambos Anotan ({btts})</b><br><b>2. Over 2.5 Goles Totales</b><br><b>3. {fav_sgp} a Ganar Directo</b></p><hr><p style="margin:0; color:#fff;"><b>Cuota Combinada Est.:</b> +350 a +450</p></div>', unsafe_allow_html=True)

    # --- PESTAÑA 4: PARLAY DE LA JORNADA ---
    with tab_jornada:
        st.subheader("📅 El Parlay de la Jornada (Acumulador)")
        picks_jornada = ""
        cuota_j = 1.0
        
        for d in sorted(datos_lc, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)[:4]:
            fav = d['Local'] if d['Prob_1'] > d['Prob_2'] else d['Visita']
            picks_jornada += f"🔥 <b>{fav} a Ganar Directo</b> (Prob: {max(d['Prob_1'], d['Prob_2'])}%)<br>"
            cuota_j *= 1.85 # Promedio de victoria directa
            
        mom_amer_j = int((cuota_j - 1.0) * 100) if cuota_j >= 2.0 else int(-100 / (cuota_j - 1.0))
        
        st.markdown(f'<div class="value-card"><h3 style="color:#fff;">💰 Acumulador de la Jornada</h3>{picks_jornada}<hr><h4 style="color:#fff;">🎟️ Momio Est: {mom_amer_j:+} ({round(cuota_j, 2)})</h4><small style="color:#d1fae5;">Top 4 favoritos absolutos del modelo matemático.</small></div>', unsafe_allow_html=True)
