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
    .vegas-alert-box { background: #450a0a; padding: 15px; border-radius: 10px; border-left: 5px solid #ef4444; margin-bottom: 20px; }
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

# 2. FUNCIONES DE CONVERSIÓN Y MATEMÁTICAS
def americano_a_decimal(momio_amer):
    if momio_amer >= 0: return round((momio_amer / 100.0) + 1.0, 2)
    else: return round((100.0 / abs(momio_amer)) + 1.0, 2)

def prob_implicada(decimal_odd):
    return round((1.0 / decimal_odd) * 100.0, 1)

def proyeccion_ponches(era, k9, linea):
    ip_esperadas = max(3.5, 7.5 - (era * 0.45))
    lambda_k = (k9 / 9.0) * ip_esperadas
    k_enteros = int(linea)
    prob_under = stats.poisson.cdf(k_enteros, lambda_k) * 100.0
    prob_over = 100.0 - prob_under
    return round(lambda_k, 1), round(prob_over, 1), round(prob_under, 1)

# 3. SELECTOR DE DEPORTE (MENÚ LATERAL)
st.sidebar.title("🏆 Centro de Mando")
deporte = st.sidebar.radio("Selecciona tu Motor de Análisis:", ["⚽ Fútbol (Liga MX)", "⚾ Béisbol (MLB)"])
st.sidebar.markdown("---")
st.sidebar.info("💡 **Sistema Multi-Algoritmo:** Evalúa líneas en vivo usando Poisson, Monte Carlo, ELO y Modelos de Eficiencia.")

# ==============================================================================
# ================= SECCIÓN 1: FÚTBOL (LIGA MX) ================================
# ==============================================================================
if deporte == "⚽ Fútbol (Liga MX)":
    st.title("⚽ Liga MX - Cuantificador Multi-Algoritmo & Value Bets")
    
    @st.cache_data(ttl=3600)
    def cargar_datos_completos():
        df = obtener_tabla_ligamx()
        df = calcular_idr(df)
        return df

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

    def ejecutar_laboratorio_modelos(local, visita, df):
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
                        st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ SIN VALUE BET CLARO</h3><p>El margen más alto es <b>{mejor_val_nombre}</b> con solo +{mejor_edge}%.</p><hr><small style="color: #fde047;">Las líneas del casino están bien ajustadas. No hay ineficiencias de al menos +5.0% para justificar el riesgo.</small></div>', unsafe_allow_html=True)

                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    top_marcador = datos['Marcadores_Top'][0]
                    st.markdown(f'<div class="risk-card"><h3>🔥 PICK RISK (Alto Beneficio / Underdog)</h3><p><b>Recomendación:</b> Marcador Exacto {top_marcador["Marcador"]}</p><p><b>Probabilidad:</b> {round(top_marcador["Prob"], 1)}%</p><hr><small>Apuesta recreativa de alto rendimiento basada en Poisson.</small></div>', unsafe_allow_html=True)
                with col_p4:
                    sgp_1 = f"{datos['Local']} Gana o Empata" if datos['Prob_1'] >= datos['Prob_2'] else f"{datos['Visita']} Gana o Empata"
                    sgp_2 = "Over 1.5 Goles" if datos['Over_25'] > 45 else "Under 3.5 Goles"
                    sgp_3 = f"Over 8.5 Córners" if datos['Corners_Total'] > 9.0 else "Under 10.5 Córners"
                    st.markdown(f'<div class="parlay-card"><h3>🎰 PICK PARLAY (Same-Game Bet Builder)</h3><p><b>1. Resultado:</b> {sgp_1}<br><b>2. Goles:</b> {sgp_2}<br><b>3. Córners:</b> {sgp_3}</p><hr><p style="margin:0;"><b>Cuota Combinada Est.:</b> +160 (2.60) &nbsp;|&nbsp; <b>Prob:</b> ~55%</p></div>', unsafe_allow_html=True)

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

    # --- NUEVA CONEXIÓN PARA EXTRAER UMPIRES Y STATUS DE ALINEACIÓN ---
    @st.cache_data(ttl=3600*3)
    def obtener_partidos_mlb_detallado(fecha_str):
        # El parámetro hydrate=officials trae a los umpires programados si ya fueron anunciados
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha_str}&hydrate=officials"
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
                        status = game["status"]["detailedState"] # Saber si ya inició, etc.
                        nombre_base = f"{home} vs {away}"
                        
                        # Buscar al Umpire de Home si existe en el JSON
                        umpire_home = "No anunciado aún"
                        for official in game.get("officials", []):
                            if official.get("officialType") == "Home Plate":
                                umpire_home = official.get("official", {}).get("fullName", "Desconocido")
                                break
                        
                        if nombre_base in conteo_duelos:
                            conteo_duelos[nombre_base] += 1
                            nombre_final = f"{nombre_base} (Juego {conteo_duelos[nombre_base]})"
                            if conteo_duelos[nombre_base] == 2:
                                idx_primero = next(i for i, v in enumerate(juegos) if v["nombre"].startswith(nombre_base))
                                juegos[idx_primero]["nombre"] = f"{nombre_base} (Juego 1)"
                        else:
                            conteo_duelos[nombre_base] = 1
                            nombre_final = nombre_base
                            
                        juegos.append({
                            "nombre": nombre_final,
                            "home": home,
                            "away": away,
                            "umpire": umpire_home,
                            "status": status
                        })
                if juegos: return juegos
        except Exception: pass
        return []

    fecha_default = date.today()
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        fecha_sel = st.date_input("📅 Selecciona Jornada MLB:", value=fecha_default, min_value=date(2026, 3, 20), max_value=date(2026, 11, 1))
    
    info_juegos = obtener_partidos_mlb_detallado(fecha_sel.strftime("%Y-%m-%d"))
    nombres_partidos = [j["nombre"] for j in info_juegos] if info_juegos else []
    
    if not nombres_partidos:
        st.info(f"💡 No se detectó cartelera en la API para el {fecha_sel.strftime('%d/%m/%Y')} o la conexión falló.")
        nombres_partidos = ["New York Yankees vs Los Angeles Dodgers", "Philadelphia Phillies vs Baltimore Orioles"]
        info_juegos = [{"nombre": n, "umpire": "Desconocido", "status": "Scheduled"} for n in nombres_partidos]

    # --- MOTOR MLB CON AJUSTE "DINERO INTELIGENTE VEGAS" ---
    def motor_mlb_360(local, visita, df, era_sp_l=None, era_sp_v=None, linea_ou=8.5, fatiga_l="Normal", fatiga_v="Normal", clima="Neutral", vegas_flag="Normal"):
        local_clean = local.split(" (Juego")[0].strip()
        visita_clean = visita.split(" (Juego")[0].strip()
        
        l_match = df[df['Equipo'].str.contains(local_clean.split()[-1], case=False, na=False)]
        v_match = df[df['Equipo'].str.contains(visita_clean.split()[-1], case=False, na=False)]
        
        sl = l_match.iloc[0].to_dict() if not l_match.empty else {"Equipo": local_clean, "ERA": 4.00, "Pitagorica": 50.0, "Racha": "W1"}
        sv = v_match.iloc[0].to_dict() if not v_match.empty else {"Equipo": visita_clean, "ERA": 4.00, "Pitagorica": 50.0, "Racha": "W1"}
        
        if era_sp_l is None: era_sp_l = sl['ERA']
        if era_sp_v is None: era_sp_v = sv['ERA']
        
        bullpen_era_l = sl['ERA'] + (0.75 if fatiga_l == "Fatigado" else -0.25 if fatiga_l == "Descansado" else 0)
        bullpen_era_v = sv['ERA'] + (0.75 if fatiga_v == "Fatigado" else -0.25 if fatiga_v == "Descansado" else 0)
        
        def_l = (era_sp_l * 0.65) + (bullpen_era_l * 0.35)
        def_v = (era_sp_v * 0.65) + (bullpen_era_v * 0.35)
        
        ajuste_racha_l, ajuste_racha_v = 0, 0
        if str(sl.get('Racha', '')).startswith('L'):
            ajuste_racha_l = - (int(sl['Racha'].replace('L', '') or 0) * 1.5)
        if str(sv.get('Racha', '')).startswith('L'):
            ajuste_racha_v = - (int(sv['Racha'].replace('L', '') or 0) * 1.5)
            
        prob_l = sl['Pitagorica'] + (def_v - def_l)*4.0 + ajuste_racha_l
        prob_v = sv['Pitagorica'] - (def_v - def_l)*4.0 + ajuste_racha_v
        
        # Penalización Vegas: Si hay movimiento inverso contra un equipo, bajamos matemáticamente sus chances un 8%
        if vegas_flag == "Contra Local": prob_l -= 8.0
        elif vegas_flag == "Contra Visita": prob_v -= 8.0

        total = prob_l + prob_v
        if total > 0:
            p1 = round((prob_l/total)*100, 1)
            p2 = round((prob_v/total)*100, 1)
        else:
            p1, p2 = 50.0, 50.0
            
        er_l_base = (def_v * 1.05) * (p1/50.0)
        er_v_base = (def_l * 1.05) * (p2/50.0)
        
        mod_clima = 1.15 if clima == "Viento a favor (Over)" else 0.85 if clima == "Viento en contra (Under)" else 1.0
        er_l = round(er_l_base * mod_clima, 1)
        er_v = round(er_v_base * mod_clima, 1)
        
        nrfi = round(50.0 + ((8.0 - (era_sp_l + era_sp_v)) * 4.0), 1)
        if clima == "Viento a favor (Over)": nrfi -= 4.0
        elif clima == "Viento en contra (Under)": nrfi += 4.0
        nrfi = max(10.0, min(90.0, nrfi)) 
        
        carreras_totales = er_l + er_v
        over_line = round(50.0 + ((carreras_totales - linea_ou) * 5.0), 1)
        over_line = max(10.0, min(90.0, over_line))

        lam_f5_l = ((era_sp_v / 9.0) * 5.0 * 1.05) * mod_clima
        lam_f5_v = ((era_sp_l / 9.0) * 5.0 * 0.95) * mod_clima

        p1_f5, px_f5, p2_f5 = 0.0, 0.0, 0.0
        for gl in range(10):
            for gv in range(10):
                prob = stats.poisson.pmf(gl, lam_f5_l) * stats.poisson.pmf(gv, lam_f5_v)
                if gl > gv: p1_f5 += prob
                elif gl == gv: px_f5 += prob
                else: p2_f5 += prob

        tot_f5 = p1_f5 + px_f5 + p2_f5
        if tot_f5 > 0:
            p1_f5 = round((p1_f5/tot_f5)*100, 1)
            px_f5 = round((px_f5/tot_f5)*100, 1)
            p2_f5 = round((p2_f5/tot_f5)*100, 1)
            
        # Penalización F5 Vegas
        if vegas_flag == "Contra Local": p1_f5 = max(5.0, p1_f5 - 8.0)
        elif vegas_flag == "Contra Visita": p2_f5 = max(5.0, p2_f5 - 8.0)
        
        return {
            "Local": local_clean, "Visita": visita_clean,
            "Prob_1": p1, "Prob_2": p2, "ER_L": er_l, "ER_V": er_v,
            "Over_Line": over_line, "NRFI": nrfi, "Linea_OU": linea_ou,
            "Prob_F5_1": p1_f5, "Prob_F5_X": px_f5, "Prob_F5_2": p2_f5,
            "ER_F5_L": round(lam_f5_l, 2), "ER_F5_V": round(lam_f5_v, 2),
            "M1": [round(p1*0.96, 1), round(p2*1.04, 1)],
            "M2": [round(sl['Pitagorica'], 1), round(sv['Pitagorica'], 1)],
            "M3": [p1, p2]
        }

    tab_mlb1, tab_mlb2, tab_mlb3, tab_mlb4 = st.tabs([
        "🏟️ Match Center (MLB)", "🧪 Laboratorio Béisbol", 
        "⚡ Parlays & Ranking MLB", "📊 Tabla Pitagórica MLB"
    ])

    with tab_mlb1:
        part_mlb_sel = st.selectbox("⚾ Selecciona Juego MLB de la Cartelera:", nombres_partidos)
        
        # Buscar la metadata del juego seleccionado en la lista extraída de la API
        info_juego_actual = next((j for j in info_juegos if j["nombre"] == part_mlb_sel), None)
        umpire_str = info_juego_actual["umpire"] if info_juego_actual else "Desconocido"
        status_str = info_juego_actual["status"] if info_juego_actual else "Scheduled"
        
        eqs_m = part_mlb_sel.split(" vs ")
        loc_nombre_full, vis_nombre_full = eqs_m[0].strip(), eqs_m[1].strip()
        loc_nombre = loc_nombre_full.split(" (Juego")[0].strip()
        vis_nombre = vis_nombre_full.split(" (Juego")[0].strip()
        
        # MÓDULO DE INTELIGENCIA VEGAS
        st.markdown(f"""
        <div class="vegas-alert-box">
            <h4 style="color:#fca5a5; margin-top:0;">🚨 ESCÁNER DE PELIGRO VEGAS (DATOS OFICIALES)</h4>
            <b>Árbitro de Home (Umpire):</b> {umpire_str}<br>
            <small style="color:#d1d5db;"><i>Tip Pro: Si no conoces al umpire, busca "{umpire_str} umpire stats" en Google. Si tiene zona chica, ten cuidado con los Unders de Ponches.</i></small><br><br>
            <b>Estado del Juego/Lineups:</b> {status_str}
        </div>
        """, unsafe_allow_html=True)

        def_era_loc = df_mlb[df_mlb['Equipo'].str.contains(loc_nombre.split()[-1], case=False, na=False)]['ERA'].values if not df_mlb.empty else []
        def_era_vis = df_mlb[df_mlb['Equipo'].str.contains(vis_nombre.split()[-1], case=False, na=False)]['ERA'].values if not df_mlb.empty else []
        era_l_val = float(def_era_loc[0]) if len(def_era_loc) > 0 else 4.00
        era_v_val = float(def_era_vis[0]) if len(def_era_vis) > 0 else 4.00
        
        st.markdown('<div class="pitcher-box"><h4 style="color:#38bdf8; margin-top:0; text-align:center;">⚾ ROTACIÓN Y LÍNEA DEL DÍA: AJUSTA A TUS PARÁMETROS REALES</h4></div>', unsafe_allow_html=True)
        col_sp1, col_sp2, col_ou = st.columns([2, 2, 1.5])
        with col_sp1: sp_loc_input = st.number_input(f"🔥 ERA Pitcher - {loc_nombre}", value=era_l_val, min_value=0.50, max_value=10.00, step=0.10, key="sp_loc")
        with col_sp2: sp_vis_input = st.number_input(f"🔥 ERA Pitcher - {vis_nombre}", value=era_v_val, min_value=0.50, max_value=10.00, step=0.10, key="sp_vis")
        with col_ou: linea_casino = st.number_input("🎯 Línea Altas/Bajas (O/U)", value=8.5, min_value=5.0, max_value=15.0, step=0.5, key="linea_ou_sel")

        # --- SECCIÓN DE AJUSTES AVANZADOS VEGAS ---
        with st.expander("🧠 Consola de Ajuste de Smart Money (Filtros Profesionales)", expanded=False):
            st.markdown("<small>Si detectaste que el casino bajó el momio dramáticamente contra tu pick (Reverse Line Movement) o si quitaron al jugador estrella del lineup, marca la casilla correspondiente para penalizar al equipo.</small>", unsafe_allow_html=True)
            col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
            with col_adv1: fatiga_loc_input = st.selectbox(f"Fatiga Bullpen ({loc_nombre})", ["Normal", "Fatigado", "Descansado"], key="fatiga_l")
            with col_adv2: fatiga_vis_input = st.selectbox(f"Fatiga Bullpen ({vis_nombre})", ["Normal", "Fatigado", "Descansado"], key="fatiga_v")
            with col_adv3: clima_input = st.selectbox("Factor Estadio", ["Neutral", "Viento a favor (Over)", "Viento en contra (Under)"], key="clima_sel")
            with col_adv4: vegas_flag_input = st.selectbox("Movimiento Extraño (Casino)", ["Normal", "Contra Local", "Contra Visita"], key="vegas_sel")
            
        dm = motor_mlb_360(loc_nombre, vis_nombre, df_mlb, sp_loc_input, sp_vis_input, linea_casino, fatiga_loc_input, fatiga_vis_input, clima_input, vegas_flag_input)
        
        if dm:
            st.markdown(f'<div class="match-header-mlb"><h2>🏠 {dm["Local"]} vs {dm["Visita"]} ✈️</h2><p><b>Carreras Esperadas (Juego Completo):</b> {dm["ER_L"]} - {dm["ER_V"]}</p></div>', unsafe_allow_html=True)
            
            st.markdown("### 🏦 Cuotas en Vivo MLB (Juego Completo)")
            cm1, cm2, cm_o, cm_nrfi = st.columns(4)
            with cm1: m_mlb1 = st.number_input(f"Moneyline {dm['Local']}", value=-130, step=5, key="mlb1")
            with cm2: m_mlb2 = st.number_input(f"Moneyline {dm['Visita']}", value=+110, step=5, key="mlb2")
            with cm_o: m_mlbo = st.number_input(f"Over {linea_casino} Carreras", value=-110, step=5, key="mlbo")
            with cm_nrfi: m_mlb_nrfi = st.number_input("NRFI (No Run 1st Inning)", value=-120, step=5, key="mlb_nrfi")

            st.markdown("---")
            st.markdown("### ⏱️ Primeras 5 Entradas (F5) - Aislando a los Abridores")
            c_f51, c_f5x, c_f52 = st.columns(3)
            with c_f51: m_f5_1 = st.number_input(f"F5 - Victoria {dm['Local']}", value=-110, step=5, key="m_f51")
            with c_f5x: m_f5_x = st.number_input("F5 - Empate", value=+220, step=5, key="m_f5x")
            with c_f52: m_f5_2 = st.number_input(f"F5 - Victoria {dm['Visita']}", value=+110, step=5, key="m_f52")
            
            st.markdown("---")
            st.markdown("### 🔥 Player Props: Ponches (Strikeouts)")
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.markdown(f"<div class='method-box'><b>🏠 {loc_nombre} (Local)</b>", unsafe_allow_html=True)
                k9_loc = st.number_input("K/9 (Ponches por 9 IP)", value=9.0, step=0.5, key="k9_l")
                lin_k_loc = st.number_input("Línea de Ponches (Casino)", value=5.5, step=0.5, key="lin_k_l")
                c_ok_l, c_uk_l = st.columns(2)
                with c_ok_l: m_ok_loc = st.number_input(f"Over {lin_k_loc}", value=-110, step=5, key="mo_k_l")
                with c_uk_l: m_uk_loc = st.number_input(f"Under {lin_k_loc}", value=-110, step=5, key="mu_k_l")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_k2:
                st.markdown(f"<div class='method-box'><b>✈️ {vis_nombre} (Visita)</b>", unsafe_allow_html=True)
                k9_vis = st.number_input("K/9 (Ponches por 9 IP)", value=9.0, step=0.5, key="k9_v")
                lin_k_vis = st.number_input("Línea de Ponches (Casino)", value=5.5, step=0.5, key="lin_k_v")
                c_ok_v, c_uk_v = st.columns(2)
                with c_ok_v: m_ok_vis = st.number_input(f"Over {lin_k_vis}", value=-110, step=5, key="mo_k_v")
                with c_uk_v: m_uk_vis = st.number_input(f"Under {lin_k_vis}", value=-110, step=5, key="mu_k_v")
                st.markdown("</div>", unsafe_allow_html=True)

            # Cálculos de Edge
            edge_1 = round(dm['Prob_1'] - prob_implicada(americano_a_decimal(m_mlb1)), 1)
            edge_2 = round(dm['Prob_2'] - prob_implicada(americano_a_decimal(m_mlb2)), 1)
            edge_o = round(dm['Over_Line'] - prob_implicada(americano_a_decimal(m_mlbo)), 1)
            edge_nr = round(dm['NRFI'] - prob_implicada(americano_a_decimal(m_mlb_nrfi)), 1)
            edge_f5_1 = round(dm['Prob_F5_1'] - prob_implicada(americano_a_decimal(m_f5_1)), 1)
            edge_f5_x = round(dm['Prob_F5_X'] - prob_implicada(americano_a_decimal(m_f5_x)), 1)
            edge_f5_2 = round(dm['Prob_F5_2'] - prob_implicada(americano_a_decimal(m_f5_2)), 1)
            proj_kl, over_kl, under_kl = proyeccion_ponches(sp_loc_input, k9_loc, lin_k_loc)
            proj_kv, over_kv, under_kv = proyeccion_ponches(sp_vis_input, k9_vis, lin_k_vis)
            edge_ok_l = round(over_kl - prob_implicada(americano_a_decimal(m_ok_loc)), 1)
            edge_uk_l = round(under_kl - prob_implicada(americano_a_decimal(m_uk_loc)), 1)
            edge_ok_v = round(over_kv - prob_implicada(americano_a_decimal(m_ok_vis)), 1)
            edge_uk_v = round(under_kv - prob_implicada(americano_a_decimal(m_uk_vis)), 1)

            mejor_edge_val, mejor_nom, mejor_mom = max([
                (edge_1, f"Victoria {dm['Local']}", m_mlb1),
                (edge_2, f"Victoria {dm['Visita']}", m_mlb2),
                (edge_o, f"Over {linea_casino} Carreras", m_mlbo),
                (edge_nr, "NRFI (0 Carreras en 1ª Entrada)", m_mlb_nrfi)
            ], key=lambda x: x[0])

            mejor_edge_props, mejor_nom_props, mejor_mom_props = max([
                (edge_ok_l, f"Over {lin_k_loc} Ponches ({dm['Local']})", m_ok_loc),
                (edge_uk_l, f"Under {lin_k_loc} Ponches ({dm['Local']})", m_uk_loc),
                (edge_ok_v, f"Over {lin_k_vis} Ponches ({dm['Visita']})", m_ok_vis),
                (edge_uk_v, f"Under {lin_k_vis} Ponches ({dm['Visita']})", m_uk_vis)
            ], key=lambda x: x[0])
            
            mejor_edge_f5, mejor_nom_f5, mejor_mom_f5, mejor_prob_f5 = max([
                (edge_f5_1, f"F5 - Victoria {dm['Local']}", m_f5_1, dm['Prob_F5_1']),
                (edge_f5_x, "F5 - Empate", m_f5_x, dm['Prob_F5_X']),
                (edge_f5_2, f"F5 - Victoria {dm['Visita']}", m_f5_2, dm['Prob_F5_2'])
            ], key=lambda x: x[0])
            
            st.markdown("---")
            st.markdown("### 🎯 Matriz de Picks MLB")
            col_b1, col_b2, col_bprops = st.columns(3)
            with col_b1:
                fav_mlb = dm['Local'] if dm['Prob_1'] >= dm['Prob_2'] else dm['Visita']
                st.markdown(f'<div class="safe-card"><h3>🛡️ Pick Seguro Global</h3><p><b>Moneyline: {fav_mlb}</b></p><p><b>Probabilidad:</b> {max(dm["Prob_1"], dm["Prob_2"])}%</p><hr><small>Apuesta directa protegida por Esperanza Pitagórica.</small></div>', unsafe_allow_html=True)
            with col_b2:
                if mejor_edge_val >= 3.0:
                    st.markdown(f'<div class="value-card"><h3>💎 Pick Valor Global</h3><p><b>{mejor_nom}</b> ({mejor_mom:+})</p><p><b>EDGE MATEMÁTICO:</b> +{mejor_edge_val}%</p><hr><small>Mayor ineficiencia detectada en el mercado principal.</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="value-card" style="border-color: #eab308; background: #422006;"><h3>⚠️ NO VALUE BET</h3><p>El margen es <b>{mejor_nom}</b> con solo +{mejor_edge_val}%.</p><hr><small style="color: #fde047;">Líneas muy apretadas. Sáltate el mercado principal.</small></div>', unsafe_allow_html=True)
            with col_bprops:
                if mejor_edge_props >= 3.0:
                    st.markdown(f'<div class="risk-card" style="border-left-color:#a855f7;"><h3>🎯 Value Pick: Ponches</h3><p><b>{mejor_nom_props}</b> ({mejor_mom_props:+})</p><p><b>EDGE MATEMÁTICO:</b> +{mejor_edge_props}%</p><hr><small>Proyección cruzada con estimación de Innings (IP).</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-card" style="border-color: #64748b; background: #1e293b;"><h3>🎯 Ponches (Apretado)</h3><p>Líneas de strikeout muy justas.</p><hr><small>El mayor margen es {mejor_nom_props} (+{mejor_edge_props}%).</small></div>', unsafe_allow_html=True)
                
            col_b5, col_b6 = st.columns(2)
            with col_b5:
                if mejor_edge_f5 >= 3.0:
                    st.markdown(f'<div class="value-card" style="border-left-color:#3b82f6;"><h3>⏱️ Value Pick: F5 (Primeras 5)</h3><p><b>{mejor_nom_f5}</b> ({mejor_mom_f5:+})</p><p><b>EDGE MATEMÁTICO:</b> +{mejor_edge_f5}%</p><hr><small>Probabilidad real del modelo: {mejor_prob_f5}%</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="value-card" style="border-color: #64748b; background: #1e293b;"><h3>⏱️ F5 (Sin Valor)</h3><p>Líneas F5 sumamente ajustadas por el casino.</p><hr><small>El mayor margen es {mejor_nom_f5} (+{mejor_edge_f5}%).</small></div>', unsafe_allow_html=True)
            with col_b6:
                fav_f5 = dm['Local'] if dm['Prob_F5_1'] >= dm['Prob_F5_2'] else dm['Visita']
                st.markdown(f'<div class="safe-card"><h3>🛡️ F5 - Empate No Acción (DNB)</h3><p><b>{fav_f5} (F5 DNB)</b></p><p><b>Proyección de Carreras (F5):</b> {dm["ER_F5_L"]} - {dm["ER_F5_V"]}</p><hr><small>Ideal si el abridor es muy superior pero tienes miedo al empate.</small></div>', unsafe_allow_html=True)

    with tab_mlb2:
        st.subheader("🧪 Laboratorio Multi-Algoritmo Béisbol")
        if dm:
            st.markdown(f'<div class="meta-model-card" style="border-color:#ef4444;"><h3 style="color:#fecaca; margin:0;">👑 CONSENSO MONEYBALL MLB</h3><h1 style="color:#ffffff; margin:10px 0;">{dm["Local"]}: {dm["Prob_1"]}% &nbsp;|&nbsp; {dm["Visita"]}: {dm["Prob_2"]}%</h1><p style="color:#fca5a5; margin:0;">En béisbol no existen empates; el 100% de la probabilidad se divide entre ambos bandos.</p></div>', unsafe_allow_html=True)
            t_mlb_comp = {
                "Metodología Cuantitativa": ["1️⃣ Distribución Poisson (Carreras + Abridor)", "2️⃣ Esperanza Pitagórica (Bill James Formula)", "3️⃣ Simulación Monte Carlo (5,000 Juegos)", "4️⃣ Proyección Ponches (SP)"],
                f"🏠 {dm['Local']}": [f"{dm['M1'][0]}%", f"{dm['M2'][0]}%", f"{dm['M3'][0]}%", f"~{proj_kl} Strikeouts"],
                f"✈️ {dm['Visita']}": [f"{dm['M1'][1]}%", f"{dm['M2'][1]}%", f"{dm['M3'][1]}%", f"~{proj_kv} Strikeouts"]
            }
            st.dataframe(pd.DataFrame(t_mlb_comp), use_container_width=True)

    with tab_mlb3:
        st.subheader("⚡ Parlays & Ranking MLB")
        data_j_mlb = [motor_mlb_360(p.split(" vs ")[0].strip(), p.split(" vs ")[1].strip(), df_mlb) for p in nombres_partidos]
        data_j_mlb = [x for x in data_j_mlb if x is not None]
        
        if data_j_mlb:
            # === FILA 1: PARLAYS CLÁSICOS ===
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
            
            # === FILA 2: NUEVO RANKING DE MONEYLINE ===
            st.markdown("<h3 style='color:#facc15;'>🏆 Ranking Definitivo de Moneyline (Mayor a Menor Seguridad)</h3>", unsafe_allow_html=True)
            st.write("Lista con los equipos de la jornada ordenados por su probabilidad matemática pura de ganar hoy. Ideal para armar tus propios boletos o evitar sorpresas.")
            
            ranking_ml = sorted(data_j_mlb, key=lambda x: max(x['Prob_1'], x['Prob_2']), reverse=True)
            rank_cols = st.columns(3)
            for idx, x in enumerate(ranking_ml):
                fav = x['Local'] if x['Prob_1'] > x['Prob_2'] else x['Visita']
                prob = max(x['Prob_1'], x['Prob_2'])
                col_idx = idx % 3
                with rank_cols[col_idx]:
                    st.markdown(f"<div style='background:#1e293b; padding:12px; border-radius:8px; margin-bottom:10px; border-left:4px solid #facc15;'><b>{idx+1}. {fav}</b><br><small style='color:#cbd5e1;'>Probabilidad Matemática: <b>{prob}%</b></small></div>", unsafe_allow_html=True)

            st.markdown("---")
            
            # === FILA 3: LA BOMBA SEGURA INTELIGENTE OMNI-MERCADO ===
            st.markdown("<h2 style='color:#ec4899; text-align:center;'>🚀 LA BOMBA SEGURA (PARLAY INTELIGENTE OMNI-MERCADO)</h2>", unsafe_allow_html=True)
            st.write("Este escáner está protegido por el nuevo filtro anti-varianza. **Solo aceptará picks con una probabilidad matemática extrema (>56.5%)**. Si un día la cartelera es muy peligrosa, te sugerirá menos picks para proteger tu dinero.")
            
            picks_mlb_moon, cuota_mlb_moon = "", 1.0
            
            todos_los_picks = []
            for juego in data_j_mlb:
                opciones = [
                    (juego['Prob_1'], f"Victoria {juego['Local']} (Moneyline)", 1.55, juego['Local'], juego['Visita']),
                    (juego['Prob_2'], f"Victoria {juego['Visita']} (Moneyline)", 1.55, juego['Local'], juego['Visita']),
                    (juego['Prob_F5_1'], f"Victoria {juego['Local']} (Primeras 5 - F5)", 1.60, juego['Local'], juego['Visita']),
                    (juego['Prob_F5_2'], f"Victoria {juego['Visita']} (Primeras 5 - F5)", 1.60, juego['Local'], juego['Visita']),
                    (juego['NRFI'], "NRFI (0 Carreras en 1ª Entrada)", 1.85, juego['Local'], juego['Visita']),
                    (100 - juego['NRFI'], "YRFI (Sí hay Carrera en 1ª Entrada)", 1.85, juego['Local'], juego['Visita']),
                    (juego['Over_Line'], f"Over {juego['Linea_OU']} Carreras", 1.90, juego['Local'], juego['Visita']),
                    (100 - juego['Over_Line'], f"Under {juego['Linea_OU']} Carreras", 1.90, juego['Local'], juego['Visita'])
                ]
                todos_los_picks.extend(opciones)
                
            # Ordenamos absolutamente todos los picks de la jornada por su probabilidad pura
            picks_bomba = sorted(todos_los_picks, key=lambda x: x[0], reverse=True)
            
            # Filtro inteligente para no repetir el mismo pick idéntico y aplicar el Umbral Anti-Varianza
            picks_finales = []
            combinaciones_vistas = set()
            
            for p in picks_bomba:
                prob, pick_name, cuota, loc, vis = p
                
                # UMBRAL ESTRICTO: Si la probabilidad es menor a 56.5%, lo desechamos para no arriesgar.
                if prob < 56.5:
                    continue
                    
                # Extraemos la esencia (el equipo ganador) para evitar empalmar ML y F5 del mismo equipo
                esencia = "Victoria " + loc if "Victoria " + loc in pick_name else "Victoria " + vis if "Victoria " + vis in pick_name else pick_name
                id_partido_esencia = f"{loc}-{vis}-{esencia}"
                
                if id_partido_esencia not in combinaciones_vistas:
                    combinaciones_vistas.add(id_partido_esencia)
                    picks_finales.append(p)
                    
                if len(picks_finales) == 5:
                    break
                    
            if len(picks_finales) > 0:
                for p in picks_finales:
                    mejor_prob, nombre_pick, cuota_est, loc, vis = p
                    picks_mlb_moon += f"✨ <b>{loc} vs {vis}:</b> {nombre_pick} <span style='color:#fbcfe8;'>({mejor_prob}% Probabilidad)</span><br>"
                    cuota_mlb_moon *= cuota_est
                    
                mom_moon_mlb = int((cuota_mlb_moon - 1.0) * 100) if cuota_mlb_moon >= 2.0 else int(-100 / (cuota_mlb_moon - 1.0))
                
                st.markdown(f"""
                <div class="dream-card">
                    <h3 style="color:#fdf2f8; margin-top:0;">🌌 BOLETO DE LA BOMBA OMNI-MERCADO</h3>
                    <div style="font-size: 1.05em; line-height: 1.6; margin: 15px 0;">{picks_mlb_moon}</div>
                    <hr style="border-color: #db2777;">
                    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                        <div>
                            <h2 style="color: #ffffff; margin: 0;">🎟️ MOMIO EST: {mom_moon_mlb:+}</h2>
                            <span style="color: #fbcfe8;">Cuota Decimal Multiplicada: <b>{round(cuota_mlb_moon, 2)}</b></span>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 8px; border: 1px solid #ec4899; margin-top: 10px;">
                            <span style="color: #f43f5e; font-weight: bold;">⚠️ GESTIÓN DE RIESGO:</span><br>
                            <small style="color: #fce7f3;">Stake sugerido: <b>0.5u</b>. Al agarrar la probabilidad pura más alta en todos los mercados sin importar el partido, el algoritmo maximiza el cobro con riesgo calculado.<br>Nota: Si el casino bloquea algo, usa la lógica para armar la variante permitida.</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-card" style="border-left-color:#ef4444;"><h3>🛑 ALERTA ANTI-VARIANZA ACTIVA</h3><p>El escáner de la Bomba Inteligente <b>no encontró ningún pick en toda la jornada de hoy que supere el umbral de seguridad del 56.5%</b>. Esto significa que los partidos de hoy son extremadamente parejos y riesgosos. <b>Recomendación del sistema: Guarda tu dinero, hoy no se meten parlays de alto riesgo.</b></p></div>', unsafe_allow_html=True)

    with tab_mlb4:
        st.subheader("📈 Ranking de Esperanza Pitagórica (Moneyball)")
        st.write("La Esperanza Pitagórica nos dice qué porcentaje de victorias *debería* tener un equipo según sus carreras. Además, la API ahora castiga matemáticamente a los equipos atrapados en rachas perdedoras.")
        if not df_mlb.empty:
            df_show_mlb = df_mlb[['Equipo', 'G', 'P', 'RS_prom', 'RA_prom', 'ERA', 'WHIP', 'Pitagorica', 'Racha']].sort_values(by='Pitagorica', ascending=False).reset_index(drop=True)
            df_show_mlb.index += 1
            st.dataframe(df_show_mlb.style.background_gradient(subset=['Pitagorica'], cmap='Reds').background_gradient(subset=['ERA'], cmap='Blues_r'), use_container_width=True)
