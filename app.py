import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import math

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="AI ORACLE - Apuestas Deportivas & MLB",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Cyberpunk / Terminal Profesional
st.markdown("""
<style>
    .global-header { background: linear-gradient(135deg, #0f172a 0%, #000000 100%); padding: 25px; border-radius: 15px; text-align: center; border: 2px solid #38bdf8; box-shadow: 0 0 20px rgba(56, 189, 248, 0.4); margin-bottom: 25px; }
    .global-header-mlb { background: linear-gradient(135deg, #450a0a 0%, #000000 100%); padding: 25px; border-radius: 15px; text-align: center; border: 2px solid #ef4444; box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); margin-bottom: 25px; }
    .bookie-section { background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #475569; margin-bottom: 15px; }
    .pick-top { background: linear-gradient(90deg, #064e3b 0%, #022c22 100%); padding: 20px; border-radius: 12px; border-left: 6px solid #10b981; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2); }
    .parlay-slip { background: #312e81; padding: 20px; border-radius: 12px; border: 2px dashed #f472b6; margin-top: 20px; }
    .expert-alert { background: #1e1b4b; padding: 15px; border-radius: 8px; border-left: 5px solid #818cf8; margin-top: 10px; }
    h1, h2, h3, h4 { font-family: 'Courier New', Courier, monospace; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

# INICIALIZAR CARRITO DE PARLAY MULTIDEPORTE
if 'parlay_cart' not in st.session_state:
    st.session_state.parlay_cart = []

# FUNCION CORREGIDA PARA EL CALLBACK
def add_to_parlay(match, pick, odds):
    st.session_state.parlay_cart.append({"Match": match, "Pick": pick, "Odds": odds})

def clear_parlay():
    st.session_state.parlay_cart = []

# FUNCIONES MATEMÁTICAS CORE
def americano_a_decimal(momio_amer):
    if momio_amer == 0: return 1.01
    if momio_amer > 0: return round((momio_amer / 100.0) + 1.0, 2)
    else: return round((100.0 / abs(momio_amer)) + 1.0, 2)

def decimal_a_americano(dec):
    if dec <= 1.0: return 0
    if dec >= 2.0: return int((dec - 1.0) * 100)
    else: return int(-100 / (dec - 1.0))

def implied_prob(dec):
    return (1.0 / dec) * 100 if dec > 0 else 0

# MOTOR 1: INGENIERÍA INVERSA FÚTBOL
def reverse_engineer_xg(dec_1, dec_x, dec_2, sharp_adjustment=0.0):
    prob_1 = implied_prob(dec_1) / 100.0
    prob_x = implied_prob(dec_x) / 100.0
    prob_2 = implied_prob(dec_2) / 100.0
    margin = prob_1 + prob_x + prob_2
    
    true_1 = (prob_1 / margin) + sharp_adjustment
    true_x = prob_x / margin
    true_2 = (prob_2 / margin) - sharp_adjustment
    
    diff = true_1 - true_2
    xg_base = 1.35
    xg_a = max(0.4, min(xg_base + (diff * 2.5) + (0.2 if true_1 > true_2 else -0.1), 4.0))
    xg_b = max(0.4, min(xg_base - (diff * 2.5) + (0.2 if true_2 > true_1 else -0.1), 4.0))
    
    return xg_a, xg_b, true_1, true_x, true_2

def generate_soccer_market(xg_a, xg_b):
    max_goals = 7
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i,j] = stats.poisson.pmf(i, xg_a) * stats.poisson.pmf(j, xg_b)
            
    btts_yes = np.sum(matrix[1:, 1:]) * 100
    btts_no = 100 - btts_yes
    
    over_15, over_25, under_35 = 0, 0, 0
    marcadores = []
    
    for i in range(max_goals):
        for j in range(max_goals):
            prob = matrix[i,j] * 100
            marcadores.append({"Score": f"{i}-{j}", "Prob": prob})
            if (i+j) > 1.5: over_15 += prob
            if (i+j) > 2.5: over_25 += prob
            if (i+j) < 3.5: under_35 += prob

    marcadores = sorted(marcadores, key=lambda x: x['Prob'], reverse=True)[:5]
    exp_corners = 8.5 + (xg_a * 1.2) + (xg_b * 0.8)
    
    return {
        "BTTS_Y": btts_yes, "BTTS_N": btts_no,
        "O25": over_25, "U25": 100 - over_25,
        "O15": over_15, "U35": under_35,
        "Corners": round(exp_corners, 1),
        "Scores": marcadores
    }

# MOTOR 2: INGENIERÍA INVERSA BÉISBOL (MLB)
def reverse_engineer_mlb(dec_1, dec_2, line_ou, dec_over, dec_under, sharp_adjustment=0.0):
    prob_1 = implied_prob(dec_1) / 100.0
    prob_2 = implied_prob(dec_2) / 100.0
    margin_ml = prob_1 + prob_2
    
    true_1 = (prob_1 / margin_ml) + sharp_adjustment
    true_2 = (prob_2 / margin_ml) - sharp_adjustment
    
    prob_o = implied_prob(dec_over) / 100.0
    prob_u = implied_prob(dec_under) / 100.0
    margin_ou = prob_o + prob_u
    true_over = prob_o / margin_ou
    
    # Estimación de carreras totales esperadas
    ajuste_carreras = (true_over - 0.50) * 1.2
    carreras_totales = line_ou + ajuste_carreras
    
    # Distribución de carreras esperadas por equipo
    ratio_1 = math.pow(true_1 / max(true_2, 0.01), 0.55)
    lambda_l = (carreras_totales * ratio_1) / (1.0 + ratio_1)
    lambda_v = carreras_totales - lambda_l
    
    return lambda_l, lambda_v, true_1, true_2, carreras_totales

def generate_mlb_market(lambda_l, lambda_v, line_ou):
    max_runs = 14
    matrix = np.zeros((max_runs, max_runs))
    for i in range(max_runs):
        for j in range(max_runs):
            matrix[i,j] = stats.poisson.pmf(i, lambda_l) * stats.poisson.pmf(j, lambda_v)
            
    # Totales y Run Line
    over_line, under_line = 0, 0
    rl_local, rl_visita = 0, 0
    
    for i in range(max_runs):
        for j in range(max_runs):
            prob = matrix[i,j] * 100
            if (i + j) > line_ou: over_line += prob
            elif (i + j) < line_ou: under_line += prob
            if (i - j) >= 2: rl_local += prob
            if (j - i) >= 2: rl_visita += prob
            
    # Primeras 5 Entradas (F5)
    lam_f5_l = lambda_l * (5.0 / 9.0) * 1.02
    lam_f5_v = lambda_v * (5.0 / 9.0) * 0.98
    p_f5_1, p_f5_x, p_f5_2 = 0, 0, 0
    for i in range(8):
        for j in range(8):
            p = stats.poisson.pmf(i, lam_f5_l) * stats.poisson.pmf(j, lam_f5_v) * 100
            if i > j: p_f5_1 += p
            elif i == j: p_f5_x += p
            else: p_f5_2 += p
            
    # 1st Inning (NRFI / YRFI)
    lam_1st_l = (lambda_l / 9.0) * 1.15
    lam_1st_v = (lambda_v / 9.0) * 1.15
    nrfi_prob = (stats.poisson.pmf(0, lam_1st_l) * stats.poisson.pmf(0, lam_1st_v)) * 100
    yrfi_prob = 100.0 - nrfi_prob
    
    return {
        "Over_Line": over_line, "Under_Line": under_line,
        "RL_Local": rl_local, "RL_Visita": rl_visita,
        "F5_1": p_f5_1, "F5_X": p_f5_x, "F5_2": p_f5_2,
        "NRFI": nrfi_prob, "YRFI": yrfi_prob,
        "Lam_L": round(lambda_l, 2), "Lam_V": round(lambda_v, 2)
    }

# ==============================================================================
# UI PRINCIPAL - SELECTOR DE MÓDULOS
# ==============================================================================
st.sidebar.title("👁️ ORACLE AI TERMINAL")
st.sidebar.markdown("---")
modo = st.sidebar.radio("Selecciona Motor Cuantitativo:", [
    "⚽ Fútbol Universal (Cualquier Liga)",
    "⚾ Béisbol Universal (Cualquier Juego MLB)",
    "⚡ Constructor de Parlays Multideporte",
    "🇲🇽 Guía Pro / Análisis Forense"
])

# ------------------------------------------------------------------------------
# MÓDULO 1: FÚTBOL UNIVERSAL
# ------------------------------------------------------------------------------
if modo == "⚽ Fútbol Universal (Cualquier Liga)":
    st.markdown("""
    <div class="global-header">
        <h1 style="color:#ffffff; margin:0;">🤖 ORACLE SOCCER PREDICTOR</h1>
        <p style="color:#94a3b8; font-size:16px;">Ingresa cualquier duelo de fútbol en el mundo. La IA extraerá los xG y el mercado completo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3 = st.columns([2, 1, 2])
    with col_s1: eq_l = st.text_input("🏠 Equipo Local", value="Real Madrid", key="f_loc")
    with col_s2: st.markdown("<h2 style='text-align:center; color:#64748b; margin-top:25px;'>VS</h2>", unsafe_allow_html=True)
    with col_s3: eq_v = st.text_input("✈️ Equipo Visita", value="Barcelona", key="f_vis")
    
    if eq_l and eq_v:
        st.markdown("### 🏦 Cuotas del Casino (Momios Americanos)")
        c1, cx, c2 = st.columns(3)
        with c1: l_1 = st.number_input(f"Victoria {eq_l} (1)", value=-110, step=10, key="fl1")
        with cx: l_x = st.number_input("Empate (X)", value=+240, step=10, key="flx")
        with c2: l_2 = st.number_input(f"Victoria {eq_v} (2)", value=+220, step=10, key="fl2")
        
        sharp_act = st.selectbox("🧠 Radar Vegas (Dinero Inteligente):", 
                                 ["Sin movimiento claro (Neutral)", f"Dinero entrando fuerte a {eq_l}", f"Dinero entrando fuerte a {eq_v}"], key="f_sharp")
        
        if st.button("🔮 DECODIFICAR MERCADO COMPLETO (FÚTBOL)", use_container_width=True, type="primary"):
            d1, dx, d2 = americano_a_decimal(l_1), americano_a_decimal(l_x), americano_a_decimal(l_2)
            adj = 0.04 if sharp_act == f"Dinero entrando fuerte a {eq_l}" else -0.04 if sharp_act == f"Dinero entrando fuerte a {eq_v}" else 0.0
            
            xgl, xgv, t1, tx, t2 = reverse_engineer_xg(d1, dx, d2, adj)
            mercado = generate_soccer_market(xgl, xgv)
            
            picks = [
                (t1*100, f"Victoria {eq_l}", decimal_a_americano(1/t1 if t1>0 else 1.01)),
                (t2*100, f"Victoria {eq_v}", decimal_a_americano(1/t2 if t2>0 else 1.01)),
                (mercado['O25'], "Over 2.5 Goles", decimal_a_americano(1/(mercado['O25']/100))),
                (mercado['U25'], "Under 2.5 Goles", decimal_a_americano(1/(mercado['U25']/100))),
                (mercado['BTTS_Y'], "Ambos Anotan - SÍ", decimal_a_americano(1/(mercado['BTTS_Y']/100))),
                (mercado['BTTS_N'], "Ambos Anotan - NO", decimal_a_americano(1/(mercado['BTTS_N']/100)))
            ]
            top = max(picks, key=lambda x: x[0])
            
            st.markdown("---")
            st.markdown(f"""
            <div class="pick-top">
                <h2 style="color:#ffffff; margin:0;">⭐ TOP VALUE PICK</h2>
                <h3 style="color:#34d399; margin:5px 0;">{top[1]}</h3>
                <p style="color:#a7f3d0; margin:0;"><b>Probabilidad Matemática:</b> {round(top[0], 1)}% | <b>Momio Justo:</b> {top[2]:+}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # BOTON CORREGIDO USANDO CALLBACK (on_click)
            st.button(f"🛒 Agregar '{top[1]}' al Ticket de Parlay", 
                      on_click=add_to_parlay, 
                      args=(f"{eq_l} vs {eq_v}", top[1], top[2]), 
                      key="btn_add_fut",
                      type="secondary")
                
            st.markdown("---")
            c_g, c_b, c_s = st.columns(3)
            with c_g:
                st.markdown('<div class="bookie-section"><h4>🥅 Línea de Goles</h4>', unsafe_allow_html=True)
                st.write(f"**Over 1.5:** {round(mercado['O15'],1)}%")
                st.write(f"**Over 2.5:** {round(mercado['O25'],1)}%")
                st.write(f"**Under 2.5:** {round(mercado['U25'],1)}%")
                st.write(f"**Under 3.5:** {round(mercado['U35'],1)}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with c_b:
                st.markdown('<div class="bookie-section"><h4>🤝 Ambos Anotan (BTTS)</h4>', unsafe_allow_html=True)
                st.write(f"**Ambos Anotan - SÍ:** {round(mercado['BTTS_Y'],1)}%")
                st.write(f"**Ambos Anotan - NO:** {round(mercado['BTTS_N'],1)}%")
                st.markdown('<hr><h4>🚩 Tiros de Esquina</h4>', unsafe_allow_html=True)
                st.write(f"**Proyección Base:** {mercado['Corners']} Córners")
                st.markdown('</div>', unsafe_allow_html=True)
            with c_s:
                st.markdown('<div class="bookie-section"><h4>🎯 Marcadores Más Probables</h4>', unsafe_allow_html=True)
                for sc in mercado['Scores']:
                    st.write(f"**{sc['Score']}** ({round(sc['Prob'],1)}%)")
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MÓDULO 2: BÉISBOL UNIVERSAL (MLB)
# ------------------------------------------------------------------------------
elif modo == "⚾ Béisbol Universal (Cualquier Juego MLB)":
    st.markdown("""
    <div class="global-header-mlb">
        <h1 style="color:#ffffff; margin:0;">⚾ ORACLE MLB DIAMOND PREDICTOR</h1>
        <p style="color:#fca5a5; font-size:16px;">Analiza cualquier duelo de MLB. Desglosa Moneyline, F5, Totales, Run Line y NRFI/YRFI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns([2, 1, 2])
    with col_m1: mlb_l = st.text_input("🏠 Equipo Local", value="Los Angeles Dodgers", key="mlb_loc")
    with col_m2: st.markdown("<h2 style='text-align:center; color:#ef4444; margin-top:25px;'>VS</h2>", unsafe_allow_html=True)
    with col_m3: mlb_v = st.text_input("✈️ Equipo Visita", value="New York Yankees", key="mlb_vis")
    
    if mlb_l and mlb_v:
        st.markdown("### 🏦 Cuotas del Mercado Principal (Formato Americano)")
        cm_1, cm_2, cm_line, cm_o, cm_u = st.columns(5)
        with cm_1: m_ml1 = st.number_input(f"ML {mlb_l}", value=-140, step=5, key="m_ml1")
        with cm_2: m_ml2 = st.number_input(f"ML {mlb_v}", value=+120, step=5, key="m_ml2")
        with cm_line: m_tot_line = st.number_input("Línea O/U", value=8.5, step=0.5, key="m_tot_l")
        with cm_o: m_over = st.number_input(f"Over {m_tot_line}", value=-110, step=5, key="m_ov")
        with cm_u: m_under = st.number_input(f"Under {m_tot_line}", value=-110, step=5, key="m_un")
        
        sharp_mlb = st.selectbox("🧠 Radar Vegas MLB (Dinero Inteligente / Viento):", [
            "Sin movimiento / Neutral",
            f"Dinero entrando fuerte a {mlb_l}",
            f"Dinero entrando fuerte a {mlb_v}",
            "Viento fuerte hacia afuera (Favorece Over)",
            "Viento fuerte hacia adentro (Favorece Under)"
        ], key="sharp_mlb")
        
        if st.button("🔮 DECODIFICAR JUEGO MLB", use_container_width=True, type="primary"):
            d_ml1, d_ml2 = americano_a_decimal(m_ml1), americano_a_decimal(m_ml2)
            d_ov, d_un = americano_a_decimal(m_over), americano_a_decimal(m_under)
            
            adj_mlb = 0.04 if sharp_mlb == f"Dinero entrando fuerte a {mlb_l}" else -0.04 if sharp_mlb == f"Dinero entrando fuerte a {mlb_v}" else 0.0
            
            lam_l, lam_v, t1_m, t2_m, exp_tot = reverse_engineer_mlb(d_ml1, d_ml2, m_tot_line, d_ov, d_un, adj_mlb)
            
            # Ajuste de viento si aplica
            if sharp_mlb == "Viento fuerte hacia afuera (Favorece Over)":
                lam_l *= 1.12
                lam_v *= 1.12
            elif sharp_mlb == "Viento fuerte hacia adentro (Favorece Under)":
                lam_l *= 0.88
                lam_v *= 0.88
                
            m_mlb = generate_mlb_market(lam_l, lam_v, m_tot_line)
            
            # DETERMINAR TOP PICK MLB
            picks_mlb = [
                (t1_m*100, f"Victoria {mlb_l} (ML)", decimal_a_americano(1/t1_m if t1_m>0 else 1.01)),
                (t2_m*100, f"Victoria {mlb_v} (ML)", decimal_a_americano(1/t2_m if t2_m>0 else 1.01)),
                (m_mlb['NRFI'], "NRFI (Sin Carrera en 1ª Entrada)", decimal_a_americano(1/(m_mlb['NRFI']/100))),
                (m_mlb['YRFI'], "YRFI (Sí hay Carrera en 1ª Entrada)", decimal_a_americano(1/(m_mlb['YRFI']/100))),
                (m_mlb['Over_Line'], f"Over {m_tot_line} Carreras", decimal_a_americano(1/(m_mlb['Over_Line']/100))),
                (m_mlb['Under_Line'], f"Under {m_tot_line} Carreras", decimal_a_americano(1/(m_mlb['Under_Line']/100))),
                (m_mlb['F5_1'], f"{mlb_l} Gana F5 (Primeras 5)", decimal_a_americano(1/(m_mlb['F5_1']/100))),
                (m_mlb['F5_2'], f"{mlb_v} Gana F5 (Primeras 5)", decimal_a_americano(1/(m_mlb['F5_2']/100)))
            ]
            top_m = max(picks_mlb, key=lambda x: x[0])
            
            st.markdown("---")
            st.markdown(f"""
            <div class="pick-top" style="border-left-color: #ef4444;">
                <h2 style="color:#ffffff; margin:0;">⭐ TOP VALUE PICK MLB</h2>
                <h3 style="color:#f87171; margin:5px 0;">{top_m[1]}</h3>
                <p style="color:#fca5a5; margin:0;"><b>Probabilidad Real del Algoritmo:</b> {round(top_m[0], 1)}% | <b>Momio Justo:</b> {top_m[2]:+}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # BOTON CORREGIDO USANDO CALLBACK (on_click)
            st.button(f"🛒 Agregar '{top_m[1]}' al Ticket de Parlay", 
                      on_click=add_to_parlay, 
                      args=(f"{mlb_l} vs {mlb_v}", top_m[1], top_m[2]), 
                      key="btn_add_mlb",
                      type="secondary")
                
            st.markdown("---")
            st.markdown("### 📊 RADIOGRAFÍA COMPLETA DEL DIAMANTE")
            
            c_mlb1, c_mlb2, c_mlb3 = st.columns(3)
            with c_mlb1:
                st.markdown('<div class="bookie-section"><h4>⏱️ Primeras 5 Entradas (F5)</h4>', unsafe_allow_html=True)
                st.write(f"**Victoria {mlb_l}:** {round(m_mlb['F5_1'],1)}%")
                st.write(f"**Empate F5 (Push):** {round(m_mlb['F5_X'],1)}%")
                st.write(f"**Victoria {mlb_v}:** {round(m_mlb['F5_2'],1)}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with c_mlb2:
                st.markdown('<div class="bookie-section"><h4>🔥 1st Inning Props</h4>', unsafe_allow_html=True)
                st.write(f"**NRFI (0 Carreras):** {round(m_mlb['NRFI'],1)}%")
                st.write(f"**YRFI (1+ Carreras):** {round(m_mlb['YRFI'],1)}%")
                st.markdown('<hr><h4>🛡️ Run Line (-1.5)</h4>', unsafe_allow_html=True)
                st.write(f"**{mlb_l} -1.5:** {round(m_mlb['RL_Local'],1)}%")
                st.write(f"**{mlb_v} -1.5:** {round(m_mlb['RL_Visita'],1)}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with c_mlb3:
                st.markdown('<div class="bookie-section"><h4>⚾ Proyección de Carreras</h4>', unsafe_allow_html=True)
                st.write(f"**Carreras {mlb_l}:** {m_mlb['Lam_L']}")
                st.write(f"**Carreras {mlb_v}:** {m_mlb['Lam_V']}")
                st.write(f"**Total Esperado:** {round(m_mlb['Lam_L'] + m_mlb['Lam_V'], 2)}")
                st.write(f"**Probabilidad Over {m_tot_line}:** {round(m_mlb['Over_Line'],1)}%")
                st.write(f"**Probabilidad Under {m_tot_line}:** {round(m_mlb['Under_Line'],1)}%")
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MÓDULO 3: CONSTRUCTOR DE PARLAYS MULTIDEPORTE
# ------------------------------------------------------------------------------
elif modo == "⚡ Constructor de Parlays Multideporte":
    st.markdown("## 🎫 Tu Ticket Inteligente Multideporte")
    st.write("Combina apuestas de fútbol y MLB en un mismo boleto con la matemática multiplicada.")
    
    if len(st.session_state.parlay_cart) == 0:
        st.info("💡 El ticket está vacío. Analiza duelos en Fútbol o MLB y presiona 'Agregar al Ticket'.")
    else:
        cuota_decimal_total = 1.0
        st.markdown('<div class="parlay-slip">', unsafe_allow_html=True)
        for i, item in enumerate(st.session_state.parlay_cart):
            st.markdown(f"**{i+1}. {item['Match']}**")
            st.markdown(f"↳ 🎯 Pick: <span style='color:#34d399; font-weight:bold;'>{item['Pick']}</span> | Momio: {item['Odds']:+}", unsafe_allow_html=True)
            st.markdown("---")
            dec = americano_a_decimal(item['Odds'])
            cuota_decimal_total *= dec
            
        momio_final = decimal_a_americano(cuota_decimal_total)
        st.markdown(f"<h2 style='text-align:right; color:#f472b6; margin:0;'>🎟️ MOMIO COMBINADO EST: {momio_final:+}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:right; color:#cbd5e1; margin:0;'>Multiplicador Decimal: <b>{round(cuota_decimal_total, 2)}x</b></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Limpiar Ticket Completo", type="secondary"):
            clear_parlay()
            st.rerun()

# ------------------------------------------------------------------------------
# MÓDULO 4: GUÍA PRO Y ANÁLISIS FORENSE
# ------------------------------------------------------------------------------
elif modo == "🇲🇽 Guía Pro / Análisis Forense":
    st.markdown("## 🧠 El Manual del Apostador Cuantitativo")
    st.markdown("""
    ### 1. ¿Por qué la Liga MX y la MLB fallan en apuestas directas?
    * **Liga MX:** El formato de liguilla hace que los equipos favoritos dosifiquen y jueguen a empatar de visita. La solución matemática no es apostar al ganador, sino a **Córners Totales** o **Ambos Anotan**.
    * **MLB:** El béisbol tiene la varianza más alta del deporte mundial. Apostar a que un equipo gana el juego completo a 9 innings expone tu dinero al bullpen en la 8ª y 9ª entrada. La solución es jugar **F5 (Primeras 5 entradas)** o **NRFI**.

    ### 2. Cómo usar este Terminal como un Profesional
    1. Abre tu casa de apuestas (Playdoit, Caliente, etc.).
    2. Ingresa los momios del partido en el Escáner.
    3. Si detectas que una línea se movió raro, ajusta el selector de *Dinero Inteligente*.
    4. Toma el **Top Pick** y agrégalo a tu ticket.
    """)
    
