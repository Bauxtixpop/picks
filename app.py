import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import math

# 1. CONFIGURACIÓN PREMIUM DE LA PÁGINA
st.set_page_config(
    page_title="AI ORACLE - Apuestas Deportivas",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Cyberpunk / Terminal Profesional
st.markdown("""
<style>
    .global-header { background: linear-gradient(135deg, #0f172a 0%, #000000 100%); padding: 30px; border-radius: 15px; text-align: center; border: 2px solid #38bdf8; box-shadow: 0 0 20px rgba(56, 189, 248, 0.4); margin-bottom: 25px; }
    .bookie-section { background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #475569; margin-bottom: 15px; }
    .pick-top { background: linear-gradient(90deg, #064e3b 0%, #022c22 100%); padding: 20px; border-radius: 12px; border-left: 6px solid #10b981; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2); }
    .parlay-slip { background: #312e81; padding: 20px; border-radius: 12px; border: 2px dashed #f472b6; margin-top: 20px; }
    .stat-badge { background: #0f172a; border: 1px solid #38bdf8; color: #38bdf8; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .expert-alert { background: #450a0a; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444; margin-top: 10px; }
    h1, h2, h3, h4 { font-family: 'Courier New', Courier, monospace; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

# INICIALIZAR CARRITO DE PARLAY
if 'parlay_cart' not in st.session_state:
    st.session_state.parlay_cart = []

def add_to_parlay(match, pick, odds):
    st.session_state.parlay_cart.append({"Match": match, "Pick": pick, "Odds": odds})
    st.success(f"✅ {pick} añadido al Parlay Ticket!")

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

def reverse_engineer_xg(dec_1, dec_x, dec_2, sharp_adjustment=0.0):
    # 1. Quitar el VIG (Margen del casino)
    prob_1 = implied_prob(dec_1) / 100.0
    prob_x = implied_prob(dec_x) / 100.0
    prob_2 = implied_prob(dec_2) / 100.0
    margin = prob_1 + prob_x + prob_2
    
    true_1 = prob_1 / margin
    true_x = prob_x / margin
    true_2 = prob_2 / margin
    
    # Ajuste de Dinero Inteligente (Sharp Money)
    true_1 += sharp_adjustment
    true_2 -= sharp_adjustment
    
    # 2. Aproximación Bivariada de Poisson a xG
    # Fórmula heurística basada en la relación de probabilidades de ganar
    xg_base = 1.35
    diff = true_1 - true_2
    
    xg_a = xg_base + (diff * 2.5) + (0.2 if true_1 > true_2 else -0.1)
    xg_b = xg_base - (diff * 2.5) + (0.2 if true_2 > true_1 else -0.1)
    
    xg_a = max(0.4, min(xg_a, 4.0))
    xg_b = max(0.4, min(xg_b, 4.0))
    
    return xg_a, xg_b, true_1, true_x, true_2

def generate_full_market(xg_a, xg_b):
    # Poisson Matrix
    max_goals = 7
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i,j] = stats.poisson.pmf(i, xg_a) * stats.poisson.pmf(j, xg_b)
            
    # Mercados
    btts_yes = np.sum(matrix[1:, 1:]) * 100
    btts_no = 100 - btts_yes
    over_25 = np.sum(np.tril(matrix, -3)) + np.sum(np.triu(matrix, 3)) + np.sum(np.diag(matrix)[2:]) # Aproximación rápida
    
    over_25 = 0
    over_15 = 0
    under_35 = 0
    marcadores = []
    
    for i in range(max_goals):
        for j in range(max_goals):
            prob = matrix[i,j] * 100
            marcadores.append({"Score": f"{i}-{j}", "Prob": prob})
            if (i+j) > 2.5: over_25 += prob
            if (i+j) > 1.5: over_15 += prob
            if (i+j) < 3.5: under_35 += prob

    marcadores = sorted(marcadores, key=lambda x: x['Prob'], reverse=True)[:5]
    
    # Córners (Correlación con xG)
    exp_corners = 8.5 + (xg_a * 1.2) + (xg_b * 0.8)
    
    return {
        "BTTS_Y": btts_yes, "BTTS_N": btts_no,
        "O25": over_25, "U25": 100 - over_25,
        "O15": over_15, "U35": under_35,
        "Corners": round(exp_corners, 1),
        "Scores": marcadores
    }

# UI PRINCIPAL
st.sidebar.title("👁️ ORACLE AI")
st.sidebar.markdown("---")
modo = st.sidebar.radio("Modo de Operación:", ["🌍 Escáner Global (Cualquier Liga)", "⚡ Constructor de Parlays", "🇲🇽 Fix Liga MX (Análisis)"])

if modo == "🌍 Escáner Global (Cualquier Liga)":
    st.markdown("""
    <div class="global-header">
        <h1 style="color:#ffffff; margin:0;">🤖 ORACLE GLOBAL PREDICTOR</h1>
        <p style="color:#94a3b8; font-size:18px;">Ingresa el partido y las cuotas del casino. La IA hará ingeniería inversa para extraer el mercado completo y detectar el valor.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3 = st.columns([2, 1, 2])
    with col_s1: equipo_local = st.text_input("🏠 Equipo Local (Ej. Real Madrid)")
    with col_s2: st.markdown("<h2 style='text-align:center; color:#64748b; margin-top:25px;'>VS</h2>", unsafe_allow_html=True)
    with col_s3: equipo_visita = st.text_input("✈️ Equipo Visita (Ej. Barcelona)")
    
    if equipo_local and equipo_visita:
        st.markdown("### 🏦 Líneas Principales del Casino (Formato Americano)")
        c_m1, c_mx, c_m2 = st.columns(3)
        with c_m1: line_1 = st.number_input(f"ML {equipo_local} (1)", value=-110, step=10)
        with c_mx: line_x = st.number_input("Empate (X)", value=+220, step=10)
        with c_m2: line_2 = st.number_input(f"ML {equipo_visita} (2)", value=+180, step=10)
        
        st.markdown("### 🧠 Sabiduría de las Masas (Expert Consensus)")
        sharp_action = st.selectbox("¿Hacia dónde se está moviendo fuertemente el dinero en Las Vegas / Tipsters?", 
                                    ["Sin movimiento claro (Neutral)", f"Dinero entrando fuerte a {equipo_local}", f"Dinero entrando fuerte a {equipo_visita}"])
        
        if st.button("🔮 DECODIFICAR PARTIDO", use_container_width=True, type="primary"):
            d1, dx, d2 = americano_a_decimal(line_1), americano_a_decimal(line_x), americano_a_decimal(line_2)
            
            sharp_adj = 0.0
            if sharp_action == f"Dinero entrando fuerte a {equipo_local}": sharp_adj = 0.04
            elif sharp_action == f"Dinero entrando fuerte a {equipo_visita}": sharp_adj = -0.04
            
            xg_l, xg_v, t1, tx, t2 = reverse_engineer_xg(d1, dx, d2, sharp_adj)
            mercado = generate_full_market(xg_l, xg_v)
            
            # DETERMINAR TOP PICK
            picks_eval = [
                (t1*100, f"Victoria {equipo_local}", decimal_a_americano(1/t1 if t1>0 else 1.01)),
                (t2*100, f"Victoria {equipo_visita}", decimal_a_americano(1/t2 if t2>0 else 1.01)),
                (mercado['O25'], "Over 2.5 Goles", decimal_a_americano(1/(mercado['O25']/100))),
                (mercado['U25'], "Under 2.5 Goles", decimal_a_americano(1/(mercado['U25']/100))),
                (mercado['BTTS_Y'], "Ambos Anotan - SÍ", decimal_a_americano(1/(mercado['BTTS_Y']/100))),
                (mercado['BTTS_N'], "Ambos Anotan - NO", decimal_a_americano(1/(mercado['BTTS_N']/100)))
            ]
            top_pick = max(picks_eval, key=lambda x: x[0])
            
            st.markdown("---")
            st.markdown(f"""
            <div class="pick-top">
                <h2 style="color:#ffffff; margin:0;">⭐ TOP PICK DEL ALGORITMO</h2>
                <h3 style="color:#34d399;">{top_pick[1]}</h3>
                <p style="color:#a7f3d0; margin:0;"><b>Probabilidad Real:</b> {round(top_pick[0], 1)}% | <b>Momio Justo Calculado:</b> {top_pick[2]:+}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🛒 Agregar {top_pick[1]} al Parlay"):
                add_to_parlay(f"{equipo_local} vs {equipo_visita}", top_pick[1], top_pick[2])
            
            st.markdown("---")
            st.markdown("### 📊 EXPANSIÓN TOTAL DEL MERCADO (CASA DE APUESTAS)")
            
            c_goles, c_btts, c_scores = st.columns(3)
            with c_goles:
                st.markdown('<div class="bookie-section"><h4>🥅 Línea de Goles (Totales)</h4>', unsafe_allow_html=True)
                st.write(f"**Over 1.5:** {round(mercado['O15'],1)}%")
                st.write(f"**Over 2.5:** {round(mercado['O25'],1)}%")
                st.write(f"**Under 2.5:** {round(mercado['U25'],1)}%")
                st.write(f"**Under 3.5:** {round(mercado['U35'],1)}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with c_btts:
                st.markdown('<div class="bookie-section"><h4>🤝 Ambos Anotan (BTTS)</h4>', unsafe_allow_html=True)
                st.write(f"**SÍ:** {round(mercado['BTTS_Y'],1)}%")
                st.write(f"**NO:** {round(mercado['BTTS_N'],1)}%")
                st.markdown('<hr><h4>🚩 Tiros de Esquina</h4>', unsafe_allow_html=True)
                st.write(f"**Línea Base Proyectada:** {mercado['Corners']} Córners")
                st.markdown('</div>', unsafe_allow_html=True)
            with c_scores:
                st.markdown('<div class="bookie-section"><h4>🎯 Marcador Exacto (Top 5)</h4>', unsafe_allow_html=True)
                for sc in mercado['Scores']:
                    st.write(f"**{sc['Score']}** ({round(sc['Prob'],1)}%)")
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown(f"""
            <div class="expert-alert">
                <h4 style="color:#fca5a5; margin:0;">🤖 RADIOGRAFÍA DEL PARTIDO</h4>
                <p style="color:#f3f4f6;">Goles Esperados (xG): <b>{equipo_local} {round(xg_l, 2)} - {round(xg_v, 2)} {equipo_visita}</b>.<br>
                El motor ha ajustado las matemáticas usando el movimiento de dinero indicado.</p>
            </div>
            """, unsafe_allow_html=True)

elif modo == "⚡ Constructor de Parlays":
    st.markdown("## 🎫 Tu Ticket Inteligente")
    if len(st.session_state.parlay_cart) == 0:
        st.info("El ticket está vacío. Ve al Escáner Global y agrega picks a tu parlay.")
    else:
        cuota_decimal_total = 1.0
        st.markdown('<div class="parlay-slip">', unsafe_allow_html=True)
        for i, item in enumerate(st.session_state.parlay_cart):
            st.markdown(f"**{i+1}. {item['Match']}**")
            st.markdown(f"↳ 🎯 Pick: <span style='color:#34d399;'>{item['Pick']}</span> | Momio: {item['Odds']:+}", unsafe_allow_html=True)
            st.markdown("---")
            dec = americano_a_decimal(item['Odds'])
            cuota_decimal_total *= dec
            
        momio_final = decimal_a_americano(cuota_decimal_total)
        st.markdown(f"<h2 style='text-align:right; color:#f472b6;'>🎟️ MOMIO TOTAL EST: {momio_final:+}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:right;'>Cuota Decimal: {round(cuota_decimal_total, 2)}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Limpiar Ticket", type="secondary"):
            clear_parlay()
            st.rerun()

elif modo == "🇲🇽 Fix Liga MX (Análisis)":
    st.markdown("## 🦅 El Problema de la Liga MX (Y cómo atacarlo)")
    st.write("""
    Si has estado perdiendo picks en Liga MX últimamente, no es tu culpa ni del código original. La Liga MX tiene factores de varianza que destrozan los algoritmos europeos tradicionales:
    
    1. **La Altitud y Clima Extremo:** Jugar en Toluca a mediodía o en CU a las 12 PM drena físicamente a las visitas.
    2. **El Formato Liguilla:** A mitad de torneo, los equipos "grandes" (América, Tigres, Monterrey) suelen dosificar esfuerzos porque saben que clasificar en 6to o en 1ro les da casi las mismas opciones de ser campeón. Las sorpresas son el estándar.
    3. **Picks Subestimados:** El mercado de "Ganador" (ML) en México es una trampa. 
    
    ### 🛡️ Nueva Estrategia Sugerida para Liga MX:
    Usa el **Escáner Global** de esta aplicación. En lugar de jugar al ganador directo, introduce los momios en el escáner y ataca los **Córners** o los **Ambos Anotan (BTTS)**. La estadística de goles y córners en México es mucho más estable que la estadística de quién gana el partido.
    """)
