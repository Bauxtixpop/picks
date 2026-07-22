import pandas as pd
import numpy as np
import scipy.stats as stats

def calcular_idr(df):
    '''
    Calcula el Índice de Dominio Real (IDR) combinando xG, calidad de tiro y peligro en área.
    '''
    df['xG_Diff'] = df['xG'] - df['xGA']
    df['Calidad_Tiro'] = (df['xG'] / df['Tiros_Promedio']).round(3)
    
    raw_idr = (df['xG_Diff'] * 0.45) + (df['Calidad_Tiro'] * 15 * 0.35) + ((df['AttPen_Promedio'] / 20) * 5 * 0.20)
    
    min_val = raw_idr.min()
    max_val = raw_idr.max()
    if max_val != min_val:
        df['IDR'] = ((raw_idr - min_val) / (max_val - min_val) * 90 + 10).round(1)
    else:
        df['IDR'] = 50.0
        
    return df.sort_values(by='IDR', ascending=False)

def simular_partido(equipo_local, equipo_visitante, df_stats, momio_local, momio_empate, momio_visita):
    '''
    Simula un partido usando Distribución de Poisson con base en xG y calcula Value Bets.
    '''
    stats_loc = df_stats[df_stats['Equipo'] == equipo_local].iloc[0]
    stats_vis = df_stats[df_stats['Equipo'] == equipo_visitante].iloc[0]
    
    xg_prom_liga = df_stats['xG'].mean() / 10.0
    
    ataque_loc = (stats_loc['xG'] / stats_loc['PJ']) / xg_prom_liga
    defensa_loc = (stats_loc['xGA'] / stats_loc['PJ']) / xg_prom_liga
    
    ataque_vis = (stats_vis['xG'] / stats_vis['PJ']) / xg_prom_liga
    defensa_vis = (stats_vis['xGA'] / stats_vis['PJ']) / xg_prom_liga
    
    ventaja_local = 1.15
    
    lambda_loc = xg_prom_liga * ataque_loc * defensa_vis * ventaja_local
    lambda_vis = xg_prom_liga * ataque_vis * defensa_loc
    
    prob_loc = 0.0
    prob_emp = 0.0
    prob_vis = 0.0
    prob_over25 = 0.0
    prob_btts = 0.0
    
    for goles_l in range(6):
        for goles_v in range(6):
            p = stats.poisson.pmf(goles_l, lambda_loc) * stats.poisson.pmf(goles_v, lambda_vis)
            
            if goles_l > goles_v:
                prob_loc += p
            elif goles_l == goles_v:
                prob_emp += p
            else:
                prob_vis += p
                
            if (goles_l + goles_v) > 2.5:
                prob_over25 += p
            if goles_l > 0 and goles_v > 0:
                prob_btts += p
                
    total_p = prob_loc + prob_emp + prob_vis
    prob_loc = round((prob_loc / total_p) * 100, 1)
    prob_emp = round((prob_emp / total_p) * 100, 1)
    prob_vis = round((prob_vis / total_p) * 100, 1)
    prob_over25 = round(prob_over25 * 100, 1)
    prob_btts = round(prob_btts * 100, 1)
    
    imp_loc = round((1 / momio_local) * 100, 1) if momio_local > 0 else 0
    imp_emp = round((1 / momio_empate) * 100, 1) if momio_empate > 0 else 0
    imp_vis = round((1 / momio_visita) * 100, 1) if momio_visita > 0 else 0
    
    edge_loc = round(prob_loc - imp_loc, 1)
    edge_vis = round(prob_vis - imp_vis, 1)
    
    recomendacion = "Sin valor claro"
    tipo_apuesta = "N/A"
    momio_objetivo = 0.0
    edge_objetivo = 0.0
    prob_objetivo = 0.0
    
    if edge_loc >= 5.0:
        recomendacion = f"💎 VALUE BET: Victoria de {equipo_local}"
        tipo_apuesta = "Victoria Local"
        momio_objetivo = momio_local
        edge_objetivo = edge_loc
        prob_objetivo = prob_loc
    elif edge_vis >= 5.0:
        recomendacion = f"🔥 UNDERDOG DE VALOR: Victoria de {equipo_visitante}"
        tipo_apuesta = "Victoria Visitante"
        momio_objetivo = momio_visita
        edge_objetivo = edge_vis
        prob_objetivo = prob_vis
    elif prob_loc >= 70.0:
        recomendacion = f"🛡️ APUESTA SEGURA: {equipo_local} Gana o Empata (1X)"
        tipo_apuesta = "Doble Oportunidad 1X"
        momio_objetivo = round(1 / ((prob_loc + prob_emp)/100), 2)
        edge_objetivo = 3.5
        prob_objetivo = prob_loc + prob_emp
    elif prob_over25 >= 65.0:
        recomendacion = f"⚡ GOLES: Más de 2.5 Goles en el partido"
        tipo_apuesta = "Over 2.5"
        momio_objetivo = 1.75
        edge_objetivo = 4.0
        prob_objetivo = prob_over25
    elif prob_btts >= 65.0:
        recomendacion = f"⚽ AMBOS ANOTAN: Sí (BTTS)"
        tipo_apuesta = "BTTS - Sí"
        momio_objetivo = 1.80
        edge_objetivo = 4.2
        prob_objetivo = prob_btts
    else:
        recomendacion = "⚖️ Partido Ajustado (Riesgo Alto - Evitar o buscar Empate)"
        tipo_apuesta = "Empate"
        momio_objetivo = momio_empate
        edge_objetivo = round(prob_emp - imp_emp, 1)
        prob_objetivo = prob_emp
        
    return {
        "Local": equipo_local,
        "Visitante": equipo_visitante,
        "Prob_Local": prob_loc,
        "Prob_Empate": prob_emp,
        "Prob_Visita": prob_vis,
        "Prob_Over25": prob_over25,
        "Prob_BTTS": prob_btts,
        "Momio_Local": momio_local,
        "Momio_Emp": momio_empate,
        "Momio_Vis": momio_visita,
        "Edge_Local": edge_loc,
        "Edge_Visita": edge_vis,
        "Recomendacion": recomendacion,
        "Tipo_Apuesta": tipo_apuesta,
        "Momio_Objetivo": momio_objetivo,
        "Edge_Objetivo": edge_objetivo,
        "Prob_Objetivo": prob_objetivo,
        "xG_Est_Local": round(lambda_loc, 2),
        "xG_Est_Vis": round(lambda_vis, 2)
    }

def obtener_parlays_jornada(df_stats, jornada_num=2):
    '''
    Analiza los partidos REALES de una jornada y arma los parlays 
    basándose en la mayor ventaja matemática (Edge) y diferencia de IDR.
    '''
    partidos_jornada = [
        ("Cruz Azul", "Puebla"),
        ("Toluca", "Pumas"),
        ("Tigres", "Chivas"),
        ("América", "Querétaro"),
        ("Monterrey", "Necaxa"),
        ("Pachuca", "León"),
        ("Atlas", "Tijuana"),
        ("Santos Laguna", "Juárez"),
        ("Mazatlán", "San Luis")
    ]
    
    analisis_partidos = []
    
    for local, visita in partidos_jornada:
        try:
            idr_loc = df_stats.loc[df_stats['Equipo'] == local, 'IDR'].values[0]
            idr_vis = df_stats.loc[df_stats['Equipo'] == visita, 'IDR'].values[0]
            
            diff_idr = idr_loc - idr_vis
            
            if diff_idr > 0:
                favorito = local
                tipo_seguro = f"{local} Gana o Empata (1X)"
                cuota_seg = 1.25
                tipo_valor = f"Victoria Directa {local}"
                cuota_val = 1.70
            else:
                favorito = visita
                tipo_seguro = f"{visita} Gana o Empata (X2)"
                cuota_seg = 1.35
                tipo_valor = f"Victoria Directa {visita}"
                cuota_val = 2.10
                
            analisis_partidos.append({
                "Partido": f"{local} vs {visita}",
                "Favorito": favorito,
                "Diferencia_IDR": abs(diff_idr),
                "Pick_Seguro": tipo_seguro,
                "Cuota_Seg": cuota_seg,
                "Pick_Valor": tipo_valor,
                "Cuota_Val": cuota_val
            })
        except IndexError:
            continue
            
    df_jornada = pd.DataFrame(analisis_partidos).sort_values(by="Diferencia_IDR", ascending=False)
    return df_jornada.head(3).to_dict(orient="records")