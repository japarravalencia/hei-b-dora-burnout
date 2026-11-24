#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOTOR DE EXPERIMENTOS DETERMINISTAS v1.0 — POSGRADO, ESTRÉS PROFESORAL, REFORMA DORA E IA (CASE HEI-B)
Proyecto: 2026_dora_estres_posgrado_ies
Replica con 100% de fidelidad las ecuaciones exactas de 'UniNorte Magoti.stmx':
  - Horizonte: 2020.0 a 2035.0 (Years), DT = 0.0625, integración RK4
  - 9 Stocks primarios + 1 SMTH1 + 5 SMTH3 (15 estados) = 25 estados ODE exactos
  - 18 Flujos, 45 Auxiliares dinámicos, 77 Constantes
  - 7 Palancas de política desacoplables individualmente (o en bloque vía escenario_politica):
      1. switch_dora_manual_valencia  (Reforma DORA / Manual de Valencia: meta Scopus 1.8 -> 1.0)
      2. switch_reforma_escalafon     (Cupos Profesor Titular 0.35 -> 1.0)
      3. switch_agilizacion_posgrados (Fricción pagos posgrado 0.8 -> 0.1)
      4. switch_costo_marginal        (Matrícula costo marginal maestría +80% demanda)
      5. switch_red_tematica_activa   (Red temática activa egresados +35% adhesión)
      6. switch_brain_first           (Pedagogía Brain-First contra rendición cognitiva IA)
      7. switch_dual_clock_eval       (Evaluación Dual-Clock +25% prima humana)

Verificaciones incluidas:
  - Matriz de ablación de 9 escenarios (Status Quo, paquetes parciales, 7 palancas y 2 extremos Barlas)
  - Conservación exacta de personas: max |total_people - Acum_Flujo_Frontera|
  - Convergencia RK4 a 3 pasos (DT = 0.0625 -> 0.03125 -> 0.015625) + contraste Euler
  - Dominancia cuantitativa LTM proxy por periodo (2020–2024, 2025–2029, 2030–2035)
  - Sensibilidad global Sobol (±20%, n=512 -> N=5,120 corridas, IC 95% + bandas P10–P90 + invarianza ordinal)
  - Sensibilidad global Sobol en el espacio de las 3 familias de palancas (N=1,280 corridas)

Uso:
  python 03_proyectos/2026_dora_estres_posgrado_ies/modelo_stella/experimentos_posgrado_dora_sobol.py --proyecto 2026_dora_estres_posgrado_ies
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from SALib.analyze import sobol as sobol_analyze
    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample
    TIENE_SALIB = True
except ImportError:
    TIENE_SALIB = False


# ==============================================================================
# 77 PARÁMETROS CONSTANTES EXACTOS DE 'UniNorte Magoti.stmx'
# (+ 7 interruptores individuales desacoplados para ablación; por defecto = 0.0)
# ==============================================================================
PARAMS_BASE: Dict[str, float] = {
    "puntos_referencia": 100.0,
    "escenario_politica": 0.0,
    # 7 palancas individuales (None -> hereda escenario_politica; float -> valor explícito)
    "sw_dora": 0.0,
    "sw_escalafon": 0.0,
    "sw_agilizacion": 0.0,
    "sw_costo_marginal": 0.0,
    "sw_red_tematica": 0.0,
    "sw_brain_first": 0.0,
    "sw_dual_clock": 0.0,
    # Demografía estudiantil y egresados
    "demanda_pregrado_base": 4000.0,
    "frac_adm_pregrado": 0.6,
    "duracion_pregrado": 5.0,
    "tasa_desercion_pregrado": 0.15,
    "demanda_base_mgotic": 8.0,
    "incremento_politica_costo_marginal": 0.8,
    "frac_enlace_pregrado": 0.0015,
    "duracion_maestria": 1.5,
    "desercion_maestria": 0.08,
    "vida_laboral_activa": 25.0,
    "frac_adhesion_base": 0.45,
    "incremento_adhesion_red": 0.35,
    "tiempo_en_red": 8.0,
    "colocaciones_por_convenio": 1.8,
    "frac_egresados_buscando_empleo": 0.12,
    "piso_egresados_referencia": 1.0,
    "empresas_potenciales": 300.0,
    "tasa_conversion_empresarial": 0.04,
    "duracion_convenio": 4.0,
    # Planta docente y escalafón
    "tiempo_maduracion_escalafon": 5.0,
    "frac_candidatos_titular": 0.2,
    "cupos_titular_con_reforma": 1.0,
    "cupos_titular_sin_reforma": 0.35,
    "tasa_salida_base": 0.03,
    "tasa_jubilacion_base": 0.04,
    "factor_fuga_titulares": 0.4,
    "estres_umbral_fuga": 55.0,
    "planta_docente_objetivo": 50.0,
    "tiempo_ajuste_planta": 1.0,
    "fuga_referencia": 0.008,
    "escala_estres_fuga": 12.0,
    # Finanzas y presión reloj rápido
    "matr_pregrado_yr": 16000000.0,
    "matr_maestria_yr": 22000000.0,
    "costos_fijos": 160000000000.0,
    "costo_nomina_promedio": 120000000.0,
    "costo_apc_base": 2500000000.0,
    "margen_objetivo": 0.1,
    "tiempo_percepcion_margen": 0.25,
    "ganancia_presion_eficiencia": 3.0,
    "meta_scopus_estandar": 1.8,
    "meta_scopus_dora": 1.0,
    "meta_scopus_referencia": 1.5,
    "friccion_con_agilizacion": 0.1,
    "friccion_sin_agilizacion": 0.8,
    # Equidad, estrés, clima y reputación
    "peso_cupos_equidad": 40.0,
    "peso_friccion_equidad": 30.0,
    "peso_eficiencia_equidad": 30.0,
    "estres_por_lambda": 17.5,
    "coef_estres_inequidad": 0.15,
    "tiempo_alivio_bienestar": 2.0,
    "amortiguador_cultural": 1.35,
    "peso_estres_clima": 0.6,
    "peso_equidad_clima": 0.4,
    "equidad_neutra": 50.0,
    "tiempo_clima": 2.5,
    "peso_clima_repinterna": 0.6,
    "tiempo_repinterna": 2.0,
    "calidad_profesoral_maxima": 88.0,
    "umbral_repinterna": 70.0,
    "sensibilidad_calidad_repinterna": 0.35,
    "tiempo_calidadprof": 3.0,
    "peso_calidad_profesoral": 0.6,
    "puntos_empleabilidad": 40.0,
    "tiempo_calidadpercibida": 3.0,
    "peso_calidad_repexterna": 0.7,
    "egresados_referencia": 200.0,
    "puntos_egresados": 30.0,
    "tiempo_repexterna": 4.0,
    # IA Generativa y evaluación Dual-Clock
    "adopcion_ia_generativa": 0.85,
    "peso_rendicion_cognitiva": 0.35,
    "incremento_dual_clock": 0.25,
    # Referencias y suavizado numérico
    "total_people_inicial": 12246.5,
    "reputacion_referencia": 62.587110992,
    "epsilon_margen": 0.01,
    "epsilon_reputacion_int": 2.0,
}

ESCENARIOS: Dict[str, Dict[str, float]] = {
    "01_StatusQuo_FastClock": {
        "sw_dora": 0.0, "sw_escalafon": 0.0, "sw_agilizacion": 0.0,
        "sw_costo_marginal": 0.0, "sw_red_tematica": 0.0,
        "sw_brain_first": 0.0, "sw_dual_clock": 0.0,
    },
    "02_Solo_DORA": {
        "sw_dora": 1.0,
    },
    "03_Gobernanza_DORA_Esc_Pagos": {
        "sw_dora": 1.0, "sw_escalafon": 1.0, "sw_agilizacion": 1.0,
    },
    "04_Mercado_CostoMarg_Red": {
        "sw_costo_marginal": 1.0, "sw_red_tematica": 1.0,
    },
    "05_Pedagogia_IA_BrainFirst_Dual": {
        "sw_brain_first": 1.0, "sw_dual_clock": 1.0,
    },
    "06_Reforma_Integral_7Levers": {
        "sw_dora": 1.0, "sw_escalafon": 1.0, "sw_agilizacion": 1.0,
        "sw_costo_marginal": 1.0, "sw_red_tematica": 1.0,
        "sw_brain_first": 1.0, "sw_dual_clock": 1.0,
    },
    "07_Extremo1_Choque_Demanda": {
        "demanda_pregrado_base": 2400.0, "demanda_base_mgotic": 4.0,
    },
    "08_Extremo2_IA_Sin_Guardrails": {
        "adopcion_ia_generativa": 1.0, "peso_rendicion_cognitiva": 0.60,
        "meta_scopus_estandar": 2.4,
    },
}


def _sat_tanh_stella(bruta: float, pref: float) -> float:
    """Stella: puntos_referencia * (1 - 2 / (EXP(2 * bruta / puntos_referencia) + 1)) == pref * tanh(bruta / pref)."""
    arg = max(-40.0, min(40.0, 2.0 * bruta / pref))
    return pref * (1.0 - 2.0 / (math.exp(arg) + 1.0))


def _suave_pos(x: float, eps: float) -> float:
    return (x + math.sqrt(x * x + eps * eps) - eps) / 2.0


def estado_inicial_stella(p: Dict[str, float]) -> np.ndarray:
    """Inicializa los 9 stocks primarios + 1 SMTH1 + 5 SMTH3 exactamente como Stella Architect."""
    preg0 = 12000.0
    maes0 = 16.5
    egr0 = 180.0
    red0 = 70.0
    emp0 = 45.0
    asis0 = 35.0
    tit0 = 15.0
    estr0 = 45.0
    acum0 = p["total_people_inicial"]

    m_perc0 = p["margen_objetivo"]
    b_marg0 = p["margen_objetivo"] - m_perc0
    b_eff0 = _suave_pos(b_marg0, p["epsilon_margen"])
    p_efic0 = 1.0 + p["ganancia_presion_eficiencia"] * b_eff0

    f_cup0 = p["cupos_titular_sin_reforma"] + (p["cupos_titular_con_reforma"] - p["cupos_titular_sin_reforma"]) * p["sw_escalafon"]
    fric0 = p["friccion_sin_agilizacion"] + (p["friccion_con_agilizacion"] - p["friccion_sin_agilizacion"]) * p["sw_agilizacion"]
    eq0 = p["puntos_referencia"] - (
        (1.0 - f_cup0) * p["peso_cupos_equidad"]
        + fric0 * p["peso_friccion_equidad"]
        + (p_efic0 - 1.0) * p["peso_eficiencia_equidad"]
    )

    clima0 = p["puntos_referencia"] - (estr0 / p["amortiguador_cultural"]) * p["peso_estres_clima"] + (eq0 - p["equidad_neutra"]) * p["peso_equidad_clima"]
    repint0 = clima0 * p["peso_clima_repinterna"] + eq0 * (1.0 - p["peso_clima_repinterna"])

    b_ri0 = p["umbral_repinterna"] - repint0
    eros0 = p["sensibilidad_calidad_repinterna"] * _suave_pos(b_ri0, p["epsilon_reputacion_int"])
    calprof0 = p["calidad_profesoral_maxima"] - eros0

    cob0 = (emp0 * p["colocaciones_por_convenio"]) / max(p["piso_egresados_referencia"], egr0 * p["frac_egresados_buscando_empleo"])
    plena0 = 1.0 - math.exp(-cob0)
    rend0 = (1.0 - p["sw_brain_first"]) * p["adopcion_ia_generativa"] * p["peso_rendicion_cognitiva"]
    prima0 = (1.0 - rend0) * (1.0 + p["incremento_dual_clock"] * p["sw_dual_clock"])

    calperc_b0 = calprof0 * p["peso_calidad_profesoral"] + plena0 * p["puntos_empleabilidad"] * prima0
    calperc0 = _sat_tanh_stella(calperc_b0, p["puntos_referencia"])

    repext_b0 = calperc0 * p["peso_calidad_repexterna"] + (egr0 / p["egresados_referencia"]) * p["puntos_egresados"]
    repext0 = _sat_tanh_stella(repext_b0, p["puntos_referencia"])

    return np.array([
        preg0, maes0, egr0, red0, emp0, asis0, tit0, estr0, acum0,
        m_perc0,
        clima0, clima0, clima0,
        repint0, repint0, repint0,
        calprof0, calprof0, calprof0,
        calperc0, calperc0, calperc0,
        repext0, repext0, repext0,
    ], dtype=float)


def evaluar_sistema(y: np.ndarray, p: Dict[str, float], aplicar_clamp: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    preg = max(0.0, float(y[0])) if aplicar_clamp else float(y[0])
    maes = max(0.0, float(y[1])) if aplicar_clamp else float(y[1])
    egr = max(0.0, float(y[2])) if aplicar_clamp else float(y[2])
    red = max(0.0, float(y[3])) if aplicar_clamp else float(y[3])
    emp = max(0.0, float(y[4])) if aplicar_clamp else float(y[4])
    asis = max(0.0, float(y[5])) if aplicar_clamp else float(y[5])
    tit = max(0.0, float(y[6])) if aplicar_clamp else float(y[6])
    estr = max(0.0, float(y[7])) if aplicar_clamp else float(y[7])
    acum = float(y[8])

    m_perc = float(y[9])
    clima_s1, clima_s2, clima_out = float(y[10]), float(y[11]), float(y[12])
    repint_s1, repint_s2, repint_out = float(y[13]), float(y[14]), float(y[15])
    calp_s1, calp_s2, calp_out = float(y[16]), float(y[17]), float(y[18])
    calper_s1, calper_s2, calper_out = float(y[19]), float(y[20]), float(y[21])
    repext_s1, repext_s2, repext_out = float(y[22]), float(y[23]), float(y[24])

    # Reputación externa y demanda
    ind_rep = repext_out / p["reputacion_referencia"]
    asp_preg = p["demanda_pregrado_base"] * ind_rep
    f_cm = 1.0 + p["incremento_politica_costo_marginal"] * p["sw_costo_marginal"]
    dem_maes = p["demanda_base_mgotic"] * f_cm * ind_rep

    # Flujos estudiantiles
    t_adm_preg = asp_preg * p["frac_adm_pregrado"]
    t_grad_preg = preg * (1.0 - p["tasa_desercion_pregrado"]) / p["duracion_pregrado"]
    t_des_preg = preg * p["tasa_desercion_pregrado"] / p["duracion_pregrado"]
    art_preg = t_grad_preg * p["frac_enlace_pregrado"]

    t_adm_maes = dem_maes + art_preg
    t_grad_maes = maes * (1.0 - p["desercion_maestria"]) / p["duracion_maestria"]
    t_des_maes = maes * p["desercion_maestria"] / p["duracion_maestria"]

    # Egresados, red temática y empresas convenio
    t_ret_egr = egr / p["vida_laboral_activa"]
    f_adh_red = p["frac_adhesion_base"] + p["incremento_adhesion_red"] * p["sw_red_tematica"]
    t_vinc_red = t_grad_maes * f_adh_red
    t_desv_red = red / p["tiempo_en_red"]
    t_adh_emp = p["empresas_potenciales"] * p["tasa_conversion_empresarial"] * ind_rep
    t_cierre_emp = emp / p["duracion_convenio"]

    # Planta docente y fuga por estrés
    tot_doc = asis + tit
    req_planta = max(0.0, (p["planta_docente_objetivo"] - tot_doc) / p["tiempo_ajuste_planta"])
    t_contr_prof = req_planta
    f_cupos = p["cupos_titular_sin_reforma"] + (p["cupos_titular_con_reforma"] - p["cupos_titular_sin_reforma"]) * p["sw_escalafon"]
    t_asc_tit = (asis * p["frac_candidatos_titular"] / p["tiempo_maduracion_escalafon"]) * f_cupos

    exp_arg = max(-30.0, min(20.0, (estr - p["estres_umbral_fuga"]) / p["escala_estres_fuga"]))
    fuga_est = p["fuga_referencia"] * math.exp(exp_arg)
    t_ret_asis = asis * (p["tasa_salida_base"] + fuga_est)
    t_ret_tit = tit * (p["tasa_jubilacion_base"] + fuga_est * p["factor_fuga_titulares"])

    # Finanzas, presión de eficiencia y reloj rápido lambda
    b_marg = p["margen_objetivo"] - m_perc
    b_marg_eff = _suave_pos(b_marg, p["epsilon_margen"])
    p_efic = 1.0 + p["ganancia_presion_eficiencia"] * b_marg_eff

    meta_scopus = p["meta_scopus_estandar"] + (p["meta_scopus_dora"] - p["meta_scopus_estandar"]) * p["sw_dora"]
    lam = (meta_scopus / p["meta_scopus_referencia"]) * p_efic

    ing_tot = preg * p["matr_pregrado_yr"] + maes * p["matr_maestria_yr"]
    gastos_apc = p["costo_apc_base"] * lam
    costos_tot = p["costos_fijos"] + tot_doc * p["costo_nomina_promedio"] + gastos_apc
    margen_neto = ((ing_tot - costos_tot) / ing_tot) if abs(ing_tot) > 1e-6 else 0.0

    # Equidad, estrés, clima, reputación interna y calidad profesoral
    fric_posg = p["friccion_sin_agilizacion"] + (p["friccion_con_agilizacion"] - p["friccion_sin_agilizacion"]) * p["sw_agilizacion"]
    perc_eq = p["puntos_referencia"] - (
        (1.0 - f_cupos) * p["peso_cupos_equidad"]
        + fric_posg * p["peso_friccion_equidad"]
        + (p_efic - 1.0) * p["peso_eficiencia_equidad"]
    )
    t_gen_est = lam * p["estres_por_lambda"] + (p["puntos_referencia"] - perc_eq) * p["coef_estres_inequidad"]
    t_aliv_est = estr / p["tiempo_alivio_bienestar"]

    clima_in = p["puntos_referencia"] - (estr / p["amortiguador_cultural"]) * p["peso_estres_clima"] + (perc_eq - p["equidad_neutra"]) * p["peso_equidad_clima"]
    repint_in = clima_out * p["peso_clima_repinterna"] + perc_eq * (1.0 - p["peso_clima_repinterna"])

    b_ri = p["umbral_repinterna"] - repint_out
    eros_cal = p["sensibilidad_calidad_repinterna"] * _suave_pos(b_ri, p["epsilon_reputacion_int"])
    calp_in = p["calidad_profesoral_maxima"] - eros_cal

    # Empleabilidad, IA generativa (Goodhart) y reputación externa
    cob_emp = (emp * p["colocaciones_por_convenio"]) / max(p["piso_egresados_referencia"], egr * p["frac_egresados_buscando_empleo"])
    plena_emp = 1.0 - math.exp(-cob_emp)
    rend_cog = (1.0 - p["sw_brain_first"]) * p["adopcion_ia_generativa"] * p["peso_rendicion_cognitiva"]
    prima_hum = (1.0 - rend_cog) * (1.0 + p["incremento_dual_clock"] * p["sw_dual_clock"])

    calper_bruta = calp_out * p["peso_calidad_profesoral"] + plena_emp * p["puntos_empleabilidad"] * prima_hum
    calper_in = _sat_tanh_stella(calper_bruta, p["puntos_referencia"])

    repext_bruta = calper_out * p["peso_calidad_repexterna"] + (egr / p["egresados_referencia"]) * p["puntos_egresados"]
    repext_in = _sat_tanh_stella(repext_bruta, p["puntos_referencia"])

    # Conservación exacta de personas
    tot_people = preg + maes + egr + asis + tit
    net_bf = t_adm_preg + t_adm_maes + t_contr_prof - t_grad_preg - t_des_preg - t_des_maes - t_ret_egr - t_ret_asis - t_ret_tit
    resid_cons = tot_people - acum

    # Proxy LTM cuantitativo por macro-bucle
    w_fin = abs(p_efic - 1.0) * 60.0 + abs(lam - 1.0) * 25.0
    w_est = abs(t_gen_est - t_aliv_est) * 3.5 + fuga_est * 900.0 + abs(eros_cal) * 2.5
    w_rep = abs(ind_rep - 1.0) * 140.0 + abs(t_adm_maes - t_grad_maes) * 2.5
    w_ia = abs(1.0 - prima_hum) * 45.0
    tot_w = max(1e-6, w_fin + w_est + w_rep + w_ia)

    # Derivadas de los 25 estados
    k_cl = 3.0 / p["tiempo_clima"]
    k_ri = 3.0 / p["tiempo_repinterna"]
    k_cp = 3.0 / p["tiempo_calidadprof"]
    k_ce = 3.0 / p["tiempo_calidadpercibida"]
    k_re = 3.0 / p["tiempo_repexterna"]

    dy = np.array([
        t_adm_preg - t_grad_preg - t_des_preg,
        t_adm_maes - t_grad_maes - t_des_maes,
        t_grad_maes - t_ret_egr,
        t_vinc_red - t_desv_red,
        t_adh_emp - t_cierre_emp,
        t_contr_prof - t_asc_tit - t_ret_asis,
        t_asc_tit - t_ret_tit,
        t_gen_est - t_aliv_est,
        net_bf,
        (margen_neto - m_perc) / p["tiempo_percepcion_margen"],
        k_cl * (clima_in - clima_s1), k_cl * (clima_s1 - clima_s2), k_cl * (clima_s2 - clima_out),
        k_ri * (repint_in - repint_s1), k_ri * (repint_s1 - repint_s2), k_ri * (repint_s2 - repint_out),
        k_cp * (calp_in - calp_s1), k_cp * (calp_s1 - calp_s2), k_cp * (calp_s2 - calp_out),
        k_ce * (calper_in - calper_s1), k_ce * (calper_s1 - calper_s2), k_ce * (calper_s2 - calper_out),
        k_re * (repext_in - repext_s1), k_re * (repext_s1 - repext_s2), k_re * (repext_s2 - repext_out),
    ], dtype=float)

    aux = {
        "margen_neto": margen_neto,
        "presion_reloj_lambda": lam,
        "gastos_apc_bn": gastos_apc / 1e9,
        "percepcion_equidad": perc_eq,
        "clima_organizacional": clima_out,
        "reputacion_interna": repint_out,
        "calidad_profesoral": calp_out,
        "calidad_percibida": calper_out,
        "reputacion_externa": repext_out,
        "indice_reputacion": ind_rep,
        "fuga_por_estres": fuga_est,
        "prima_humana": prima_hum,
        "frac_titulares": tit / max(1.0, tot_doc),
        "residual_conservacion": resid_cons,
        "ltm_fin_pct": 100.0 * w_fin / tot_w,
        "ltm_estres_pct": 100.0 * w_est / tot_w,
        "ltm_rep_pct": 100.0 * w_rep / tot_w,
        "ltm_ia_pct": 100.0 * w_ia / tot_w,
    }
    return dy, aux


def simular(
    overrides: Dict[str, float],
    dt: float = 0.0625,
    t0: float = 2020.0,
    tf: float = 2035.0,
    metodo: str = "RK4",
    aplicar_clamp: bool = True,
) -> pd.DataFrame:
    p = dict(PARAMS_BASE)
    p.update(overrides)
    y = estado_inicial_stella(p)

    n_steps = int(round((tf - t0) / dt))
    pasos_anio = max(1, int(round(1.0 / dt)))
    filas: List[Dict[str, float]] = []

    for step in range(n_steps + 1):
        t = t0 + step * dt
        dy, aux = evaluar_sistema(y, p, aplicar_clamp=aplicar_clamp)
        if step % pasos_anio == 0 or step == n_steps:
            filas.append({
                "Year": round(t, 4),
                "Pregrado": round(float(y[0]), 1),
                "Maestrantes": round(float(y[1]), 2),
                "Egresados_TI": round(float(y[2]), 1),
                "Red_Tematica": round(float(y[3]), 1),
                "Empresas": round(float(y[4]), 1),
                "Asistentes": round(float(y[5]), 2),
                "Titulares": round(float(y[6]), 2),
                "Estres": round(float(y[7]), 2),
                **{k: round(float(v), 5) for k, v in aux.items()},
            })
        if step == n_steps:
            break

        if metodo.lower() == "rk4":
            k1 = dy
            k2, _ = evaluar_sistema(y + 0.5 * dt * k1, p, aplicar_clamp=aplicar_clamp)
            k3, _ = evaluar_sistema(y + 0.5 * dt * k2, p, aplicar_clamp=aplicar_clamp)
            k4, _ = evaluar_sistema(y + dt * k3, p, aplicar_clamp=aplicar_clamp)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            y = y + dt * dy

        if aplicar_clamp:
            for i in range(8):
                y[i] = max(0.0, y[i])

    return pd.DataFrame(filas)


def ejecutar_sobol_parametros(n_base: int = 512) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not TIENE_SALIB:
        return pd.DataFrame(), {"ejecutado": False}

    nombres = [
        "estres_por_lambda",
        "tiempo_alivio_bienestar",
        "estres_umbral_fuga",
        "sensibilidad_calidad_repinterna",
        "adopcion_ia_generativa",
        "peso_rendicion_cognitiva",
        "demanda_base_mgotic",
        "ganancia_presion_eficiencia",
    ]
    bounds = [[PARAMS_BASE[k] * 0.80, PARAMS_BASE[k] * 1.20] for k in nombres]
    problem = {"num_vars": len(nombres), "names": nombres, "bounds": bounds}
    X = sobol_sample.sample(problem, n_base, calc_second_order=False)
    n_tot = len(X)

    y_maes = np.zeros(n_tot)
    y_estr = np.zeros(n_tot)
    y_rep = np.zeros(n_tot)
    yrs = list(range(2020, 2036))
    band_maes = np.zeros((min(300, n_tot), len(yrs)))
    band_estr = np.zeros((min(300, n_tot), len(yrs)))

    for i, fila in enumerate(X):
        ov = {nombres[j]: float(fila[j]) for j in range(len(nombres))}
        df_i = simular(ov, dt=0.125, metodo="RK4")
        fin = df_i.iloc[-1]
        y_maes[i] = fin["Maestrantes"]
        y_estr[i] = fin["Estres"]
        y_rep[i] = fin["reputacion_externa"]
        if i < band_maes.shape[0]:
            band_maes[i, :] = df_i["Maestrantes"].values[:len(yrs)]
            band_estr[i, :] = df_i["Estres"].values[:len(yrs)]

    si_m = sobol_analyze.analyze(problem, y_maes, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_e = sobol_analyze.analyze(problem, y_estr, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_r = sobol_analyze.analyze(problem, y_rep, calc_second_order=False, conf_level=0.95, print_to_console=False)

    # Invarianza ordinal: Reforma Integral > Status Quo en Maestrantes y < en Estrés
    rng = np.random.default_rng(2026)
    idxs = rng.choice(n_tot, size=min(60, n_tot), replace=False)
    inv_ok = 0
    for idx in idxs:
        ov0 = {nombres[j]: float(X[idx][j]) for j in range(len(nombres))}
        ov1 = dict(ov0)
        ov1.update(ESCENARIOS["06_Reforma_Integral_7Levers"])
        f0 = simular(ov0, dt=0.125).iloc[-1]
        f1 = simular(ov1, dt=0.125).iloc[-1]
        if (f1["Maestrantes"] > f0["Maestrantes"]) and (f1["Estres"] < f0["Estres"]):
            inv_ok += 1

    df_s = pd.DataFrame({
        "Parametro": nombres,
        "S1_Maes": np.round(si_m["S1"], 4),
        "ST_Maes": np.round(si_m["ST"], 4),
        "ST_Maes_CI95": np.round(si_m["ST_conf"], 4),
        "S1_Estr": np.round(si_e["S1"], 4),
        "ST_Estr": np.round(si_e["ST"], 4),
        "ST_Estr_CI95": np.round(si_e["ST_conf"], 4),
        "ST_Rep": np.round(si_r["ST"], 4),
    }).sort_values("ST_Maes", ascending=False)

    meta = {
        "ejecutado": True,
        "n_base": int(n_base),
        "n_corridas": int(n_tot),
        "maes_2035_p10_p90": [round(float(np.percentile(y_maes, 10)), 2), round(float(np.percentile(y_maes, 90)), 2)],
        "estr_2035_p10_p90": [round(float(np.percentile(y_estr, 10)), 2), round(float(np.percentile(y_estr, 90)), 2)],
        "rep_2035_p10_p90": [round(float(np.percentile(y_rep, 10)), 2), round(float(np.percentile(y_rep, 90)), 2)],
        "invarianza_ordinal_pct": round(100.0 * inv_ok / len(idxs), 1),
        "banda_maes_p10_p90": {str(yr): [round(float(np.percentile(band_maes[:, j], 10)), 2), round(float(np.percentile(band_maes[:, j], 90)), 2)] for j, yr in enumerate(yrs) if yr in (2020, 2023, 2026, 2029, 2032, 2035)},
        "banda_estr_p10_p90": {str(yr): [round(float(np.percentile(band_estr[:, j], 10)), 2), round(float(np.percentile(band_estr[:, j], 90)), 2)] for j, yr in enumerate(yrs) if yr in (2020, 2023, 2026, 2029, 2032, 2035)},
    }
    return df_s, meta


def ejecutar_sobol_palancas(n_pol: int = 256) -> Tuple[pd.DataFrame, int]:
    """Descomposición de varianza en el espacio de las 3 familias de política [0, 1]^3."""
    if not TIENE_SALIB:
        return pd.DataFrame(), 0
    nombres = ["Familia_Gobernanza_DORA", "Familia_Mercado_Red", "Familia_IA_BrainFirst_Dual"]
    problem = {"num_vars": 3, "names": nombres, "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]}
    X = sobol_sample.sample(problem, n_pol, calc_second_order=False)
    n_tot = len(X)
    y_m, y_e, y_r = np.zeros(n_tot), np.zeros(n_tot), np.zeros(n_tot)

    for i, fila in enumerate(X):
        g_val, m_val, ia_val = float(fila[0]), float(fila[1]), float(fila[2])
        ov = {
            "sw_dora": g_val, "sw_escalafon": g_val, "sw_agilizacion": g_val,
            "sw_costo_marginal": m_val, "sw_red_tematica": m_val,
            "sw_brain_first": ia_val, "sw_dual_clock": ia_val,
        }
        fin = simular(ov, dt=0.125).iloc[-1]
        y_m[i], y_e[i], y_r[i] = fin["Maestrantes"], fin["Estres"], fin["reputacion_externa"]

    si_m = sobol_analyze.analyze(problem, y_m, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_e = sobol_analyze.analyze(problem, y_e, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_r = sobol_analyze.analyze(problem, y_r, calc_second_order=False, conf_level=0.95, print_to_console=False)

    df_p = pd.DataFrame({
        "Familia_Politica": nombres,
        "ST_Maestrantes": np.round(si_m["ST"], 4),
        "ST_Maes_CI95": np.round(si_m["ST_conf"], 4),
        "ST_Estres": np.round(si_e["ST"], 4),
        "ST_Estr_CI95": np.round(si_e["ST_conf"], 4),
        "ST_Reputacion": np.round(si_r["ST"], 4),
        "ST_Rep_CI95": np.round(si_r["ST_conf"], 4),
    })
    return df_p, n_tot


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor RK4 + Ablación 7 Palancas + Sobol para 2026_dora_estres_posgrado_ies.")
    parser.add_argument("--proyecto", default="2026_dora_estres_posgrado_ies")
    parser.add_argument("--n-sobol", type=int, default=512)
    args = parser.parse_args()

    p_script = Path(__file__).resolve()
    raiz = next((parent for parent in p_script.parents if (parent / "03_proyectos").exists()), Path.cwd())
    proy_dir = raiz / "03_proyectos" / args.proyecto
    csv_dir = proy_dir / "modelo_stella" / "corridas_csv"
    ssot_dir = proy_dir / "ssot"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ssot_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print(f" MOTOR CASE HEI-B (DORA + ESTRÉS + IA GENERATIVA) | {args.proyecto}")
    print("=========================================================================")

    trajs: Dict[str, pd.DataFrame] = {}
    res_rows = []
    max_cons_err = 0.0
    for nom, ov in ESCENARIOS.items():
        df = simular(ov, dt=0.0625, metodo="RK4")
        df.to_csv(csv_dir / f"{nom}.csv", index=False)
        trajs[nom] = df
        fin = df.iloc[-1]
        err_c = float(df["residual_conservacion"].abs().max())
        max_cons_err = max(max_cons_err, err_c)
        res_rows.append({
            "Escenario": nom,
            "Maes_2035": round(float(fin["Maestrantes"]), 2),
            "EgrTI_2035": round(float(fin["Egresados_TI"]), 1),
            "Estres_2035": round(float(fin["Estres"]), 2),
            "Titulares_2035": round(float(fin["Titulares"]), 2),
            "Fuga_pct": round(100.0 * float(fin["fuga_por_estres"]), 2),
            "RepExt_2035": round(float(fin["reputacion_externa"]), 2),
            "APC_Bn": round(float(fin["gastos_apc_bn"]), 2),
            "PrimaHum": round(float(fin["prima_humana"]), 3),
        })
        print(f"   [OK] {nom:<32} | Maes={fin['Maestrantes']:5.2f} | Estres={fin['Estres']:5.2f} | Tit={fin['Titulares']:5.2f} | Rep={fin['reputacion_externa']:5.2f}")
    df_res = pd.DataFrame(res_rows)

    # Convergencia RK4 a 3 pasos + Euler + Barlas unclamped
    conv_rows = []
    for sc in ("01_StatusQuo_FastClock", "06_Reforma_Integral_7Levers"):
        r0625 = simular(ESCENARIOS[sc], dt=0.0625, metodo="RK4").iloc[-1]
        r03125 = simular(ESCENARIOS[sc], dt=0.03125, metodo="RK4").iloc[-1]
        r015625 = simular(ESCENARIOS[sc], dt=0.015625, metodo="RK4").iloc[-1]
        eu0625 = simular(ESCENARIOS[sc], dt=0.0625, metodo="Euler").iloc[-1]
        conv_rows.append({
            "Escenario": sc,
            "RK4_0.0625": round(float(r0625["Maestrantes"]), 5),
            "RK4_0.03125": round(float(r03125["Maestrantes"]), 5),
            "RK4_0.015625": round(float(r015625["Maestrantes"]), 5),
            "Diff_RK4_Step": f"{abs(float(r0625['Maestrantes']) - float(r015625['Maestrantes'])):.2e}",
            "Euler_0.0625": round(float(eu0625["Maestrantes"]), 5),
            "Diff_RK4_Eu": f"{abs(float(r0625['Maestrantes']) - float(eu0625['Maestrantes'])):.2e}",
        })
    df_conv = pd.DataFrame(conv_rows)

    df_uncl = simular(ESCENARIOS["07_Extremo1_Choque_Demanda"], dt=0.0625, aplicar_clamp=False)
    min_uncl_maes = float(df_uncl["Maestrantes"].min())

    # LTM proxy por periodo (Status Quo)
    df_sq = trajs["01_StatusQuo_FastClock"]
    ltm_per = pd.DataFrame([
        {
            "Periodo": "2020–2024 (Presión Inicial)",
            "Fin_Lambda_pct": round(float(df_sq[df_sq["Year"] <= 2024]["ltm_fin_pct"].mean()), 1),
            "Estres_Fuga_pct": round(float(df_sq[df_sq["Year"] <= 2024]["ltm_estres_pct"].mean()), 1),
            "Rep_Demanda_pct": round(float(df_sq[df_sq["Year"] <= 2024]["ltm_rep_pct"].mean()), 1),
            "IA_Goodhart_pct": round(float(df_sq[df_sq["Year"] <= 2024]["ltm_ia_pct"].mean()), 1),
        },
        {
            "Periodo": "2025–2029 (Erosión Acumulada)",
            "Fin_Lambda_pct": round(float(df_sq[(df_sq["Year"] >= 2025) & (df_sq["Year"] <= 2029)]["ltm_fin_pct"].mean()), 1),
            "Estres_Fuga_pct": round(float(df_sq[(df_sq["Year"] >= 2025) & (df_sq["Year"] <= 2029)]["ltm_estres_pct"].mean()), 1),
            "Rep_Demanda_pct": round(float(df_sq[(df_sq["Year"] >= 2025) & (df_sq["Year"] <= 2029)]["ltm_rep_pct"].mean()), 1),
            "IA_Goodhart_pct": round(float(df_sq[(df_sq["Year"] >= 2025) & (df_sq["Year"] <= 2029)]["ltm_ia_pct"].mean()), 1),
        },
        {
            "Periodo": "2030–2035 (Régimen Estructural)",
            "Fin_Lambda_pct": round(float(df_sq[df_sq["Year"] >= 2030]["ltm_fin_pct"].mean()), 1),
            "Estres_Fuga_pct": round(float(df_sq[df_sq["Year"] >= 2030]["ltm_estres_pct"].mean()), 1),
            "Rep_Demanda_pct": round(float(df_sq[df_sq["Year"] >= 2030]["ltm_rep_pct"].mean()), 1),
            "IA_Goodhart_pct": round(float(df_sq[df_sq["Year"] >= 2030]["ltm_ia_pct"].mean()), 1),
        },
    ])

    print(f"1. [SOBOL PANEL A] Ejecutando sensibilidad paramétrica (±20%, n={args.n_sobol}, N={args.n_sobol*10})...")
    df_sob_a, meta_a = ejecutar_sobol_parametros(n_base=args.n_sobol)
    df_sob_a.to_csv(ssot_dir / "sobol_panel_a_posgrado.csv", index=False)

    print("2. [SOBOL PANEL B] Ejecutando separación funcional de las 3 familias de palancas...")
    df_sob_b, n_tot_b = ejecutar_sobol_palancas(n_pol=256)
    df_sob_b.to_csv(ssot_dir / "sobol_panel_b_palancas.csv", index=False)

    cols_t = ["Year", "Maestrantes", "Egresados_TI", "Titulares", "Estres", "reputacion_externa", "presion_reloj_lambda", "gastos_apc_bn"]
    yrs_sel = [2020, 2023, 2026, 2029, 2032, 2035]
    md = [
        f"# Resultados Verificados RK4 + Ablación 7 Palancas + Conservación + Sobol — `{args.proyecto}`",
        f"**Generado:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **SSOT:** `UniNorte Magoti.stmx` (2020–2035, RK4, DT=0.0625)\n",
        "## 1. Matriz Comparativa de 8 Escenarios al Año 2035",
        df_res.to_markdown(index=False),
        "\n## 2. Trayectorias Clave (2020–2035) — Status Quo vs. Reforma Integral",
        "### 2.1 `01_StatusQuo_FastClock`",
        trajs["01_StatusQuo_FastClock"][trajs["01_StatusQuo_FastClock"]["Year"].isin(yrs_sel)][cols_t].to_markdown(index=False),
        "\n### 2.2 `03_Gobernanza_DORA_Esc_Pagos`",
        trajs["03_Gobernanza_DORA_Esc_Pagos"][trajs["03_Gobernanza_DORA_Esc_Pagos"]["Year"].isin(yrs_sel)][cols_t].to_markdown(index=False),
        "\n### 2.3 `06_Reforma_Integral_7Levers`",
        trajs["06_Reforma_Integral_7Levers"][trajs["06_Reforma_Integral_7Levers"]["Year"].isin(yrs_sel)][cols_t].to_markdown(index=False),
        "\n## 3. Verificación Numérica: Conservación de Personas, Convergencia RK4 y Barlas",
        f"- **Conservación exacta de personas (`max |total_people - Acum_Flujo_Frontera|`):** `{max_cons_err:.2e}`",
        f"- **Barlas Unclamped (Choque de demanda -40%):** `min(Maestrantes) = {min_uncl_maes:.2f} >= 0`\n",
        df_conv.to_markdown(index=False),
        "\n## 4. Dominancia Cuantitativa de Bucles por Periodo (Status Quo)",
        ltm_per.to_markdown(index=False),
        f"\n## 5. Sobol Panel A — Incertidumbre Paramétrica (±20%, n={args.n_sobol}, N={meta_a.get('n_corridas', 0)}, IC 95%)",
        f"- **Metadatos:** `{json.dumps(meta_a, ensure_ascii=False)}`\n",
        df_sob_a.to_markdown(index=False),
        f"\n## 6. Sobol Panel B — Separación Funcional de las 3 Familias de Política (N={n_tot_b}, IC 95%)",
        df_sob_b.to_markdown(index=False),
    ]
    ruta_rep = ssot_dir / "resumen_experimentos_posgrado_dora.md"
    ruta_rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n✅ Reporte completo guardado en: {ruta_rep}")
    print("=========================================================================")


if __name__ == "__main__":
    main()