"""Tarea 3 (Opción A): ajuste de la temperatura inicial T0 del Recocido Simulado (2-opt, TSP)
mediante un diseño factorial 3 x 3 (temperatura x tamaño), 10 réplicas por punto = 90 corridas.
"""
import argparse
import json
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from common import (RESULTS, dist_matrix, held_karp, make_schedule, random_points,
                    simulated_annealing, tour_length)

# ---------------------------------------------------- configuración
MATRICULA = 2115459
random.seed(MATRICULA)
R = random.random()                      # se genera UNA sola vez y se usa en todo el experimento
T_MIN, T_MAX = 1.0, 100.0                # rango de temperatura elegido (coordenadas en [0,100]²)
NIVELES = ["bajo", "medio", "alto"]
FRACCION = {"bajo": 0.3 * R, "medio": 0.3 * R + 0.3, "alto": 0.3 * R + 0.6}
TEMP = {k: T_MIN + f * (T_MAX - T_MIN) for k, f in FRACCION.items()}
TAMANOS = [5, 10, 15]
REPLICAS = 10
T_FINAL = 0.01                           # temperatura final fija (esquema exponencial)
ALFA, B_BOOT = 0.05, 10_000
N_PRINCIPAL = 300                        # presupuesto de iteraciones del experimento principal
N_SENSIBILIDAD = [100, 300, 1000]        # presupuestos para el análisis de sensibilidad


def preparar_instancias():
    """Una instancia TSP por tamaño (semilla = matrícula + n) y su óptimo exacto (Held-Karp)."""
    D = {n: dist_matrix(random_points(n, random.Random(MATRICULA + n))) for n in TAMANOS}
    return D, {n: held_karp(D[n]) for n in TAMANOS}


def correr_experimento(n_iter, Ds, optimos):
    filas = []
    for n in TAMANOS:
        D = Ds[n]
        for j, nivel in enumerate(NIVELES):
            T0 = TEMP[nivel]
            for rep in range(REPLICAS):
                rng = random.Random(MATRICULA * 1000 + n * 100 + j * 10 + rep)   # semilla distinta por corrida
                ini = list(range(n)); rng.shuffle(ini)
                _, tour_mejor, *_ = simulated_annealing(D, ini, make_schedule("exponencial", T0, T_FINAL, n_iter),
                                                        n_iter, rng, paso=n_iter)
                # se recalcula la longitud del mejor tour (evita error acumulado de punto flotante)
                mejor = round(tour_length(D, tour_mejor), 6)
                filas.append(dict(tamano=n, nivel=nivel, temperatura=round(T0, 3), rep=rep + 1, costo=mejor,
                                  gap=100 * (mejor / optimos[n] - 1)))
    return pd.DataFrame(filas)


def bootstrap_mediana(x, rng, B=B_BOOT):
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(np.median(meds)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def anova2(df, y):
    modelo = ols(f"{y} ~ C(nivel) * C(tamano)", data=df).fit()
    return sm.stats.anova_lm(modelo, typ=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=N_PRINCIPAL, help="iteraciones del recocido por corrida")
    a = ap.parse_args()
    N_ITER = a.iter

    print(f"r = {R:.6f} (semilla = matrícula {MATRICULA})")
    for k in NIVELES:
        print(f"  nivel {k:5s}: fracción {FRACCION[k]:.4f} -> T0 = {TEMP[k]:.2f}")
    Ds, optimos = preparar_instancias()
    df = correr_experimento(N_ITER, Ds, optimos)
    df.to_csv(RESULTS / "doe_datos.csv", index=False)
    print(f"\n{len(df)} corridas. Óptimos exactos (Held-Karp): { {n: round(v, 2) for n, v in optimos.items()} }")

    # --------------------------------------------------------- ANOVA de dos factores
    tab = anova2(df, "costo")
    tab_gap = anova2(df, "gap")
    print("\nANOVA de dos factores (variable: costo)\n", tab.round(4))
    print("\nANOVA de dos factores (variable: % sobre el óptimo, comprobación de robustez)\n", tab_gap.round(4))
    tab.to_csv(RESULTS / "doe_anova_costo.csv"); tab_gap.to_csv(RESULTS / "doe_anova_gap.csv")

    # --------------------------------- análisis por tamaño + bootstrap + Tukey
    rng = np.random.default_rng(MATRICULA)
    resumen = {"r": R, "T_MIN": T_MIN, "T_MAX": T_MAX, "n_iter": N_ITER, "niveles": {k: dict(fraccion=FRACCION[k], T0=TEMP[k]) for k in NIVELES},
               "optimos": optimos, "anova_costo": tab.reset_index().to_dict("records"),
               "anova_gap": tab_gap.reset_index().to_dict("records"), "por_tamano": {}}
    boot_filas = []
    for n in TAMANOS:
        d = df[df.tamano == n]
        # ANOVA de un factor (temperatura) dentro del tamaño
        if d.costo.nunique() == 1:
            p_n, F_n = float("nan"), float("nan")
        else:
            t1 = sm.stats.anova_lm(ols("costo ~ C(nivel)", data=d).fit(), typ=2)
            p_n, F_n = float(t1.loc["C(nivel)", "PR(>F)"]), float(t1.loc["C(nivel)", "F"])
        signif = (p_n == p_n) and p_n < ALFA
        stats = {}
        for nivel in NIVELES:
            x = d[d.nivel == nivel].costo.values
            med, lo, hi = bootstrap_mediana(x, rng)
            stats[nivel] = dict(T0=TEMP[nivel], media=float(x.mean()), desv=float(x.std(ddof=1)), mediana_boot=med, ic_inf=lo, ic_sup=hi,
                                minimo=float(x.min()), pct_optimo=float((x <= optimos[n] + 1e-6).mean() * 100))
            boot_filas.append(dict(tamano=n, nivel=nivel, T0=TEMP[nivel], mediana_boot=med, ic_inf=lo, ic_sup=hi))
        mejor = min(NIVELES, key=lambda k: (round(stats[k]["mediana_boot"], 9), stats[k]["media"]))
        tukey = None
        if signif:
            tk = pairwise_tukeyhsd(d.costo.values, d.nivel.values, alpha=ALFA)
            tukey = [dict(g1=r[0], g2=r[1], diff=float(r[2]), p=float(r[3]), rechaza=bool(r[6])) for r in tk._results_table.data[1:]]
            print(f"\nTukey HSD, n = {n}\n{tk}")
        resumen["por_tamano"][str(n)] = dict(F=F_n, p=p_n, significativo=bool(signif), mejor=mejor, T_mejor=TEMP[mejor], stats=stats, tukey=tukey)
        print(f"\nn={n}: ANOVA de 1 factor F={F_n:.3f} p={p_n:.4f} -> {'significativo' if signif else ('sin variación: todas las corridas dieron el mismo costo' if p_n != p_n else 'NO significativo')}; "
              f"mejor nivel = {mejor} (T0={TEMP[mejor]:.1f})")
        for nivel in NIVELES:
            s = stats[nivel]
            print(f"   {nivel:5s} T0={s['T0']:6.1f}  media {s['media']:8.2f} ± {s['desv']:6.2f}  mediana boot {s['mediana_boot']:8.2f}  IC95 [{s['ic_inf']:.2f}, {s['ic_sup']:.2f}]  óptimas {s['pct_optimo']:.0f}%")
    pd.DataFrame(boot_filas).to_csv(RESULTS / "doe_bootstrap.csv", index=False)
    json.dump(resumen, open(RESULTS / "doe_resumen.json", "w"), indent=2, ensure_ascii=False, default=float)

    # ------------------------------------------------------ gráficas
    ICt = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    cols = {5: "tab:blue", 10: "tab:orange", 15: "tab:green"}
    etiqT = [f"{k}\n(T0={TEMP[k]:.0f})" for k in NIVELES]

    # 1) efectos principales
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    m = [df[df.nivel == k].costo for k in NIVELES]
    ax[0].errorbar(range(3), [x.mean() for x in m], yerr=[ICt(x) for x in m], fmt="o-", capsize=4)
    ax[0].set(xticks=range(3), xticklabels=etiqT, xlabel="Temperatura inicial T0", ylabel="Costo medio (promediando tamaños)",
              title="Efecto principal: temperatura")
    m = [df[df.tamano == n].costo for n in TAMANOS]
    ax[1].errorbar(range(3), [x.mean() for x in m], yerr=[ICt(x) for x in m], fmt="s-", color="tab:red", capsize=4)
    ax[1].set(xticks=range(3), xticklabels=[f"n = {n}" for n in TAMANOS], xlabel="Tamaño del problema", ylabel="Costo medio (promediando temperaturas)",
              title="Efecto principal: tamaño")
    for x in ax: x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "doe_efectos_principales.png", dpi=130); plt.close(fig)

    # 2) interacción temperatura x tamaño (costo y % sobre óptimo)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in TAMANOS:
        d = df[df.tamano == n]
        ax[0].errorbar(range(3), [d[d.nivel == k].costo.mean() for k in NIVELES], yerr=[ICt(d[d.nivel == k].costo) for k in NIVELES],
                       fmt="o-", color=cols[n], capsize=3, label=f"n = {n}")
        ax[1].errorbar(range(3), [d[d.nivel == k].gap.mean() for k in NIVELES], yerr=[ICt(d[d.nivel == k].gap) for k in NIVELES],
                       fmt="o-", color=cols[n], capsize=3, label=f"n = {n}")
    ax[0].set(xticks=range(3), xticklabels=etiqT, xlabel="Temperatura inicial T0", ylabel="Costo medio", title="Interacción T0 × tamaño (costo)")
    ax[1].set(xticks=range(3), xticklabels=etiqT, xlabel="Temperatura inicial T0", ylabel="% sobre el óptimo (medio)", title="Interacción T0 × tamaño (% sobre óptimo)")
    for x in ax: x.grid(alpha=.3)
    ax[0].legend(loc="center right"); ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(RESULTS / "doe_interaccion.png", dpi=130); plt.close(fig)

    # 3) cajas por tamaño con mediana bootstrap e IC
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for a_, n in zip(ax, TAMANOS):
        d = df[df.tamano == n]
        a_.boxplot([d[d.nivel == k].costo.values for k in NIVELES], tick_labels=etiqT, widths=.5)
        for i, k in enumerate(NIVELES):
            s = resumen["por_tamano"][str(n)]["stats"][k]
            a_.errorbar(i + 1, s["mediana_boot"], yerr=[[s["mediana_boot"] - s["ic_inf"]], [s["ic_sup"] - s["mediana_boot"]]],
                        fmt="D", color="crimson", capsize=5, ms=5, label="mediana bootstrap ± IC95" if i == 0 else None)
        a_.axhline(optimos[n], color="g", ls=":", label="óptimo exacto")
        a_.set(title=f"n = {n}  (p = {resumen['por_tamano'][str(n)]['p']:.3g})" if resumen['por_tamano'][str(n)]['p'] == resumen['por_tamano'][str(n)]['p'] else f"n = {n}  (sin variación)",
               xlabel="Temperatura inicial T0", ylabel="Costo del mejor tour")
        a_.grid(alpha=.3); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "doe_boxplots.png", dpi=130); plt.close(fig)

    # ------------------------------ sensibilidad al presupuesto de iteraciones
    print("\nSensibilidad al número de iteraciones (ANOVA de dos factores sobre el costo)")
    filas = []
    for it in N_SENSIBILIDAD:
        d_it = df if it == N_ITER else correr_experimento(it, Ds, optimos)
        t_it = anova2(d_it, "costo")
        fila = dict(iteraciones=it, p_temperatura=t_it.loc["C(nivel)", "PR(>F)"], p_interaccion=t_it.loc["C(nivel):C(tamano)", "PR(>F)"])
        for n in TAMANOS:
            dn = d_it[d_it.tamano == n]
            if dn.costo.nunique() == 1:
                fila[f"p_n{n}"], fila[f"mejor_n{n}"] = float("nan"), "empate"
            else:
                fila[f"p_n{n}"] = sm.stats.anova_lm(ols("costo ~ C(nivel)", data=dn).fit(), typ=2).loc["C(nivel)", "PR(>F)"]
                fila[f"mejor_n{n}"] = min(NIVELES, key=lambda k: (round(float(np.median(dn[dn.nivel == k].costo)), 6), dn[dn.nivel == k].costo.mean()))
            fila[f"pct_opt_n{n}"] = float((dn.costo <= optimos[n] + 1e-6).mean() * 100)
        filas.append(fila)
    sens = pd.DataFrame(filas)
    print(sens.round(4).to_string(index=False))
    sens.to_csv(RESULTS / "doe_sensibilidad.csv", index=False)
