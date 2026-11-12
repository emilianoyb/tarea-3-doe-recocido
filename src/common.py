"""Utilidades compartidas: instancias TSP, 2-opt, vecino más cercano,
MST (Prim), Held-Karp, recocido simulado y mochila."""
import itertools
import math
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------- TSP básico
def random_points(n, rng, size=100.0):
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def dist_matrix(pts):
    return [[math.dist(a, b) for b in pts] for a in pts]


def tour_length(D, tour):
    return sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def nearest_neighbour(D, start=0):
    n = len(D)
    tour, libres = [start], set(range(n)) - {start}
    while libres:
        nxt = min(libres, key=lambda j: D[tour[-1]][j])
        tour.append(nxt)
        libres.remove(nxt)
    return tour


# ------------------------------------------------------------- 2-opt
def two_opt(D, tour):
    """2-opt (primera mejora, subir colinas). Devuelve (tour, nº de mejoras)."""
    tour, n, mejoras = tour[:], len(tour), 0
    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % n]
                if D[a][c] + D[b][d] - D[a][b] - D[c][d] < -1e-9:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    mejoras += 1
                    mejoro = True
    return tour, mejoras

# ------------------------------------------------ MST + preorden
def mst_prim(D):
    """Prim desde la ciudad 0. Devuelve {nodo: hijos}."""
    n = len(D)
    best = {v: (D[0][v], 0) for v in range(1, n)}
    hijos = {v: [] for v in range(n)}
    while best:
        v = min(best, key=lambda u: best[u][0])
        hijos[best.pop(v)[1]].append(v)
        for u in best:
            if D[v][u] < best[u][0]:
                best[u] = (D[v][u], v)
    return hijos


def mst_weight(D, hijos):
    return sum(D[p][c] for p, cs in hijos.items() for c in cs)


def preorder(hijos, raiz=0):
    return [raiz] + [c for h in hijos[raiz] for c in preorder(hijos, h)]


def held_karp(D):
    """Longitud óptima del TSP (exacto, sólo n pequeño)."""
    n = len(D)
    INF = float("inf")
    dp = {(1, 0): 0.0}
    for mask in range(1, 1 << n):
        if not mask & 1:
            continue
        for last in range(n):
            if (mask, last) not in dp:
                continue
            for nxt in range(1, n):
                if mask & (1 << nxt):
                    continue
                key = (mask | (1 << nxt), nxt)
                val = dp[(mask, last)] + D[last][nxt]
                if val < dp.get(key, INF):
                    dp[key] = val
    full = (1 << n) - 1
    return min(dp[(full, j)] + D[j][0] for j in range(1, n))


# ----------------------------------------- Recocido simulado
def make_schedule(nombre, T0, Tf, N):
    """Esquemas de enfriamiento T(t), t = 1..N. Todos van de ~T0 a ~Tf
    (salvo 'exponencial rápido', de tasa fija)."""
    if nombre == "lineal":
        return lambda t: max(T0 * (1 - t / N), Tf)
    if nombre == "exponencial":
        return lambda t: T0 * (Tf / T0) ** (t / N)
    if nombre == "logarítmico":
        alpha = (T0 / Tf - 1) / math.log(1 + N)
        return lambda t: T0 / (1 + alpha * math.log(1 + t))
    if nombre == "exponencial rápido":
        return lambda t: max(T0 * 0.995 ** t, Tf)
    raise ValueError(nombre)


def simulated_annealing(D, tour, schedule, n_iter, rng, paso=100):
    """Recocido simulado con movimiento 2-opt aleatorio.
    Devuelve (mejor_costo, mejor_tour, its, costo_actual, mejor_hasta_ahora, T)."""
    tour = tour[:]
    n = len(tour)
    L = tour_length(D, tour)
    best, best_tour = L, tour[:]
    its, cur_h, best_h, T_h = [], [], [], []
    for t in range(1, n_iter + 1):
        T = schedule(t)
        i, k = sorted(rng.sample(range(1, n), 2))
        a, b, c, d = tour[i - 1], tour[i], tour[k], tour[(k + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:k + 1] = tour[i:k + 1][::-1]
            L += delta
            if L < best - 1e-12:
                best, best_tour = L, tour[:]
        if t % paso == 1 or t == n_iter:
            its.append(t); cur_h.append(L); best_h.append(best); T_h.append(T)
    return best, best_tour, its, cur_h, best_h, T_h


# ------------------------------------------------- Mochila
def greedy_knapsack(values, weights, W, key):
    orden = sorted(range(len(values)), key=key, reverse=True)
    tv = tw = 0
    elegidos = []
    for i in orden:
        if tw + weights[i] <= W:
            elegidos.append(i); tv += values[i]; tw += weights[i]
    return sorted(elegidos), tv, tw


def knapsack_dp(values, weights, W):
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])
    elegidos, c = [], W
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            elegidos.append(i - 1); c -= weights[i - 1]
    elegidos = sorted(elegidos)
    return elegidos, K[n][W], sum(weights[i] for i in elegidos)
