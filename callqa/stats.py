"""Статистика для обучения правил.

Точный тест Фишера и поправка на множественные сравнения — без них
перебор тысяч фраз гарантированно выдаёт «значимые» совпадения там,
где их нет. Реализовано вручную, чтобы не тянуть scipy в зависимости.
"""
from __future__ import annotations

from math import exp, lgamma
from typing import List, Sequence, Tuple


def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _hypergeom_pmf(k: int, N: int, K: int, n: int) -> float:
    """Вероятность ровно k успехов при выборке n из N, где успехов всего K."""
    num = _log_binom(K, k) + _log_binom(N - K, n - k)
    den = _log_binom(N, n)
    if num == float("-inf") or den == float("-inf"):
        return 0.0
    return exp(num - den)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Двусторонний точный тест Фишера для таблицы 2x2.

        a — «хороших» звонков с признаком      b — «плохих» с признаком
        c — «хороших» без признака             d — «плохих» без признака

    Возвращает p-значение: вероятность увидеть такую же или ещё более
    выраженную связь, если на самом деле связи нет.
    """
    a, b, c, d = int(a), int(b), int(c), int(d)
    if min(a, b, c, d) < 0:
        raise ValueError("Отрицательные значения в таблице")
    N = a + b + c + d
    if N == 0:
        return 1.0
    K = a + c          # всего «хороших»
    n = a + b          # всего с признаком
    if n == 0 or n == N or K == 0 or K == N:
        return 1.0     # признак или метка не варьируются — сравнивать нечего

    observed = _hypergeom_pmf(a, N, K, n)
    tol = observed * (1 + 1e-9)
    total = 0.0
    for k in range(max(0, n - (N - K)), min(n, K) + 1):
        p = _hypergeom_pmf(k, N, K, n)
        if p <= tol:
            total += p
    return min(1.0, max(0.0, total))


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Поправка на множественные сравнения (контроль доли ложных открытий).

    Перебирая тысячи фраз, при пороге p<0.05 получишь примерно 50 «значимых»
    находок на каждую тысячу пустых. Поправка приводит цифру к смыслу
    «какая доля отобранного окажется мусором».
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        value = min(prev, pvalues[i] * m / rank)
        q[i] = min(1.0, max(0.0, value))
        prev = q[i]
    return q


def auc(good: Sequence[float], bad: Sequence[float]) -> float:
    """Площадь под ROC: вероятность, что случайный «хороший» звонок
    получит большее значение метрики, чем случайный «плохой».
    0.5 — метрика не различает исходы вовсе."""
    if not good or not bad:
        return 0.5
    values = [(v, 1) for v in good] + [(v, 0) for v in bad]
    values.sort(key=lambda x: x[0])

    # средние ранги для одинаковых значений, иначе связки смещают оценку
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1

    rank_sum = sum(r for r, (_v, lab) in zip(ranks, values) if lab == 1)
    n1, n0 = len(good), len(bad)
    return max(0.0, min(1.0, (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)))


def best_threshold(good: Sequence[float], bad: Sequence[float]
                   ) -> Tuple[float, str, float]:
    """Ищет порог, который лучше всего разделяет два набора.

    Возвращает (порог, оператор сравнения, сбалансированная точность).
    Оператор — «<=» если у хороших звонков значения ниже, иначе «>=».
    """
    if not good or not bad:
        return 0.0, "<=", 0.5
    values = sorted(set(list(good) + list(bad)))
    if len(values) < 2:
        return values[0], "<=", 0.5

    cuts = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    best = (values[0], "<=", 0.0)
    for cut in cuts:
        for op in ("<=", ">="):
            if op == "<=":
                tp = sum(1 for v in good if v <= cut)
                tn = sum(1 for v in bad if v > cut)
            else:
                tp = sum(1 for v in good if v >= cut)
                tn = sum(1 for v in bad if v < cut)
            score = (tp / len(good) + tn / len(bad)) / 2
            if score > best[2]:
                best = (cut, op, score)
    return best


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96
                    ) -> Tuple[float, float]:
    """Доверительный интервал для доли. На малых выборках честнее обычного:
    показывает, насколько шаткая оценка стоит за красивым процентом."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)
