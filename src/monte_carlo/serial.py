"""
Метод Монте-Карло — послідовна (serial) реалізація.

Три задачі:
  1. Оцінка числа Pi
  2. Чисельне інтегрування
  3. 2D випадкове блукання (random walk)
"""

import time
import numpy as np


# ---- 1. Оцінка числа Pi ----

def estimate_pi(total_samples, seed=42):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    x = rng.random(total_samples)
    y = rng.random(total_samples)
    hits = np.sum(x * x + y * y < 1.0)
    pi_estimate = 4.0 * hits / total_samples

    elapsed = time.perf_counter() - start

    print(f"\n--- Pi (serial) ---")
    print(f"  Samples:    {total_samples:,}")
    print(f"  Result:     {pi_estimate:.10f}")
    print(f"  Reference:  {np.pi:.10f}")
    print(f"  Error:      {abs(pi_estimate - np.pi):.2e}")
    print(f"  Time:       {elapsed:.4f} s")

    return pi_estimate, elapsed


# ---- 2. Чисельне інтегрування ----

def integrate(total_samples, a=0.0, b=2 * np.pi, seed=42):
    """Інтеграл f(x) = sin(x) * exp(-x^2/10) на [a, b] методом Монте-Карло.

    Інтеграл ≈ (b - a) * середнє(f(x_i)), де x_i — випадкові точки в [a, b].
    """
    rng = np.random.default_rng(seed)

    start = time.perf_counter()

    x = rng.uniform(a, b, size=total_samples)
    fx = np.sin(x) * np.exp(-(x ** 2) / 10.0)
    result = (b - a) * np.mean(fx)

    elapsed = time.perf_counter() - start

    reference = 1.2748050317  # обчислено через np.trapezoid

    print(f"\n--- Integration (serial) ---")
    print(f"  Samples:    {total_samples:,}")
    print(f"  Result:     {result:.10f}")
    print(f"  Reference:  {reference:.10f}")
    print(f"  Error:      {abs(result - reference):.2e}")
    print(f"  Time:       {elapsed:.4f} s")

    return result, elapsed


# ---- 3. Випадкове блукання (random walk) ----

def random_walk(total_walks, steps=1000, seed=42):
    """2D випадкове блукання — обчислення середнього квадрата зміщення (MSD).

    Кожен крок — одиничний вектор у випадковому напрямку.
    Теоретичне MSD після N кроків = N.
    """
    rng = np.random.default_rng(seed)

    start = time.perf_counter()

    angles = rng.uniform(0, 2 * np.pi, size=(total_walks, steps))
    dx = np.cos(angles)
    dy = np.sin(angles)

    x = np.cumsum(dx, axis=1)
    y = np.cumsum(dy, axis=1)

    msd = np.mean(x ** 2 + y ** 2, axis=0)

    elapsed = time.perf_counter() - start

    print(f"\n--- Random Walk (serial) ---")
    print(f"  Walks:      {total_walks:,}")
    print(f"  Steps:      {steps:,}")
    print(f"  Final MSD:  {msd[-1]:.4f}")
    print(f"  Expected:   {float(steps):.4f}")
    print(f"  Error:      {abs(msd[-1] - steps) / steps * 100:.4f}%")
    print(f"  Time:       {elapsed:.4f} s")

    return msd, elapsed



if __name__ == "__main__":
    N = 10_000_000

    print("=" * 50)
    print("  ПОСЛІДОВНА (SERIAL) РЕАЛІЗАЦІЯ")
    print("=" * 50)

    estimate_pi(N)
    integrate(N)
    random_walk(10_000, steps=1000)
