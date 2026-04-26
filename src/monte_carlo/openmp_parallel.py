"""
Метод Монте-Карло — паралельна реалізація з OpenMP (через numba).

Запуск: uv run python -m monte_carlo.openmp_parallel

numba + prange — аналог #pragma omp parallel for із C/C++.
Цикл prange автоматично розподіляється між потоками ОС.
Numba компілює Python-код в машинний код через LLVM.
"""

import time
import numpy as np
import numba
from numba import njit, prange


# ---- 1. Оцінка числа Pi ----

@njit(parallel=True)
def _pi_kernel(n):
    """Ядро обчислення Pi — паралельний цикл.

    prange розподіляє ітерації між потоками.
    Numba автоматично робить reduction для змінної count.
    """
    count = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y < 1.0:
            count += 1
    return count


def estimate_pi(total_samples, n_threads=4, seed=42):
    """Оцінка Pi методом Монте-Карло з OpenMP (numba prange)."""
    numba.set_num_threads(n_threads)

    # Прогрів JIT-компіляції (не враховується в часі)
    _pi_kernel(100)

    start = time.perf_counter()
    hits = _pi_kernel(total_samples)
    elapsed = time.perf_counter() - start

    pi_estimate = 4.0 * hits / total_samples

    print(f"\n--- Pi (OpenMP, {n_threads} threads) ---")
    print(f"  Samples:    {total_samples:,}")
    print(f"  Result:     {pi_estimate:.10f}")
    print(f"  Reference:  {np.pi:.10f}")
    print(f"  Error:      {abs(pi_estimate - np.pi):.2e}")
    print(f"  Time:       {elapsed:.4f} s")

    return pi_estimate, elapsed


# ---- 2. Чисельне інтегрування ----

@njit(parallel=True)
def _integrate_kernel(n, a, b):
    """Ядро інтегрування — паралельний цикл.

    f(x) = sin(x) * exp(-x^2 / 10)
    Numba автоматично робить reduction для total.
    """
    total = 0.0
    for i in prange(n):
        x = np.random.random() * (b - a) + a
        total += np.sin(x) * np.exp(-(x ** 2) / 10.0)
    return total


def integrate(total_samples, a=0.0, b=2 * np.pi, n_threads=4, seed=42):
    """Інтеграл f(x) = sin(x) * exp(-x^2/10) з OpenMP (numba prange)."""
    numba.set_num_threads(n_threads)

    # Прогрів JIT
    _integrate_kernel(100, a, b)

    start = time.perf_counter()
    total_sum = _integrate_kernel(total_samples, a, b)
    elapsed = time.perf_counter() - start

    result = (b - a) * total_sum / total_samples
    reference = 1.2748050317

    print(f"\n--- Integration (OpenMP, {n_threads} threads) ---")
    print(f"  Samples:    {total_samples:,}")
    print(f"  Result:     {result:.10f}")
    print(f"  Reference:  {reference:.10f}")
    print(f"  Error:      {abs(result - reference):.2e}")
    print(f"  Time:       {elapsed:.4f} s")

    return result, elapsed


# ---- 3. Випадкове блукання (random walk) ----

@njit(parallel=True)
def _walk_kernel(n_walks, steps):
    """Ядро випадкового блукання — паралельний цикл по блуканнях.

    prange розподіляє блукання між потоками.
    Кожне блукання незалежне — embarrassingly parallel.
    """
    msd_sum = np.zeros(steps, dtype=np.float64)

    for w in prange(n_walks):
        x = 0.0
        y = 0.0
        for s in range(steps):
            angle = np.random.random() * 2.0 * np.pi
            x += np.cos(angle)
            y += np.sin(angle)
            msd_sum[s] += x * x + y * y

    return msd_sum


def random_walk(total_walks, steps=1000, n_threads=4, seed=42):
    """2D випадкове блукання з OpenMP (numba prange)."""
    numba.set_num_threads(n_threads)

    # Прогрів JIT
    _walk_kernel(10, 10)

    start = time.perf_counter()
    msd_sum = _walk_kernel(total_walks, steps)
    elapsed = time.perf_counter() - start

    msd = msd_sum / total_walks

    print(f"\n--- Random Walk (OpenMP, {n_threads} threads) ---")
    print(f"  Walks:      {total_walks:,}")
    print(f"  Steps:      {steps:,}")
    print(f"  Final MSD:  {msd[-1]:.4f}")
    print(f"  Expected:   {float(steps):.4f}")
    print(f"  Error:      {abs(msd[-1] - steps) / steps * 100:.4f}%")
    print(f"  Time:       {elapsed:.4f} s")

    return msd, elapsed


# ---- Запуск усіх задач ----

if __name__ == "__main__":
    N = 10_000_000
    THREADS = 4

    print("=" * 50)
    print(f"  ПАРАЛЕЛЬНА РЕАЛІЗАЦІЯ З OPENMP ({THREADS} потоків)")
    print("=" * 50)

    estimate_pi(N, n_threads=THREADS)
    integrate(N, n_threads=THREADS)
    random_walk(10_000, steps=1000, n_threads=THREADS)
