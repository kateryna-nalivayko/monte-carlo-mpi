"""
Метод Монте-Карло — паралельна реалізація з MPI.

Запуск: mpiexec -n 4 uv run python -m monte_carlo.mpi_parallel

Використані MPI операції:
  - MPI_Reduce (SUM): збір часткових результатів
  - MPI_Bcast: розсилка параметрів
  - MPI_Gather: збір масивів
"""

import numpy as np
from mpi4py import MPI


def make_rng(rank, seed=42):
    """Створити генератор випадкових чисел для кожного процесу.

    SeedSequence.spawn() гарантує незалежні потоки для кожного rank.
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(rank + 1)
    return np.random.default_rng(child_seeds[rank])


# ---- 1. Оцінка числа Pi ----

def estimate_pi(total_samples, seed=42):
    """Оцінка Pi методом Монте-Карло з MPI.

    Кожен процес генерує свою частину точок.
    MPI_Reduce збирає суму влучень на rank 0.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Розділити роботу між процесами
    local_n = total_samples // size
    if rank == 0:
        local_n += total_samples % size

    rng = make_rng(rank, seed)

    # Синхронізація перед вимірюванням часу
    comm.Barrier()
    start = MPI.Wtime()

    # Кожен процес генерує свої точки
    x = rng.random(local_n)
    y = rng.random(local_n)
    local_hits = int(np.sum(x * x + y * y < 1.0))

    # MPI_Reduce: зібрати суму влучень на rank 0
    global_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)
    global_n = comm.reduce(local_n, op=MPI.SUM, root=0)

    comm.Barrier()
    elapsed = MPI.Wtime() - start

    print(f"  [Rank {rank}] samples={local_n:,}, hits={local_hits:,}")

    pi_estimate = 0.0
    if rank == 0:
        pi_estimate = 4.0 * global_hits / global_n

        print(f"\n--- Pi (MPI, {size} processes) ---")
        print(f"  Samples:    {global_n:,}")
        print(f"  Result:     {pi_estimate:.10f}")
        print(f"  Reference:  {np.pi:.10f}")
        print(f"  Error:      {abs(pi_estimate - np.pi):.2e}")
        print(f"  Time:       {elapsed:.4f} s")

    return pi_estimate, elapsed


# ---- 2. Чисельне інтегрування ----

def integrate(total_samples, a=0.0, b=2 * np.pi, seed=42):
    """Інтеграл f(x) = sin(x) * exp(-x^2/10) на [a, b] з MPI.

    MPI_Bcast розсилає межі інтегрування.
    MPI_Reduce збирає суми значень функції.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_n = total_samples // size
    if rank == 0:
        local_n += total_samples % size

    # MPI_Bcast: розіслати межі інтегрування всім процесам
    params = comm.bcast({"a": a, "b": b}, root=0)
    a, b = params["a"], params["b"]

    rng = make_rng(rank, seed)

    comm.Barrier()
    start = MPI.Wtime()

    # Кожен процес обчислює свою частину
    x = rng.uniform(a, b, size=local_n)
    fx = np.sin(x) * np.exp(-(x ** 2) / 10.0)
    local_sum = float(np.sum(fx))

    # MPI_Reduce: зібрати суму значень функції на rank 0
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    global_n = comm.reduce(local_n, op=MPI.SUM, root=0)

    comm.Barrier()
    elapsed = MPI.Wtime() - start

    print(f"  [Rank {rank}] samples={local_n:,}, local_sum={local_sum:.4f}")

    result = 0.0
    if rank == 0:
        result = (b - a) * global_sum / global_n
        reference = 1.2748050317

        print(f"\n--- Integration (MPI, {size} processes) ---")
        print(f"  Samples:    {global_n:,}")
        print(f"  Result:     {result:.10f}")
        print(f"  Reference:  {reference:.10f}")
        print(f"  Error:      {abs(result - reference):.2e}")
        print(f"  Time:       {elapsed:.4f} s")

    return result, elapsed


# ---- 3. Випадкове блукання (random walk) ----

def random_walk(total_walks, steps=1000, seed=42):
    """2D випадкове блукання з MPI.

    MPI_Bcast розсилає кількість кроків.
    MPI_Reduce (buffer-based) збирає MSD масиви.
    MPI_Gather збирає фінальні позиції.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_walks = total_walks // size
    if rank == 0:
        local_walks += total_walks % size

    # MPI_Bcast: розіслати кількість кроків
    steps = comm.bcast(steps, root=0)

    rng = make_rng(rank, seed)

    comm.Barrier()
    start = MPI.Wtime()

    # Кожен процес симулює свою частину блукань
    angles = rng.uniform(0, 2 * np.pi, size=(local_walks, steps))
    dx = np.cos(angles)
    dy = np.sin(angles)

    x = np.cumsum(dx, axis=1)
    y = np.cumsum(dy, axis=1)

    local_msd_sum = np.sum(x ** 2 + y ** 2, axis=0)

    # MPI_Reduce (buffer-based): зібрати MSD масиви поелементно
    global_msd_sum = np.zeros(steps, dtype=np.float64) if rank == 0 else None
    comm.Reduce(
        np.ascontiguousarray(local_msd_sum, dtype=np.float64),
        global_msd_sum,
        op=MPI.SUM,
        root=0,
    )

    # MPI_Gather: зібрати фінальні позиції на rank 0
    comm.gather(x[:, -1], root=0)
    comm.gather(y[:, -1], root=0)

    global_walks = comm.reduce(local_walks, op=MPI.SUM, root=0)

    comm.Barrier()
    elapsed = MPI.Wtime() - start

    local_msd = float(np.mean(x[:, -1] ** 2 + y[:, -1] ** 2))
    print(f"  [Rank {rank}] walks={local_walks:,}, MSD={local_msd:.2f}")

    if rank == 0:
        msd = global_msd_sum / global_walks

        print(f"\n--- Random Walk (MPI, {size} processes) ---")
        print(f"  Walks:      {global_walks:,}")
        print(f"  Steps:      {steps:,}")
        print(f"  Final MSD:  {msd[-1]:.4f}")
        print(f"  Expected:   {float(steps):.4f}")
        print(f"  Error:      {abs(msd[-1] - steps) / steps * 100:.4f}%")
        print(f"  Time:       {elapsed:.4f} s")

        return msd, elapsed

    return None, elapsed


# ---- Запуск усіх задач ----

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = 10_000_000

    if rank == 0:
        print("=" * 50)
        print(f"  ПАРАЛЕЛЬНА РЕАЛІЗАЦІЯ З MPI ({size} процесів)")
        print("=" * 50)

    estimate_pi(N)
    integrate(N)
    random_walk(10_000, steps=1000)
