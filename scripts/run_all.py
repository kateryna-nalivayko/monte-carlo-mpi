"""
Порівняння швидкості трьох реалізацій:
  1. Serial (послідовна)
  2. MPI (паралельна, процеси)
  3. OpenMP (паралельна, потоки через numba)

Запуск: uv run python scripts/run_all.py
"""

import re
import subprocess
import sys

N = 10_000_000
MPI_PROCS = 4
OMP_THREADS = 4


def parse_times(output):
    """Витягнути час для кожної задачі з виводу програми."""
    times = {}
    current = None
    for line in output.splitlines():
        if "Pi" in line:
            current = "Pi"
        elif "Integration" in line:
            current = "Integration"
        elif "Random Walk" in line:
            current = "Random Walk"
        match = re.search(r"Time:\s+([\d.]+)", line)
        if match and current:
            times[current] = float(match.group(1))
            current = None
    return times


def run_cmd(cmd, label):
    """Запустити команду та повернути час."""
    print(f"\n  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        times = parse_times(result.stdout)
        if times:
            for task, t in times.items():
                print(f"    {task}: {t:.4f}s")
        else:
            print(f"    (no times parsed)")
            if result.stderr:
                print(f"    stderr: {result.stderr[:200]}")
        return times
    except FileNotFoundError:
        print(f"    ERROR: command not found")
        return {}
    except subprocess.TimeoutExpired:
        print(f"    ERROR: timeout")
        return {}


def main():
    print("=" * 60)
    print(f"  BENCHMARK: {N:,} samples, MPI={MPI_PROCS} procs, OMP={OMP_THREADS} threads")
    print("=" * 60)

    # 1. Serial
    print("\n[1] Serial")
    serial = run_cmd(["uv", "run", "mc-serial", "-n", str(N)], "serial")

    # 2. MPI
    print(f"\n[2] MPI ({MPI_PROCS} processes)")
    mpi = run_cmd(
        ["mpiexec", "-n", str(MPI_PROCS), "uv", "run", "mc-mpi", "-n", str(N)],
        "mpi",
    )

    # 3. OpenMP
    print(f"\n[3] OpenMP ({OMP_THREADS} threads)")
    openmp = run_cmd(
        ["uv", "run", "mc-openmp", "-t", str(OMP_THREADS), "-n", str(N)],
        "openmp",
    )

    # Таблиця порівняння
    tasks = ["Pi", "Integration", "Random Walk"]

    print(f"\n{'=' * 60}")
    print(f"  ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    print(f"{'=' * 60}")
    print(f"{'Task':<16} {'Serial':>10} {'MPI':>10} {'OpenMP':>10} {'MPI sp.':>10} {'OMP sp.':>10}")
    print("-" * 66)

    for task in tasks:
        t_s = serial.get(task)
        t_m = mpi.get(task)
        t_o = openmp.get(task)

        s_str = f"{t_s:.4f}s" if t_s else "N/A"
        m_str = f"{t_m:.4f}s" if t_m else "N/A"
        o_str = f"{t_o:.4f}s" if t_o else "N/A"

        sp_mpi = f"{t_s / t_m:.2f}x" if t_s and t_m else "N/A"
        sp_omp = f"{t_s / t_o:.2f}x" if t_s and t_o else "N/A"

        print(f"{task:<16} {s_str:>10} {m_str:>10} {o_str:>10} {sp_mpi:>10} {sp_omp:>10}")

    print()


if __name__ == "__main__":
    main()
