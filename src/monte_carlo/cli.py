"""CLI entry points for Monte Carlo experiments."""

import argparse
import os


def run_serial():
    """Запустити всі задачі послідовно."""
    from monte_carlo.serial import estimate_pi, integrate, random_walk

    parser = argparse.ArgumentParser(description="Monte Carlo — Serial")
    parser.add_argument("-n", "--samples", type=int, default=10_000_000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 50)
    print("  ПОСЛІДОВНА (SERIAL) РЕАЛІЗАЦІЯ")
    print("=" * 50)

    estimate_pi(args.samples, seed=args.seed)
    integrate(args.samples, seed=args.seed)
    random_walk(10_000, steps=1000, seed=args.seed)


def run_mpi():
    """Запустити всі задачі з MPI."""
    from mpi4py import MPI
    from monte_carlo.mpi_parallel import estimate_pi, integrate, random_walk

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Monte Carlo — MPI")
    parser.add_argument("-n", "--samples", type=int, default=10_000_000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    if rank == 0:
        print("=" * 50)
        print(f"  ПАРАЛЕЛЬНА РЕАЛІЗАЦІЯ З MPI ({size} процесів)")
        print("=" * 50)

    estimate_pi(args.samples, seed=args.seed)
    integrate(args.samples, seed=args.seed)
    random_walk(10_000, steps=1000, seed=args.seed)


def run_openmp():
    """Запустити всі задачі з OpenMP."""
    from monte_carlo.openmp_parallel import estimate_pi, integrate, random_walk

    parser = argparse.ArgumentParser(description="Monte Carlo — OpenMP")
    parser.add_argument("-n", "--samples", type=int, default=10_000_000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    print("=" * 50)
    print(f"  ПАРАЛЕЛЬНА РЕАЛІЗАЦІЯ З OPENMP ({args.threads} потоків)")
    print("=" * 50)

    estimate_pi(args.samples, n_threads=args.threads, seed=args.seed)
    integrate(args.samples, n_threads=args.threads, seed=args.seed)
    random_walk(10_000, steps=1000, n_threads=args.threads, seed=args.seed)
