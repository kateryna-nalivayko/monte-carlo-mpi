"""Generate speedup and efficiency plots comparing serial, MPI, and OpenMP modes."""

import json
import sys


def main():
    import matplotlib.pyplot as plt

    try:
        with open("benchmark_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: benchmark_results.json not found.")
        print("Run 'uv run python scripts/run_all.py' first.")
        sys.exit(1)

    methods = sorted(set(r["method"] for r in results))
    modes = ["mpi", "openmp"]
    mode_labels = {"mpi": "MPI", "openmp": "OpenMP (numba)"}
    mode_colors = {"mpi": "#2196F3", "openmp": "#FF9800"}
    mode_markers = {"mpi": "o", "openmp": "s"}

    fig, axes = plt.subplots(len(methods), 2, figsize=(14, 5 * len(methods)))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)

    for row, method in enumerate(methods):
        # Speedup plot
        ax = axes[row, 0]
        max_workers = 1

        for mode in modes:
            data = sorted(
                [r for r in results
                 if r["method"] == method and r["mode"] == mode and r.get("speedup")],
                key=lambda r: r["workers"],
            )
            if not data:
                continue
            workers = [r["workers"] for r in data]
            speedups = [r["speedup"] for r in data]
            ax.plot(
                workers, speedups,
                f"{mode_markers[mode]}-",
                color=mode_colors[mode],
                label=mode_labels[mode],
                linewidth=2, markersize=8,
            )
            max_workers = max(max_workers, max(workers))

        # Serial baseline point
        serial = next(
            (r for r in results if r["method"] == method and r["mode"] == "serial"),
            None,
        )
        if serial:
            ax.plot(1, 1, "D", color="#4CAF50", markersize=10, label="Serial", zorder=5)

        ideal = list(range(1, max_workers + 1))
        ax.plot(ideal, ideal, "k--", alpha=0.3, label="Ideal")
        ax.set_xlabel("Number of Workers / Threads")
        ax.set_ylabel("Speedup")
        ax.set_title(f"{method} — Speedup")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Efficiency plot
        ax = axes[row, 1]
        for mode in modes:
            data = sorted(
                [r for r in results
                 if r["method"] == method and r["mode"] == mode and r.get("efficiency")],
                key=lambda r: r["workers"],
            )
            if not data:
                continue
            workers = [r["workers"] for r in data]
            effs = [r["efficiency"] for r in data]
            ax.plot(
                workers, effs,
                f"{mode_markers[mode]}-",
                color=mode_colors[mode],
                label=mode_labels[mode],
                linewidth=2, markersize=8,
            )

        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="Ideal")
        ax.set_xlabel("Number of Workers / Threads")
        ax.set_ylabel("Efficiency")
        ax.set_title(f"{method} — Efficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("speedup_plot.png", dpi=150)
    print("Saved: speedup_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
