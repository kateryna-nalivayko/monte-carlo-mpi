#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI_REF      3.141592653589793
#define INTEG_REF   1.2748050317
#define TWO_PI      6.283185307179586

#define MC_N        10000000
#define MC_WALKS    10000
#define MC_STEPS    1000

static inline double rand_d(unsigned int *s) {
    return (double)rand_r(s) / ((double)RAND_MAX + 1.0);
}

static double integrand(double x) {
    return sin(x) * exp(-x * x / 10.0);
}

static double pi_sample(void *ctx, unsigned int *seed) {
    (void)ctx;
    double x = rand_d(seed), y = rand_d(seed);
    return (x*x + y*y < 1.0) ? 4.0 : 0.0;
}

static double integrate_sample(void *ctx, unsigned int *seed) {
    double a = ((double *)ctx)[0], b = ((double *)ctx)[1];
    double x = a + (b - a) * rand_d(seed);
    return (b - a) * integrand(x);
}

static double walk_sample(void *ctx, unsigned int *seed) {
    int steps = *(int *)ctx;
    double x = 0.0, y = 0.0;
    for (int s = 0; s < steps; s++) {
        double angle = TWO_PI * rand_d(seed);
        x += cos(angle); y += sin(angle);
    }
    return x*x + y*y;
}

static double monteCarlo(double (*sample)(void *, unsigned int *),
                          int n, unsigned int seed, void *ctx) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += sample(ctx, &seed);
    return sum / n;
}

static double mc_pi_mpi(int N, int rank, int size) {
    int local_n = N / size + (rank == 0 ? N % size : 0);
    unsigned int seed = 42 + rank * 1000;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double local_est = monteCarlo(pi_sample, local_n, seed, NULL);
    double local_wt  = local_est * local_n;

    printf("  [Rank %d] samples=%d, hits=%ld\n", rank, local_n, lround(local_wt / 4.0));

    double global_wt = 0.0;
    MPI_Reduce(&local_wt, &global_wt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        double pi = global_wt / N;
        printf("\n--- Pi (MPI, %d processes) ---\n", size);
        printf("  Samples:    %d\n", N);
        printf("  Result:     %.10f\n", pi);
        printf("  Reference:  %.10f\n", PI_REF);
        printf("  Error:      %.2e\n",  fabs(pi - PI_REF));
        printf("  Time:       %.4f s\n", elapsed);
        return elapsed;
    }
    return 0.0;
}

static double mc_integrate_mpi(int N, int rank, int size, double a, double b) {
    int local_n = N / size + (rank == 0 ? N % size : 0);

    double params[2] = {a, b};
    MPI_Bcast(params, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    unsigned int seed = 42 + rank * 1000;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double local_est = monteCarlo(integrate_sample, local_n, seed, params);
    double local_wt  = local_est * local_n;

    printf("  [Rank %d] samples=%d, local_sum=%.4f\n", rank, local_n,
           local_wt / (params[1] - params[0]));

    double global_wt = 0.0;
    MPI_Reduce(&local_wt, &global_wt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        double result = global_wt / N;
        printf("\n--- Integration (MPI, %d processes) ---\n", size);
        printf("  Samples:    %d\n", N);
        printf("  Result:     %.10f\n", result);
        printf("  Reference:  %.10f\n", INTEG_REF);
        printf("  Error:      %.2e\n",  fabs(result - INTEG_REF));
        printf("  Time:       %.4f s\n", elapsed);
        return elapsed;
    }
    return 0.0;
}

static double mc_walk_mpi(int total_walks, int rank, int size, int steps) {
    int local_walks = total_walks / size + (rank == 0 ? total_walks % size : 0);
    unsigned int seed = 42 + rank * 1000;

    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double local_msd = monteCarlo(walk_sample, local_walks, seed, &steps);
    double local_wt  = local_msd * local_walks;

    printf("  [Rank %d] walks=%d, MSD=%.2f\n", rank, local_walks, local_msd);

    double global_wt    = 0.0;
    int    global_walks = 0;
    MPI_Reduce(&local_wt,    &global_wt,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_walks, &global_walks, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        double final_msd = global_wt / global_walks;
        printf("\n--- Random Walk (MPI, %d processes) ---\n", size);
        printf("  Walks:      %d\n", global_walks);
        printf("  Steps:      %d\n", steps);
        printf("  Final MSD:  %.4f\n", final_msd);
        printf("  Expected:   %.4f\n", (double)steps);
        printf("  Error:      %.4f%%\n", fabs(final_msd - steps) / steps * 100.0);
        printf("  Time:       %.4f s\n", elapsed);
        return elapsed;
    }
    return 0.0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *csv_out = (argc > 1) ? argv[1] : NULL;

    if (rank == 0)
        printf("паралельна реалізація з MPI (%d процесів)\n", size);

    double t_pi   = mc_pi_mpi(MC_N, rank, size);
    double t_int  = mc_integrate_mpi(MC_N, rank, size, 0.0, TWO_PI);
    double t_walk = mc_walk_mpi(MC_WALKS, rank, size, MC_STEPS);

    if (rank == 0 && csv_out) {
        FILE *f = fopen(csv_out, "a");
        if (f) {
            fprintf(f, "%d,%.6f,%.6f,%.6f\n", size, t_pi, t_int, t_walk);
            fclose(f);
        }
    }

    MPI_Finalize();
    return 0;
}
