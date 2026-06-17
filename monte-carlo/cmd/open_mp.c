#include <omp.h>
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

static double mc_pi_omp(int N, int n_threads) {
    omp_set_num_threads(n_threads);
    double t0 = omp_get_wtime();

    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    {
        int tid     = omp_get_thread_num();
        int nthr    = omp_get_num_threads();
        int local_n = N / nthr + (tid == 0 ? N % nthr : 0);
        unsigned int seed = 42 + tid * 1000;
        total += monteCarlo(pi_sample, local_n, seed, NULL) * local_n;
    }

    double pi      = total / N;
    double elapsed = omp_get_wtime() - t0;

    printf("\n--- Pi (OpenMP, %d threads) ---\n", n_threads);
    printf("  Samples:    %d\n", N);
    printf("  Result:     %.10f\n", pi);
    printf("  Reference:  %.10f\n", PI_REF);
    printf("  Error:      %.2e\n",  fabs(pi - PI_REF));
    printf("  Time:       %.4f s\n", elapsed);
    return elapsed;
}

static double mc_integrate_omp(int N, double a, double b, int n_threads) {
    omp_set_num_threads(n_threads);
    double params[2] = {a, b};
    double t0 = omp_get_wtime();

    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    {
        int tid     = omp_get_thread_num();
        int nthr    = omp_get_num_threads();
        int local_n = N / nthr + (tid == 0 ? N % nthr : 0);
        unsigned int seed = 42 + tid * 1000;
        total += monteCarlo(integrate_sample, local_n, seed, params) * local_n;
    }

    double result  = total / N;
    double elapsed = omp_get_wtime() - t0;

    printf("\n--- Integration (OpenMP, %d threads) ---\n", n_threads);
    printf("  Samples:    %d\n", N);
    printf("  Result:     %.10f\n", result);
    printf("  Reference:  %.10f\n", INTEG_REF);
    printf("  Error:      %.2e\n",  fabs(result - INTEG_REF));
    printf("  Time:       %.4f s\n", elapsed);
    return elapsed;
}

static double mc_walk_omp(int total_walks, int steps, int n_threads) {
    omp_set_num_threads(n_threads);
    double t0 = omp_get_wtime();

    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    {
        int tid         = omp_get_thread_num();
        int nthr        = omp_get_num_threads();
        int local_walks = total_walks / nthr + (tid == 0 ? total_walks % nthr : 0);
        unsigned int seed = 42 + tid * 1000;
        total += monteCarlo(walk_sample, local_walks, seed, &steps) * local_walks;
    }

    double final_msd = total / total_walks;
    double elapsed   = omp_get_wtime() - t0;

    printf("\n--- Random Walk (OpenMP, %d threads) ---\n", n_threads);
    printf("  Walks:      %d\n", total_walks);
    printf("  Steps:      %d\n", steps);
    printf("  Final MSD:  %.4f\n", final_msd);
    printf("  Expected:   %.4f\n", (double)steps);
    printf("  Error:      %.4f%%\n", fabs(final_msd - steps) / steps * 100.0);
    printf("  Time:       %.4f s\n", elapsed);
    return elapsed;
}

int main(int argc, char *argv[]) {
    const char *csv_out = (argc > 1) ? argv[1] : NULL;
    int n_threads = omp_get_max_threads();

    printf("ПАРАЛЕЛЬНА РЕАЛІЗАЦІЯ З OPENMP (%d потоків)\n", n_threads);

    double t_pi   = mc_pi_omp(MC_N, n_threads);
    double t_int  = mc_integrate_omp(MC_N, 0.0, TWO_PI, n_threads);
    double t_walk = mc_walk_omp(MC_WALKS, MC_STEPS, n_threads);

    if (csv_out) {
        FILE *f = fopen(csv_out, "a");
        if (f) {
            fprintf(f, "%d,%.6f,%.6f,%.6f\n", n_threads, t_pi, t_int, t_walk);
            fclose(f);
        }
    }
    return 0;
}
