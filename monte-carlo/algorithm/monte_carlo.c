#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define PI_REF      3.141592653589793
#define INTEG_REF   1.2748050317
#define TWO_PI      6.283185307179586

#define MC_N        10000000
#define MC_WALKS    10000
#define MC_STEPS    1000

static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

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

static double mc_pi(int N, unsigned int seed) {
    double t0      = wall_time();
    double pi      = monteCarlo(pi_sample, N, seed, NULL);
    double elapsed = wall_time() - t0;

    printf("\n--- Pi (serial) ---\n");
    printf("  Samples:    %d\n", N);
    printf("  Result:     %.10f\n", pi);
    printf("  Reference:  %.10f\n", PI_REF);
    printf("  Error:      %.2e\n",  fabs(pi - PI_REF));
    printf("  Time:       %.4f s\n", elapsed);
    return elapsed;
}

static double mc_integrate(int N, double a, double b, unsigned int seed) {
    double ctx[2]  = {a, b};
    double t0      = wall_time();
    double result  = monteCarlo(integrate_sample, N, seed, ctx);
    double elapsed = wall_time() - t0;

    printf("\n--- Integration (serial) ---\n");
    printf("  Samples:    %d\n", N);
    printf("  Result:     %.10f\n", result);
    printf("  Reference:  %.10f\n", INTEG_REF);
    printf("  Error:      %.2e\n",  fabs(result - INTEG_REF));
    printf("  Time:       %.4f s\n", elapsed);
    return elapsed;
}

static double mc_walk(int total_walks, int steps, unsigned int seed) {
    double t0        = wall_time();
    double final_msd = monteCarlo(walk_sample, total_walks, seed, &steps);
    double elapsed   = wall_time() - t0;

    printf("\n--- Random Walk (serial) ---\n");
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

    printf("ПОСЛІДОВНА (SERIAL) РЕАЛІЗАЦІЯ\n");

    double t_pi   = mc_pi(MC_N, 42);
    double t_int  = mc_integrate(MC_N, 0.0, TWO_PI, 42);
    double t_walk = mc_walk(MC_WALKS, MC_STEPS, 42);

    if (csv_out) {
        FILE *f = fopen(csv_out, "a");
        if (f) {
            fprintf(f, "1,%.6f,%.6f,%.6f\n", t_pi, t_int, t_walk);
            fclose(f);
        }
    }
    return 0;
}
