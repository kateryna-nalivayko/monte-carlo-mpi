#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long N = (argc > 1) ? std::atoll(argv[1]) : 100'000'000LL;
    const double    h = 1.0 / N;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double local_sum = 0.0;
    for (long long i = rank; i < N; i += size) {
        double x = (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        double pi = h * global_sum;
        printf("==================================================\n");
        printf("  MPI Pi (прямокутники, %d процесів)\n", size);
        printf("==================================================\n");
        printf("  Intervals:  %lld\n", N);
        printf("  Result:     %.10f\n", pi);
        printf("  Reference:  %.10f\n", M_PI);
        printf("  Error:      %.2e\n", std::abs(pi - M_PI));
        printf("  Time:       %.4f s\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
