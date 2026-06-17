CC    = clang
MPICC = mpicc

ALG_DIR = algorithm
CMD_DIR = cmd
BIN_DIR = bin

# OpenMP через libomp з Homebrew
OMP_FLAGS = -Xpreprocessor -fopenmp \
            -I/opt/homebrew/opt/libomp/include \
            -L/opt/homebrew/opt/libomp/lib -lomp

SERIAL_BIN = $(BIN_DIR)/mc_serial
MPI_BIN    = $(BIN_DIR)/mc_mpi
OMP_BIN    = $(BIN_DIR)/mc_omp

$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p results)

.PHONY: all clean serial mpi omp run_serial run_mpi run_omp bench help

all: serial mpi omp

serial: $(ALG_DIR)/monte_carlo.c
	$(CC) $< -o $(SERIAL_BIN) -lm

mpi: $(CMD_DIR)/mpi.c
	$(MPICC) $(CMD_DIR)/mpi.c -o $(MPI_BIN) -lm

omp: $(CMD_DIR)/open_mp.c
	$(CC) $(OMP_FLAGS) $(CMD_DIR)/open_mp.c -o $(OMP_BIN) -lm

run_serial: serial
	./$(SERIAL_BIN)

run_mpi: mpi
	mpirun -np 4 ./$(MPI_BIN)

run_omp: omp
	./$(OMP_BIN)

bench: all
	bash scripts/bench_serial.sh
	bash scripts/bench_mpi.sh
	bash scripts/bench_omp.sh
	python3 scripts/plot.py

clean:
	rm -f $(BIN_DIR)/*
	rm -f results/*.csv results/*.png

help:
	@echo "Цілі:"
	@echo "  make all        — зібрати всі три версії"
	@echo "  make serial     — зібрати послідовну версію (algorithm/)"
	@echo "  make mpi        — зібрати MPI версію (cmd/)"
	@echo "  make omp        — зібрати OpenMP версію (cmd/)"
	@echo "  make run_serial — запустити послідовну"
	@echo "  make run_mpi    — запустити MPI з 4 процесами"
	@echo "  make run_omp    — запустити OpenMP"
	@echo "  make bench      — бенчмарки + графіки"
	@echo "  make clean      — видалити бінарники та результати"
