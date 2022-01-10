CXX=g++
MPICXX=mpicxx
CXX_FLAGS = -std=c++14 -O3 -Wall -DNDEBUG
BIN=bin
SRC=c++

dir_guard=@mkdir -p $(BIN)

all: mcpi-seq mcpi-par mcpi-mpi

mcpi-seq: $(SRC)/mcpi-seq.cpp
	$(dir_guard)
	@$(CXX) -o $(BIN)/$@ $^ $(CXX_FLAGS)

mcpi-par: $(SRC)/mcpi-par.cpp
	$(dir_guard)
	@$(CXX) -fopenmp -o $(BIN)/$@ $^ $(CXX_FLAGS)

mcpi-mpi: $(SRC)/mcpi-mpi.cpp
	$(dir_guard)
	@$(MPICXX) -o $(BIN)/$@ $^ $(CXX_FLAGS)

.PHONY: clean

clean:
	rm -f $(BIN)/*