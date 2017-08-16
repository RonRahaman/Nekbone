#!/bin/bash
#BSUB -P csc235
#BSUB -J nekbone.mpi.pgi
#BSUB -o nekbone.mpi.pgi.%J.output
#BSUB -e nekbone.mpi.pgi.%J.error
#BSUB -W 120
#BSUB -n 20

module load pgi
module load cuda

cd "$HOME/repos/Nekbone-2017-08-09/test/nek_gpu1"
#mpirun -np 1 ./nekbone data
scripts/bench_nelt.py
