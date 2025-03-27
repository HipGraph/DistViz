#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --output=%j.log


export OMP_NUM_THREADS=32

module load intel
srun -n 4 --cpu-bind=cores ../build/bin/distviz -input ./data/cora/cora.txt -output ./output -data-set-size 60000  -dimension 784  -ntrees 32  -nn 10  -locality 0 -data-file-format 0 -tree-depth-ratio 0.8 -generate-knng-output 1 -dropout-error-th 0.013  -lr 0.25 -sparse-input 0 -nsamples 5 -iterations 1200 
