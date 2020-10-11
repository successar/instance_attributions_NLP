#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40Gb
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/jain.sar/runs/%x.%j.out
#SBATCH --error=/scratch/jain.sar/runs/%x.%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1

export CUDA_DEVICE=0

$1 $2