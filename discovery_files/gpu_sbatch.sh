#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40Gb
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/jain.sar/runs/%x.%j.out
#SBATCH --error=/scratch/jain.sar/runs/%x.%j.out
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1

export CUDA_DEVICE=0
export DATADIR=/scratch/jain.sar/influence_info/Datasets
export OUTPUT_DIR=/scratch/jain.sar/influence_info/outputs

$1 $2