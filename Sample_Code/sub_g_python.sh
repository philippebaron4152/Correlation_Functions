#!/bin/bash
#SBATCH --job-name=jax-gpu       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=pb4152@princeton.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2024.10
conda activate jax-gpu

python compute_g.py -p ${PWD} -n 216 -nb 120 -r 0.9 -s 3.542 -sol 1. -solv 2.