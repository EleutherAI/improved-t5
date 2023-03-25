#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=pile-t5x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=12
#SBATCH --output=/fsx/lintangsutawika/improved_t5/logs/%x_%j.out
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --account=neox

source /fsx/lintangsutawika/t5_env/bin/activate

srun --account neox \
    seqio_cache_tasks \
	--tasks="anli_must_be_true_r1" \
	--output_cache_dir=/fsx/lintangsutawika/data \
	--module_import=t0.seqio_tasks \
	--alsologtostderr
