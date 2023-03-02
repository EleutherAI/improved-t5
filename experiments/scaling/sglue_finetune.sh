#!/bin/bash
#SBATCH --job-name="improved-t5"
#SBATCH --partition="gpu"
#SBATCH --open-mode=append
#SBATCH --exclude=gpu-st-p4d-24xlarge-377
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6           # Number of cores per tasks
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=/fsx/aran/jax/slurm_outputs/%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=/fsx/aran/jax/slurm_outputs/%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --comment=neox
#SBATCH --exclusive
#SBATCH --requeue

echo "Name of the file is $0"

source /fsx/lintangsutawika/2-scripts/env.sh
source /fsx/aran/jax/jax/bin/activate

# Cache Directories
export BASE_DIR="/fsx/aran/jax/t5x_2/"
export PROJECT_DIR=${BASE_DIR}"architecture-objective/t5x"
# export MODEL_DIR="gs://t5x-test/ckpts/alibi/regular-t5-base-c4-nodes-"${SLURM_NNODES}"-procs-"${SLURM_NPROCS}"-bs-2048/"

#export TFDS_DATA_DIR="/fsx/c4/c4-en"
#export TFDS_DATA_DIR="/fsx/improved-t5/super_glue"
export CACHED_DATA_DIR="/fsx/lintangsutawika/data"

# directory where the T5X repo is cloned.
export T5X_DIR=${BASE_DIR}"architecture-objective"
export PYTHONPATH=${PROJECT_DIR}

# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
# export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1" # Hacky and don't want
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 # Helps an NCCL out of memory issue

export ADDR="$(hostname -f):29500"

export CONFIG_PATH="${T5X_DIR}/experiments/configs"

srun --comment neox \
    python ${T5X_DIR}/t5x/train.py \
        --gin_file="${CONFIG_PATH}/finetune_sglue.gin" \
        --gin_file="${CONFIG_PATH}/size/200m/vanilla.gin" \
        --gin_file="${CONFIG_PATH}/mode/gpu.gin" \
        --gin.TRAIN_STEPS=640_000 \
        --gin.MODEL_DIR=\"'/fsx/aran/jax/ckpts/batch_size/200m/256k_finetune'\" \
        --gin.INITIAL_CHECKPOINT_PATH=\"'/fsx/aran/jax/ckpts/batch_size/200m/256k/checkpoint_512000'\" \
        --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
        --alsologtostderr \
        --multiprocess_gpu \
        --coordinator_address="${ADDR}" \
        --process_count ${SLURM_NPROCS} \
        --process_num_device -1