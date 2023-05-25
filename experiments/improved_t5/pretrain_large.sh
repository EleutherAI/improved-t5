ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="configs/t5v2/large.gin" \
    --gin_file="configs/task/pretrain/pile_mixed_objective.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""/fsx/lintangsutawika/improved_t5/tokenizer.model"\" \
    --gin.MIXTURE_OR_TASK_NAME=\""pile_ul2_causal_0_50"\" \
    --gin.TASK_FEATURE_LENGTHS="{'inputs':2048, 'targets': 2048}" \
    --gin.BATCH_SIZE=512 \
    --gin.TRAIN_STEPS=1000000 \
    --gin.SAVING_PERIOD=1000 \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/t5v2/large/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}