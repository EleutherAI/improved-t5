mkdir -p $2

python convert_t5v2_checkpoint_to_pytorch.py \
    --config_file configs/base_v2/config.json \
    --t5x_checkpoint_path $1 \
    --pytorch_dump_path $2 \
    --scalable_attention

cp configs/base_v2/* $2