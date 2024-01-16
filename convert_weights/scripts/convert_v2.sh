mkdir -p $3

python convert_t5v2_checkpoint_to_pytorch.py \
    --config_file configs/${1}_v2/config.json \
    --t5x_checkpoint_path $2 \
    --pytorch_dump_path $3 \
    --scalable_attention

cp configs/${1}_v2/* $3