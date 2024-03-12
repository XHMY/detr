export OUTPUT_DIR=outputs/pure_lora_baseline
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--lr_drop 100 --epochs 150 \
--coco_path /scratch/open_coco --num_workers 5 --batch_size 8 \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log