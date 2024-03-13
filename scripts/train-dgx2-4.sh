export OUTPUT_DIR=outputs/lora_default_street_work
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--lr_drop 1000 --epochs 2000 --eval_interval 40 \
--lora --target_modules q_proj v_proj \
--weight detr-r50-e632da11.pth \
--coco_path data/street_work --num_workers 6 --batch_size 8 \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log

# export OUTPUT_DIR=outputs/lora_mask
# mkdir -p $OUTPUT_DIR
# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
# --epochs 25 --lr_drop 15 --coco_path /scratch/open_coco --masks  \
# --coco_panoptic_path /scratch/open_coco  --dataset_file coco_panoptic \
# --frozen_weights detr-r50-e632da11.pth \
# --num_workers 6 --batch_size 3 --mha_type lora \
# --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log
