# export OUTPUT_DIR=outputs/lora_conv-plus_street_work
# mkdir -p $OUTPUT_DIR
# python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
# --lr_drop 1000 --epochs 2000 --eval_interval 40 \
# --lora --target_modules q_proj v_proj conv1 conv2 conv3 \
# --coco_path data/street_work --num_workers 16 --batch_size 16 \
# --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log


export OUTPUT_DIR=outputs/hf_bl_street_work
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
--lr_drop 500 --epochs 1500 --eval_interval 40 \
--coco_path data/street_work --num_workers 16 --batch_size 16 \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log

# export OUTPUT_DIR=outputs/ori_mask
# mkdir -p $OUTPUT_DIR
# python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
# --epochs 25 --lr_drop 15 --coco_path /tmp/zengyif/coco --masks  \
# --coco_panoptic_path /tmp/zengyif/coco  --dataset_file coco_panoptic \
# --frozen_weights detr-r50-e632da11.pth \
# --num_workers 16 --batch_size 6 --mha_type ori \
# --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log