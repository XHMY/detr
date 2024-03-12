export OUTPUT_DIR=outputs/ori_baseline
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
--lr_drop 100 --epochs 150 \
--coco_path /tmp/zengyif/coco --num_workers 15 --batch_size 16 \
--mha_type ori \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log