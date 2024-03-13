export OUTPUT_DIR=outputs/freeze_pretrain_street_work
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--mha_type ori --lr_drop 1000 --epochs 2000 --eval_interval 40 \
--weight detr-r50-e632da11.pth --train_cls_layeronly \
--coco_path data/street_work --num_workers 5 --batch_size 8 \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log