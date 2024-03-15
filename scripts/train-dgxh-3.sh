export OUTPUT_DIR=outputs/r101_freeze_enc+dec_street_work
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
--lr_drop 500 --epochs 1000 --eval_interval 40 \
--hf_model facebook/detr-resnet-101 \
--freeze_layers model.encoder model.decoder \
--coco_path data/street_work --num_workers 16 --batch_size 16 \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log

export OUTPUT_DIR=outputs/r101_lora_enc+dec_backbone_train_street_work
mkdir -p $OUTPUT_DIR
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
--lr_drop 500 --epochs 1000 --eval_interval 40 \
--hf_model facebook/detr-resnet-101 \
--lora --modules_to_save class_labels_classifier bbox_predictor model.input_projection model.backbone \
--lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
--coco_path data/street_work --num_workers 16 --batch_size 12 \
--output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/train.log