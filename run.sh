python train.py --PLM klue/roberta-large --overwrite_output_dir True --save_total_limit 1 --output_dir ./exp --do_train --do_eval --learning_rate 2e-5 --num_train_epochs 6 --per_device_train_batch_size 16 --gradient_accumulation_steps 2 --logging_dir ./logs --save_strategy steps --evaluation_strategy steps --logging_steps 20 --save_steps 109 --eval_steps 109 --weight_decay 1e-4 --warmup_ratio 0.0 --load_best_model_at_end --metric_for_best_model 'f1' --use_Smart_loss 

# 원하는 checkpoint 경로로 변경
python inference.py --use_Smart_loss --multiple_weight checkpoints