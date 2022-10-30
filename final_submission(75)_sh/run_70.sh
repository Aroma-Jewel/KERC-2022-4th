python train.py --PLM klue/roberta-large --save_total_limit 2 --do_train --do_eval --learning_rate 2e-5 --num_train_epochs 6 --per_device_train_batch_size 16 --gradient_accumulation_steps 2 --output_dir ./exp --logging_dir ./logs --save_strategy steps --evaluation_strategy steps --logging_steps 50 --save_steps 100 --eval_steps 100 --weight_decay 1e-4 --metric_for_best_model f1 --load_best_model_at_end --use_substitute_preprocess --preprocess_version v5 --past_sentence 5 --use_Smart_loss


python inference.py --use_Smart_loss --multiple_weight checkpoints