export CUDA_VISIBLE_DEVICES=1


# MODEL_PATH=/dev/ana/sft_training_outputs/bird_wo_reasoning/checkpoint-6342
# MODEL_PATH=/dev/ana/sft_training_outputs/bird_wo_reasoning/checkpoint-8456
# MODEL_PATH=/dev/ana/sft_training_outputs/bird_wo_reasoning/checkpoint-10570
# MODEL_PATH=/dev/ana/DeepRetrieval-SQL/cold_start/bird_Qwen/Qwen2.5-Coder-3B-Instruct/checkpoint-8456
# MODEL_PATH=/dev/ana/DeepRetrieval-SQL/cold_start/bird_Qwen/Qwen2.5-Coder-3B-Instruct_wo_reasoning/checkpoint-4228/
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct
# MODEL_PATH=/dev/ana/qwen-7

# MODEL_PATH=/dev/ana/sft_qwen_7/bird/checkpoint-8452
MODEL_PATH=/dev/ana/sft_qwen_7/bird_wo_reasoning/checkpoint-4226

python src/eval/SQL/bird.py \
    --model_path $MODEL_PATH \
    --data_path data/sql/bird/test.parquet \
    --model_name bird-3b-sft-full \
    --save_dir ../results \
    --with_reasoning false \
    --batch_size 8