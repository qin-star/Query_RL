export CUDA_VISIBLE_DEVICES=0

# MODEL_PATH=/dev/ana/sft_training_outputs/spider/checkpoint-3135
# MODEL_PATH=/dev/ana/sft_training_outputs/spider/checkpoint-4180
# MODEL_PATH=/dev/ana/sft_training_outputs/spider/checkpoint-5225
# MODEL_PATH=/dev/ana/sft_training_outputs/spider/checkpoint-6270


# MODEL_PATH=/dev/ana/sft_training_outputs/spider_wo_reasoning/checkpoint-4180
# MODEL_PATH=/dev/ana/sft_training_outputs/spider_wo_reasoning/checkpoint-6270
# MODEL_PATH=/dev/ana/sft_training_outputs/spider_wo_reasoning/checkpoint-8360
# MODEL_PATH=/dev/ana/sft_training_outputs/spider_wo_reasoning/checkpoint-10450
# MODEL_PATH=/dev/ana/DeepRetrieval-SQL/cold_start/spider_Qwen/Qwen2.5-Coder-3B-Instruct_wo_reasoning/checkpoint-4180
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct

# MODEL_PATH=/dev/ana/qwen-7
# MODEL_PATH=/dev/ana/sft_qwen_7/spider_wo_reasoning/checkpoint-2088
# MODEL_PATH=/dev/ana/sft_qwen_7/spider/checkpoint-4176
MODEL_PATH=DeepRetrieval/DeepRetrieval-SQL-7B

python src/eval/SQL/spider.py \
    --model_path $MODEL_PATH \
    --data_path data/sql/spider/val.parquet \
    --model_name spider-7b-full \
    --save_dir ../results \
    --with_reasoning true \
    --batch_size 8