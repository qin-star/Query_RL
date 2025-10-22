export CUDA_VISIBLE_DEVICES=0

# --model_path /shared/eng/pj20/lmr_model/squad_3b_nq_1800/actor/global_step_200 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
#  mistralai/Mistral-7B-Instruct-v0.3
python src/eval/BM25/squad.py \
    --model_path /shared/eng/pj20/lmr_model/squad_3b_no_reason/actor/global_step_450 \
    --model_name squad_ours_no_reason\
    --data_path data/local_index_search/squad/test.parquet
