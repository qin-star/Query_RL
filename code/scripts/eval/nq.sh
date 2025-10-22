export CUDA_VISIBLE_DEVICES=6

# /shared/eng/pj20/lmr_model/nq_serini_3b_continue/actor/global_step_1150 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/nq.py \
    --model_path /shared/eng/pj20/lmr_model/nq_serini_3b_no_reason/actor/global_step_700 \
    --model_name nq_serini_ours_no_reason \
    --data_path data/local_index_search/nq_serini/test.parquet