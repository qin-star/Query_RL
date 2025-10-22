export CUDA_VISIBLE_DEVICES=0

# --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/msmarco.py \
    --model_path /shared/eng/pj20/lmr_model/msmarco_health_sparse_recall/actor/global_step_650 \
    --model_name msmarco_health_sparse_recall_step_650 \
    --data_path data/local_index_search/msmarco_health/sparse/test.parquet