export CUDA_VISIBLE_DEVICES=1


# /shared/eng/pj20/lmr_model/ctgov_3b_transfer_1/actor/global_step_100
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/SearchEngine/eval_ctgov.py \
    --model_path DeepRetrieval/DeepRetrieval-CTGov-3B \
    --data_path data/search_engine/ctgov_32/test_full.parquet \
    --model_name ctgov_search_llama