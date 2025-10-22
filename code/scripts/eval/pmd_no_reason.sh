export CUDA_VISIBLE_DEVICES=6

python src/eval/SearchEngine/eval_pubmed.py \
    --model_path /shared/eng/pj20/lmr_model/pubmed_search_3b_no_reason/actor/global_step_900 \
    --data_path data/search_engine/no_reason/SearchEngine/pmd_no_reason/test_full.parquet \
    --model_name pubmed_no_reason_900
