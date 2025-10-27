python src/eval/evidence_gpt_claude.py \
    --data_path data/local_index_search/nq_serini/test.parquet \
    --model_name nq-gpt4o-no-block \
    --batch_size 8

# python src/eval/evidence_gpt_claude.py \
#     --data_path data/local_index_search/nq_serini/test.parquet \
#     --model_name nq-claude3 \
#     --batch_size 8