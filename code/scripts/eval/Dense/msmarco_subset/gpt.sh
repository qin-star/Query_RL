
DATASET=$1
DATA_PATH=data/local_index_search/$DATASET/dense/test.parquet
SAVE_DIR=../results_dense

python src/eval/Dense/baselines/model_generate/msmarco_subset/gpt-4o.py \
    --data_path $DATA_PATH \
    --dataset $DATASET \
    --save_dir $SAVE_DIR