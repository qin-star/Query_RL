DATASET=$1
MODEL_NAME=$2
DATA_PATH=data/local_index_search/no_reason/Dense/$DATASET/test.parquet
SAVE_DIR=../results_dense/no_reason

python src/eval/BM25/baselines/model_generate/claude.py \
    --data_path $DATA_PATH \
    --dataset $DATASET \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME