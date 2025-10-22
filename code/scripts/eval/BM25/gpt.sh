
DATASET=$1
MODEL_NAME=$2
DATA_PATH=data/local_index_search/$DATASET/sparse/test.parquet

python src/eval/BM25/baselines/model_generate/gpt.py \
    --data_path $DATA_PATH \
    --dataset $DATASET  \
    --model_name $MODEL_NAME