export CUDA_VISIBLE_DEVICES=0,1

python3 -m vllm.entrypoints.openai.api_server \
    --model DeepRetrieval/DeepRetrieval-NQ-BM25-3B \
    --port 8000 \
    --max-model-len 2048 \
    --tensor-parallel-size 2 


# --model DeepRetrieval/DeepRetrieval-NQ-BM25-3B
# --model DeepRetrieval/DeepRetrieval-TriviaQA-BM25-3B \
# --model DeepRetrieval/DeepRetrieval-SQuAD-BM25-3B-200 \