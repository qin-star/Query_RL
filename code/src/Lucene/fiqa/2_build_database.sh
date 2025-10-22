INPUT_DIR=code/data/local_index_search/fiqa/jsonl_docs
INDEX_DIR=code/data/local_index_search/fiqa/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -generator DefaultLuceneDocumentGenerator \
 -threads 4 \
 -storePositions -storeDocvectors -storeRaw
