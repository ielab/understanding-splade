RUN=BM25

QUERY=msmarco-passage-dev-subset
python -m pyserini.search.lucene \
  --index msmarco-v1-passage-slim \
  --topics msmarco-passage-dev-subset \
  --output runs/msmarco_dev.${RUN}.txt \
  --output-format msmarco \
  --batch 128 --threads 32 \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset runs/run.msmarco-passage.${RUN}.txt

QUERY=dl19-passage
python -m pyserini.search.lucene \
  --index msmarco-v1-passage-slim \
  --topics ${QUERY} \
  --output runs/run.msmarco-passage.${RUN}.${QUERY}.txt \
  --output-format trec \
  --batch 128 --threads 32 \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68
python -m pyserini.eval.trec_eval -l 2 -m ndcg_cut.10 -m map -m recall.1000 ${QUERY} runs/run.msmarco-passage.${RUN}.${QUERY}.txt


QUERY=dl20
python -m pyserini.search.lucene \
  --index msmarco-v1-passage-slim \
  --topics ${QUERY} \
  --output runs/run.msmarco-passage.${RUN}.${QUERY}-passage.txt \
  --output-format trec \
  --batch 128 --threads 32 \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68
python -m pyserini.eval.trec_eval -l 2 -m ndcg_cut.10 -m map -m recall.1000 ${QUERY}-passage runs/run.msmarco-passage.${RUN}.${QUERY}-passage.txt
