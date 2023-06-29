
num_token=150 # or 768
RUN=added_latent_${num_token}

export WANDB_PROJECT=understanding-splade
CUDA_VISIBLE_DEVICES=0 python train_splade.py \
  --output_dir msmarco_models/${RUN} \
  --run_name ${RUN} \
  --model_name_or_path Luyu/co-condenser-marco \
  --save_steps 40000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --latent_vocab_size ${num_token} \
  --learning_rate 5e-6 \
  --q_max_len 128 \
  --p_max_len 128 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 3 \
  --report_to wandb \
  --logging_steps 100 \
  --cache_dir cache


mkdir -p encoding_splade/query_${RUN}
mkdir -p encoding_splade/corpus_${RUN}
for i in $(seq -f "%02g" 0 9)
do
python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path encoding_splade/corpus_${RUN}/split${i}.jsonl \
  --encode_num_shard 10 \
  --encode_shard_index ${i} \
  --cache_dir cache
done

python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path encoding_splade/query_${RUN}/dev.tsv \
  --cache_dir cache

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input encoding_splade/corpus_${RUN} \
  --index indexes/lucene-index.msmarco-passage-${RUN} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 32 \
  --impact --pretokenized


python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-${RUN} \
  --topics encoding_splade/query_${RUN}/dev.tsv \
  --output runs/msmarco_dev_${RUN}.tsv \
  --output-format msmarco \
  --batch 128 --threads 32 \
  --hits 1000 \
  --impact

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset runs/msmarco_dev_${RUN}.tsv


QUERY=dl19-passage
python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 64 \
  --encode_in_path ${QUERY}-queries.tsv \
  --encoded_save_path encoding_splade/query_${RUN}/${QUERY}.tsv \
  --cache_dir cache

python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-${RUN} \
  --topics encoding_splade/query_${RUN}/${QUERY}.tsv \
  --output runs/msmarco_${QUERY}_${RUN}.trec \
  --output-format trec \
  --batch 128 --threads 32 \
  --hits 1000 \
  --impact

python -m pyserini.eval.trec_eval -l 2 -m ndcg_cut.10 -m map -m recall.1000 ${QUERY} runs/msmarco_${QUERY}_${RUN}.trec


QUERY=dl20
python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 64 \
  --encode_in_path ${QUERY}-queries.tsv \
  --encoded_save_path encoding_splade/query_${RUN}/${QUERY}.tsv \
  --cache_dir cache

python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-passage-${RUN} \
  --topics encoding_splade/query_${RUN}/${QUERY}.tsv \
  --output runs/msmarco_${QUERY}_${RUN}.trec \
  --output-format trec \
  --batch 128 --threads 32 \
  --hits 1000 \
  --impact

python -m pyserini.eval.trec_eval -l 2 -m ndcg_cut.10 -m map -m recall.1000 ${QUERY}-passage runs/msmarco_${QUERY}_${RUN}.trec