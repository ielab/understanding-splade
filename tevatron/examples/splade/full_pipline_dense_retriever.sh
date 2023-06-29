
RUN=dense_retriever

export WANDB_PROJECT=understanding-splade
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
--output_dir msmarco_models/${RUN} \
--run_name ${RUN} \
--model_name_or_path Luyu/co-condenser-marco \
--save_steps 40000 \
--dataset_name Tevatron/msmarco-passage \
--fp16 \
--per_device_train_batch_size 8 \
--train_n_passages 8 \
--learning_rate 5e-6 \
--q_max_len 32 \
--p_max_len 128 \
--num_train_epochs 3 \
--logging_steps 100 \
--cache_dir cache


mkdir -p encoding_splade/query_${RUN}
mkdir -p encoding_splade/corpus_${RUN}
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path encoding_splade/corpus_${RUN}/split${i}.pkl \
  --encode_num_shard 10 \
  --encode_shard_index ${i} \
  --cache_dir cache
done

python -m tevatron.driver.encode \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path encoding_splade/query_${RUN}/dev.pkl \
  --cache_dir cache


python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset runs/msmarco_dev_${RUN}.tsv


QUERY=dl19-passage
python -m tevatron.driver.encode \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 64 \
  --encode_in_path ${QUERY}-queries.tsv \
  --encoded_save_path encoding_splade/query_${RUN}/${QUERY}.pkl \
  --cache_dir cache

python -m tevatron.faiss_retriever \
--query_reps encoding_splade/query_${RUN}/${QUERY}.pkl \
--passage_reps "encoding_splade/corpus_${RUN}/split*.pkl" \
--depth 1000 \
--batch_size 128 \
--save_text \
--save_ranking_to runs/msmarco_${QUERY}_${RUN}.trec

python -m pyserini.eval.trec_eval -l 2 -m ndcg_cut.10 -m map -m recall.1000 ${QUERY} runs/msmarco_${QUERY}_${RUN}.trec


QUERY=dl20
python -m tevatron.driver.encode \
  --output_dir encoding_splade \
  --model_name_or_path msmarco_models/${RUN}/checkpoint-final \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 64 \
  --encode_in_path ${QUERY}-queries.tsv \
  --encoded_save_path encoding_splade/query_${RUN}/${QUERY}.pkl \
  --cache_dir cache

python -m tevatron.faiss_retriever \
--query_reps encoding_splade/query_${RUN}/${QUERY}.pkl \
--passage_reps "encoding_splade/corpus_${RUN}/split*.pkl" \
--depth 1000 \
--batch_size 128 \
--save_text \
--save_ranking_to runs/msmarco_${QUERY}_${RUN}.trec

python -m pyserini.eval.trec_eval -l 2 -m ndcg_cut.10 -m map -m recall.1000 ${QUERY}-passage runs/msmarco_${QUERY}_${RUN}.trec