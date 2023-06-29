# pip install ranx for this script

from ranx import Qrels, Run, compare
data = 'dl19-passage'  # or dl20

qrels = Qrels.from_file(f"{data}.qrels.txt")
run_1 = Run.from_file(f"msmarco_{data}_BM25.trec")
run_1.name = 'BM25'
run_2 = Run.from_file(f"msmarco_{data}_dense_retriever.trec")
run_2.name = 'DR'
run_3 = Run.from_file(f"msmarco_{data}_splade.trec")
run_3.name = 'SPLADE'
run_4 = Run.from_file(f"msmarco_{data}_splade_no_stop.trec")
run_4.name = 'no-stop'
run_5 = Run.from_file(f"msmarco_{data}_splade_stop_150.trec")
run_5.name = 'stop-150'
run_6 = Run.from_file(f"msmarco_{data}_splade_random_150.trec")
run_6.name = 'random-150'
run_7 = Run.from_file(f"msmarco_{data}_splade_random_768.trec")
run_7.name = 'random-768'
run_8 = Run.from_file(f"msmarco_{data}_splade_lowfreq_150.trec")
run_8.name = 'lowfreq-150'
run_9 = Run.from_file(f"msmarco_{data}_splade_lowfreq_768.trec")
run_9.name = 'lowfreq-768'
run_10 = Run.from_file(f"msmarco_{data}_splade_added_latent_150.trec")
run_10.name = 'added-latent-150'
run_11 = Run.from_file(f"msmarco_{data}_splade_latent_150.trec")
run_11.name = 'latent-150'
run_12= Run.from_file(f"msmarco_{data}_splade_added_latent_768.trec")
run_12.name = 'added-latent-768'
run_13 = Run.from_file(f"msmarco_{data}_splade_latent_768.trec")
run_13.name = 'latent-768'
report = compare(
    qrels,
    runs=[run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_9, run_10, run_11, run_12, run_13],
    metrics=["ndcg@10", "recall@1000-l2", "map@1000-l2"],
    max_p=0.01,  # P-value threshold
    # Use `student` for Two-sided Paired Student's t-Test (fast / default),
    # `fisher` for Fisher's Randomization Test (slow),
    # or `tukey` for Tukey's HSD test (fast)
    stat_test="student",
    make_comparable=True
)

report.rounding_digits = 3
report.show_percentages = True
# print(report.to_latex())
print(report)