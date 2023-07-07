# Understanding Splade

This repository contains the code for the paper: [Exploring the Representation Power of SPLADE Models](https://arxiv.org/pdf/2306.16680.pdf), ICTIR2023.

## Installation
We modify the [Tevatron](https://github.com/texttron/tevatron/) toolkit for our experiments.

First install our version of tevatron by:
```bash
cd tevatron
pip install --editable .
```
Then, install all the dependencies requried by tevatron follow the tevatron instructions.

You also will need to install [pyserini](https://github.com/castorini/pyserini) by `pip install pyserini` for indexing SPLADE encodings and evaluation.


## Experiments
The scripts that can reproduced our results in end-to-end manner are in the `tevatron/examples/splade` folder.

- `full_pipeline_bm25.sh`: the full pipeline of getting bm25 baseline results.
- `full_pipeline_dense_retriever.sh`: the full pipeline of getting DR baseline results.
- `full_pipeline_splade.sh`: the full pipeline of original SPLADEv2 model training, encoding, indexing and retrieval.
- `full_pipeline_nostopwords.sh`: SPLADEv2 that removed 150 stopwords from BERT vocabulary (no-stop).
- `full_pipline_splade_stopwords.sh`: SPLADEv2 that only assigns weights to 150 stopwords (stop-150).
- `full_pipline_splade_randwords.sh`: SPLADEv2 that only assigns weights to random tokens (random-150/786).
- `full_pipline_splade_lowfreq.sh`: SPLADEv2 that only assigns weights to low-frequency tokens (lowfreq-150/786).
- `full_pipline_splade_added_latentwords.sh`: SPLADEv2 that 150/768 latent tokens enlarged vocabulary (added-latent-150/786).
- `full_pipline_splade_latentwords.sh`: SPLADEv2 that only assigns weights to 150/768 latent tokens (latent-150/786).

We also provide all the runs files along with evaluation script in the `tevatron/examples/splade/runs` folder.

You can directly get our results by running the following commands:
```bash
pip install ranx
cd tevatron/examples/splade/runs
python3 eval.py
```
