import torch
import logging
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
from torch import Tensor
from .encoder import EncoderPooler, EncoderModel

logger = logging.getLogger(__name__)


class SpladePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(SpladePooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class SpladeModel(EncoderModel):
    TRANSFORMER_CLS = AutoModelForMaskedLM
    STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    stopwords_mask = None

    def set_stopwords_weight(self, tokenizer: AutoTokenizer, weight: float):
        self.stopwords_mask = torch.zeros(tokenizer.vocab_size)
        size = 0
        for k, v in tokenizer.get_vocab().items():
            if k in self.STOPWORDS:
                self.stopwords_mask[v] = weight
                size += 1
            else:
                self.stopwords_mask[v] = 1 - weight
        logger.info(f"Set weight to {weight} for {size} stopwords")

    def set_mask_weight(self, tokenizer: AutoTokenizer, weight: float, mask_file: str):
        self.stopwords_mask = torch.zeros(len(tokenizer.get_vocab()))
        size = 0
        valid_tokens = set()
        with open(mask_file, 'r') as f:
            for line in f:
                valid_tokens.add(line.strip())
        for k, v in tokenizer.get_vocab().items():
            if k in valid_tokens:
                self.stopwords_mask[v] = weight
                size += 1
            else:
                self.stopwords_mask[v] = 1 - weight
        logger.info(f"Set weight to {weight} for {size} words")

    def encode_passage(self, psg):
        if psg is None:
            return None

        psg_out = self.lm_p(**psg, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(psg_out)) * psg['attention_mask'].unsqueeze(-1), dim=1)

        if self.stopwords_mask is not None:
            self.stopwords_mask = self.stopwords_mask.to(aggregated_psg_out.device)
            aggregated_psg_out = aggregated_psg_out * self.stopwords_mask

        return aggregated_psg_out

    def encode_query(self, qry):
        if qry is None:
            return None

        qry_out = self.lm_q(**qry, return_dict=True).logits
        aggregated_qry_out, _ = torch.max(torch.log(1 + torch.relu(qry_out)) * qry['attention_mask'].unsqueeze(-1), dim=1)

        if self.stopwords_mask is not None:
            self.stopwords_mask = self.stopwords_mask.to(aggregated_qry_out.device)
            aggregated_qry_out = aggregated_qry_out * self.stopwords_mask

        return aggregated_qry_out

    @staticmethod
    def build_pooler(model_args):
        pooler = SpladePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = SpladePooler(**config)
        pooler.load(model_weights_file)
        return pooler