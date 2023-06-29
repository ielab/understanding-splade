import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer, BertLMHeadModel
from transformers import (
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from tevatron.arguments import SpladeModelArguments, DataArguments, SpladeTrainingArguments
from tevatron.data import TrainDataset, QPCollator
from tevatron.modeling import SpladeModel
from tevatron.trainer import TevatronTrainer
from tevatron.datasets import HFTrainDataset
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Dict, Optional

logger = logging.getLogger(__name__)



class SpladeTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super(SpladeTrainer, self).__init__(*args, **kwargs)
        if self.args.negatives_x_device:    
            self.world_size = torch.distributed.get_world_size()
        self._q_flops_loss = None
        self._p_flops_loss = None
        self._contrastive_loss = None
    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def compute_loss(self, model, inputs):
        query, passage = inputs
        output = model(query=query, passage=passage)
        q_reps = output.q_reps
        p_reps = output.p_reps
        contrastive_loss = output.loss
        q_flops_loss = self.args.q_flops_loss_factor*self._flops(q_reps)
        p_flops_loss = self.args.p_flops_loss_factor*self._flops(p_reps)
        if self.args.negatives_x_device:
            q_flops_loss *= self.world_size
            p_flops_loss *= self.world_size

        # # for logging different losses
        # if self._q_flops_loss is None:
        #     self._q_flops_loss = torch.tensor(0.0).to(self.args.device)
        # if self._p_flops_loss is None:
        #     self._p_flops_loss = torch.tensor(0.0).to(self.args.device)
        # if self._contrastive_loss is None:
        #     self._contrastive_loss = torch.tensor(0.0).to(self.args.device)
        # self._p_flops_loss += p_flops_loss
        # self._q_flops_loss += q_flops_loss
        # self._contrastive_loss += contrastive_loss
        # if self.state.global_step != 0 and (self.state.global_step+1) % self.args.logging_steps == 0:
        #     q_flops_loss_scalar = self._nested_gather(self._q_flops_loss).mean().item()
        #     p_flops_loss_scalar = self._nested_gather(self._p_flops_loss).mean().item()
        #     contrastive_loss_scalar = self._nested_gather(self._contrastive_loss).mean().item()
        #     # reset _loss to zero
        #     self._q_flops_loss -= self._q_flops_loss
        #     self._p_flops_loss -= self._p_flops_loss
        #     self._contrastive_loss -= self._contrastive_loss
        #     logs: Dict[str, float] = {}
        #     logs["q_flops_loss"] = round(q_flops_loss_scalar / ((self.state.global_step+1) - self._globalstep_last_logged), 4)
        #     logs["p_flops_loss"] = round(p_flops_loss_scalar / ((self.state.global_step+1) - self._globalstep_last_logged), 4)
        #     logs["contrastive_loss"] = round(
        #         contrastive_loss_scalar / ((self.state.global_step + 1) - self._globalstep_last_logged), 4)
        #     if self.args.negatives_x_device:
        #         for k, v in logs.items():
        #             logs[k] = v / self.world_size
        #     self.log(logs)

        return contrastive_loss + q_flops_loss + p_flops_loss

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint and tokenizer to %s", output_dir)
        self.model.save(output_dir)
        self.tokenizer.save_pretrained(output_dir)


TrainingArguments = SpladeTrainingArguments

def main():
    parser = HfArgumentParser((SpladeModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: SpladeModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = SpladeModel.build(
        model_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)

    if model_args.latent_vocab_size is not None:
        tokenizer.add_tokens([f"[latent_token{i}]" for i in range(model_args.latent_vocab_size)])
        model.lm_q.resize_token_embeddings(len(tokenizer))
        if model_args.mask_file is not None:
            with open(model_args.mask_file, "w") as f:
                [f.write(f"[latent_token{i}]" + "\n") for i in range(model_args.latent_vocab_size)]

    if model_args.stopwords_weight is not None:
        if model_args.mask_file is not None:
            model.set_mask_weight(tokenizer, model_args.stopwords_weight, model_args.mask_file)
        else:
            model.set_stopwords_weight(tokenizer, model_args.stopwords_weight)

    trainer = SpladeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()
