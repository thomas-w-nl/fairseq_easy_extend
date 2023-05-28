import math
from argparse import Namespace

import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import encoders
from fairseq.dataclass import FairseqDataclass

from dataclasses import dataclass, field

from fairseq.logging import metrics

from sacrebleu.metrics import BLEU, CHRF, TER
from torch import nn


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_cosine", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tokenizer = encoders.build_tokenizer(Namespace(
            tokenizer='moses'
        ))
        self.tgt_dict = task.target_dictionary
        self.bleu = BLEU()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        # get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss, reward = self._compute_loss(outs, tgt_tokens, masks, model)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward.detach()
        }

        return loss, sample_size, logging_output

    def decode(self, toks, escape_unk=False):
        with torch.no_grad():
            s = self.tgt_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            s = self.tokenizer.decode(s)
        return s

    def compute_reward(self, outputs, targets):
        """
        #we take a softmax over outputs
        probs = F.softmax(outputs, dim=-1)
        #argmax over the softmax \ sampling (e.g. multinomial)
        samples_idx = torch.multinomial(probs, 1, replacement=True)
        sample_strings = self.tgt_dict.string(samples_idx)  #see dictionary class of fairseq
        #sample_strings = "I am a sentence"
        reward_vals = evaluate(sample_strings, targets)
        return reward_vals, samples_idx
        """
        pass

    def _compute_loss(self, outputs, targets, masks=None, model=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """


        # Example 2: mask after sampling
        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)
        probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
        sample_idx = torch.multinomial(probs.detach(), 1, replacement=True).view(bsz, seq_len)


        # Compute reward
        ### Cosine of embeddings:
        sim_fn = nn.CosineSimilarity(dim=1)

        sample_embeddings = model.decoder.embed_tokens(sample_idx)
        sample_embeddings = sample_embeddings[masks]  # Ntokens x 512

        tgt_embeddings = model.decoder.embed_tokens(targets)
        tgt_embeddings = tgt_embeddings[masks]  # Ntokens x 512

        reward_raw = sim_fn(sample_embeddings, tgt_embeddings)  # shape: (Ntokens,)


        # Reward scaling
        reward = reward_raw


        # now you need to apply mask on both outputs and reward
        if masks is not None:
            probs = probs.view(*outputs.shape)
            probs = probs[masks]
            # outputs, targets = outputs[masks], targets[masks]
            sample_idx = sample_idx[masks]

        # probs = 144, 55, vocab_size

        log_probs = torch.log(probs)

        log_probs_of_samples = torch.gather(log_probs, 1, sample_idx.unsqueeze(1))

        loss = -log_probs_of_samples * reward.unsqueeze(1)
        loss = loss.mean()

        # For more about mask see notes on NLP2-notes-on-mask

        return loss, reward_raw.mean()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        reward_sum = sum(log.get("reward", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("reward", reward_sum / sample_size, sample_size, round=3)
