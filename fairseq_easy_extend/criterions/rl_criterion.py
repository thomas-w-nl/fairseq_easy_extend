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


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
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

        loss, reward = self._compute_loss(outs, tgt_tokens, masks)

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

    def _compute_loss(self, outputs, targets, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        # padding mask
        ##If you take mask before you do sampling: you sample over a BATCH and your reward is on token level
        # if you take mask after, you sample SENTENCES and calculate reward on a sentence level
        # but make sure you apply padding mask after both on log prob outputs, reward and id's (you might need them for gather function to           extract log_probs of the samples)

        # Example 1: mask before sampling
        # if masks is not None:
        #    outputs, targets = outputs[masks], targets[masks]

        # we take a softmax over outputs
        # argmax over the softmax \ sampling (e.g. multinomial)
        # sampled_sentence = [4, 17, 18, 19, 20]
        # sampled_sentence_string = tgt_dict.string([4, 17, 18, 19, 20])
        # target_sentence = "I am a sentence"
        # with torch.no_grad()
        # R(*) = eval_metric(sampled_sentence_string, target_sentence)
        # R(*) is a number, BLEU, сhrf, etc.

        # loss = -log_prob(outputs)*R()
        # loss = loss.mean()

        # Example 2: mask after sampling
        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)
        with torch.no_grad():
            probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
        sample_idx = torch.multinomial(probs, 1, replacement=True).view(bsz, seq_len)

        # sampled_sentence_string = self.tgt_dict.string(
        #     sample_idx)  # here you might also want to remove tokenization and bpe

        # TODO check this the correct way to remove tokenization
        # sampled_sentence_string = self.decode(sample_idx)

        # print(sampled_sentence_string[0])  # --> if you apply mask before, you get a sentence which is one token
        # imagine output[mask]=[MxV] where M is a sequence of all tokens in batch excluding padding symbols
        # now you sample 1 vocabulary index for each token, so you end up in [Mx1] matrix
        # when you apply string, it treats every token as a separate sentence --> hence you calc token-level metric. SO it makes much more sense to apply mask after sampling(!)

        # target_sentence = self.decode(targets)

        ###HERE calculate metric###
        with torch.no_grad():
            rewards = []
            for batch_idx in range(bsz):
                mask_i = masks[batch_idx]
                sampled_sentence_string = self.decode(sample_idx[[batch_idx], mask_i])
                target_sentence = self.decode(targets[[batch_idx], mask_i])

                # print("target_sentence          :", target_sentence[:50])
                # print("sampled_sentence_string  :", sampled_sentence_string[:50])
                try:
                    reward = self.bleu.corpus_score([sampled_sentence_string], [[target_sentence]]).score
                except IndexError as e:
                    print("IndexError, mask likely 0 or predicted 0 tokens.")
                    reward = 0
                # print(reward)
                rewards.append(reward)

        rewards = torch.tensor(rewards)
        # print("sample_idx",sample_idx.shape)

        # print("rewards", rewards.shape)
        reward = (torch.ones((seq_len, bsz)) * rewards).T.cuda()
        # reward is a number, BLEU, сhrf, etc.
        # expand it to make it of a shape BxT - each token gets the same reward value (e.g. bleu is 20, so each token gets reward of 20 [20,20,20,20,20])

        # now you need to apply mask on both outputs and reward
        if masks is not None:
            probs, targets = probs[masks], targets[masks]
            reward, sample_idx = reward[masks], sample_idx[masks]

        # outputs = 144, 55, vocab_size

        # log_probs = torch.log(outputs.view(-1, vocab_size))

        # log_probs = F.log_probs(outputs, dim=-1)
        log_probs_of_samples = torch.gather(torch.log(probs), 2, sample_idx.unsqueeze(1))
        # log_probs_of_samples = outputs[range(log_probs.shape[0]), sample_idx.ravel()]

        # print("Log_probs", log_probs.shape)
        # log_probs_of_samples = torch.gather(...)
        loss = -log_probs_of_samples * reward
        loss = loss.mean()

        # For more about mask see notes on NLP2-notes-on-mask

        return loss, reward.mean()

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
