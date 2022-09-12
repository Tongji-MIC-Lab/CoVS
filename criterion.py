import collections
import time
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging

import misc.utils as utils

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self, weight=0.0):
        super(LanguageModelCriterion, self).__init__()
        self.weight = weight

    def forward(self, input, target, weights=None, compute_prob=False):
        """
        Args:
            input: 64*5*30*9837
            target: 64*5*30
            weights: 64*5*30
        """
        
        if len(target.size()) == 3:  # separate story
            input = input.view(-1, input.size(2), input.size(3)) # 320*30*9837
            target = target.view(-1, target.size(2)) # 320*30

        seq_length = input.size(1)
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = (target > 0).float()
        mask = to_contiguous(torch.cat([Variable(mask.data.new(mask.size(0), 1).fill_(1)), mask[:, :-1]], 1))

        # reshape the variables
        input = to_contiguous(input).view(-1, input.size(2)) # (320*30)*9837
        target = to_contiguous(target).view(-1, 1) # (320*30)*1
        mask = mask.view(-1, 1) # (320*30)*1

        # cross entropy
        if weights is None:
            output = - input.gather(1, target) * mask
        else:
            output = - input.gather(1, target) * mask * to_contiguous(weights).view(-1, 1)

        if compute_prob:
            output = output.view(-1, seq_length)
            mask = mask.view(-1, seq_length)
            return output.sum(-1) / mask.sum(-1)

        output = torch.sum(output) / torch.sum(mask)

        # output's entropy, no reference
        entropy = -(torch.exp(input) * input).sum(-1) * mask
        entropy = torch.sum(entropy) / torch.sum(mask)

        return output + self.weight * entropy

class ReinforceCriterion(nn.Module):
    def __init__(self, dataset, reward_type, device):
        super(ReinforceCriterion, self).__init__()
        self.dataset = dataset
        self.reward_type = reward_type
        self.bleu = None # store BLEU type 1~4
        self.device = device

        if self.reward_type == 'METEOR':
            from vist_eval.meteor.meteor import Meteor
            self.reward_scorer = Meteor()
        elif self.reward_type == 'CIDEr':
            cached_tokens = '../datasets/VIST/VIST-train-words'
            sys.path.append("cider")
            from vist_eval.pyciderevalcap.ciderD.ciderD import CiderD
            self.reward_scorer = CiderD(df=cached_tokens)
        elif self.reward_type == 'Bleu_4' or self.reward_type == 'Bleu_3' or self.reward_type == 'Bleu_2' or self.reward_type == 'Bleu_1':
            from vist_eval.bleu.bleu import Bleu
            self.bleu = int(self.reward_type[-1])
            self.reward_scorer = Bleu(self.bleu)
        elif self.reward_type == 'ROUGE_L':
            from vist_eval.rouge.rouge import Rouge
            self.reward_scorer = Rouge()
        else:
            err_msg = "{} scorer hasn't been implemented".format(self.reward_type)
            logging.error(err_msg)
            raise Exception(err_msg)

    def _cal_action_loss(self, log_probs, reward, mask):
        output = -log_probs * reward * mask # 64*5*30, log_probs is negative
        output = torch.sum(output) / torch.sum(mask) # normalize
        return output

    def _cal_value_loss(self, reward, baseline, mask):
        output = (reward - baseline).pow(2) * mask # 64*5*30
        output = torch.sum(output) / torch.sum(mask) # single value, normalize
        return output

    def _cal_rewards(self, seq, index):
        sents = utils.decode_story(self.dataset.get_vocab(), seq)
        rewards = []
        for i, story in enumerate(sents):
            fid = self.dataset.get_fid(index[i])
            GT_story = self.dataset.get_GT(index[i])
            result = {fid: [story]}
            gt = {fid: GT_story}
            score, _ = self.reward_scorer.compute_score(gt, result)
            if self.bleu is not None:
                rewards.append(score[self.bleu - 1])
            else:
                rewards.append(score)

        return rewards

    def forward(self, seq, seq_log_probs, baseline, index, rewards=None, baseline_loss=True):
        """
        :param seq: (batch_size, 5, seq_length)
        :param seq_log_probs: (batch_size, 5, seq_length)
        :param baseline: (batch_size, 5, seq_length)
        :param index: (batch_size,)
        :param rewards: (batch_size, 5, seq_length)
        :return:
        """

        if rewards is None:
            batch_size = seq.size(0)

            rewards = self._cal_rewards(seq, index)
            rewards = torch.FloatTensor(rewards) # (batch_size,)
            avg_reward = rewards.mean()
            rewards = Variable(rewards.view(batch_size, 1, 1).expand_as(seq)).cuda() # batch_size*5*30
        else:
            avg_reward = rewards.mean()
            # rewards = rewards.view(-1, 5, 1) # 64*5*1

        # get the mask
        mask = (seq > 0).float()  # its size is supposed to be (batch_size, 5, seq_length)
        if mask.size(2) > 1:
            mask = torch.cat([mask.new(mask.size(0), mask.size(1), 1).fill_(1), mask[:, :, :-1]], 2).contiguous()
        else:
            mask.fill_(1)
        mask = torch.tensor(mask) # 64*5*30
        mask = mask.to(self.device)

        avg_baseline = baseline.mean()
        baseline = baseline.to(self.device)
        seq_log_probs = seq_log_probs.to(self.device)

        # compute the loss
        advantage = Variable(rewards) # 64*5
        action_loss = self._cal_action_loss(seq_log_probs, advantage, mask) # equation 9.2

        if baseline_loss: # calc baseline estimator MSE loss, used when NN is baseline
            value_loss = self._cal_value_loss(rewards, baseline, mask) # ^2
            total_loss = action_loss + value_loss
        else:
            total_loss = action_loss

        return total_loss, avg_reward, avg_baseline

    def selfcritical(self, model, features, keywords, seq, seq_log_probs, index):
        """ compute reward by self-critical, using greedy search score as baseline
        """

        batch_size = seq.size(0)

        with torch.no_grad():
            # greedy search as inference algorithm
            greedy_seq, _ = model.sample_greedy(features, keywords)
            # beam search as inference algorithm
            # greedy_seq, _ = model.predict(features, keywords, beam_size = 3)

            greedy_rewards = self._cal_rewards(greedy_seq, index)
            greedy_rewards = torch.FloatTensor(greedy_rewards) # (batch_size,)
            greedy_rewards = Variable(greedy_rewards.view(batch_size, 1, 1).expand_as(seq)).cuda() # batch_size*5*30

        # compute rl loss with greedy_rewards as baseline
        return self.forward(seq, seq_log_probs, greedy_rewards, index, baseline_loss = False)

    def rollout(self, model, features, keywords, seq, seq_log_probs, baseline, index, num, step):
        """ compute reward by rollout policy

        Args:
            seq: batch_size*5*30
            num: rollout number
        """

        scores = torch.zeros_like(seq, dtype = torch.float).cuda()
        batch_size = seq.size(0)
        seq_length = seq.size(2)

        with torch.no_grad():
            for i in range(num):
                for j in range(1, seq_length, step):
                    part_seq = seq[:, :, 0:j]
                    part_seq_log_probs = seq_log_probs[:, :, 0:j]
                    sample_seq, sample_seq_log_probs, _ = model.sample_rollout(features, keywords, part_seq, part_seq_log_probs)

                    # compute the sample's reward
                    sample_rewards = self._cal_rewards(sample_seq, index)
                    sample_rewards = torch.FloatTensor(sample_rewards) # batch_size,
                    sample_rewards = Variable(sample_rewards.view(batch_size, 1, 1).expand(batch_size, 5, 1)).cuda() # batch_size*5*1

                    if step == 1:
                        scores[:, :, j-1] = sample_rewards[:, :, 0]
                    else: # fill score in step range
                        if j-1+step <= seq_length:
                            tmp_upbound = j-1+step
                        else:
                            tmp_upbound = seq_length
                        scores[:, :, j-1:tmp_upbound] = sample_rewards.expand(batch_size, 5, step)

                # compute the sample's reward
                sample_rewards = self._cal_rewards(seq, index)
                sample_rewards = torch.FloatTensor(sample_rewards)
                sample_rewards = Variable(sample_rewards.view(batch_size, 1, 1).expand(batch_size, 5, 1)).cuda()

                scores[:, :, seq_length-1] = sample_rewards[:, :, 0]

        scores /= num

        # compute rl loss with rewards
        return self.forward(seq, seq_log_probs, baseline, index, rewards = scores, baseline_loss = True)