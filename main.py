import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import sys
import logging
from torch.utils.tensorboard import SummaryWriter

from dataset import VISTDataset
from eval_utils import Evaluator
import criterion
from model import *
import misc.utils as utils

#################################### hyperparameters ####################################

# OPTION = 'train'
OPTION = 'test'
VAL_ON_TEST_SET = True
VAL_TEST_MODE = 1 # 1 for original album comparison, 2 for "correct" image seqence comparison
train_resume = False
train_resume_model_path = './results-butd-fs-1/model_iter_14000.pth'
tran_resume_optimizer = False
train_resume_optimizer_path = './results-butd-fs-1/optimizer_iter_33000.pth'
test_start_from_model = './results-butd-fs-3/model_iter_45000.pth'
writer = SummaryWriter('tbruns-butd-fs-3')
save_dir = './results-butd-fs-333'

METRIC = 'METEOR' # "XE | CIDEr | ROUGE_L | METEOR | Bleu_4 | Bleu_3 | Bleu_2 | Bleu_1"
BATCH_SIZE = 64
SHUFFLE = True
NUM_WORKERS = 8
LEARNING_RATE = 4e-4
LR_HALVING = False
EPOCHS = 100
VAL_STEP = 1000

NUM_TOPICS = 128
topic_model_path = 'model_cache/topicModel_128.pth'

BEAM_SIZE = 3
MIN_LENGTH = 6 # original 6
HAMMING_DIVERSITY = False
HAMMING_F = 0.5
HAMMING_N = 2

rl_start_epoch = -1 # -1 means never
rl_weight = 1.0
# rl_weight_max = 1.0
reward_type = 'CIDEr'
rollout_num = 4
rollout_step = 5

scheduled_sampling_start = 0
scheduled_sampling_increase_every = 5
scheduled_sampling_increase_prob = 0.05
scheduled_sampling_max_prob = 0.50
ss_prob = 0
grad_clip = 10

##############################################################

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    global LEARNING_RATE, ss_prob, rl_weight

    ################### set up dataset and dataloader ########################
    dataset = VISTDataset(num_topics = NUM_TOPICS)
    vocab_size = dataset.get_vocab_size()
    seq_length = dataset.get_story_length()

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    if VAL_ON_TEST_SET:
        dataset.test()
    else:
        dataset.val()
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    ##################### set up model, criterion and optimizer ####################s
    bad_valid = 0

    # set up evaluator
    if VAL_ON_TEST_SET:
        evaluator = Evaluator(save_dir, 'test', VAL_TEST_MODE, BEAM_SIZE)
    else:
        evaluator = Evaluator(save_dir, 'val', VAL_TEST_MODE, BEAM_SIZE)

    # set up criterion
    crit = criterion.LanguageModelCriterion()
    rl_crit = criterion.ReinforceCriterion(dataset, reward_type, device)

    # set up model
    keywords_dist = dataset.keywords_dist['train']
    model = BuTdModel(dataset, vocab_size, seq_length, NUM_TOPICS, keywords_dist, MIN_LENGTH, HAMMING_DIVERSITY, HAMMING_F, HAMMING_N)
    model = model.to(device)
    if train_resume:
        if os.path.exists(train_resume_model_path):
            if rl_start_epoch >= 0: # rl, load state except NN baseline
                print("Load pretrained model into part of model now")
                pretrained_dict = torch.load(os.path.join(train_resume_model_path))
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                print("Load pretrained model")
                model.load_state_dict(torch.load(os.path.join(train_resume_model_path)))
        else:
            err_msg = "model path doesn't exist: {}".format(train_resume_model_path)
            logging.error(err_msg)
            raise Exception(err_msg)
    else:
        print('No need to load pretrained model')

    # load topic model
    topicModel = TopicModel(NUM_TOPICS)
    topicModel = topicModel.to(device)
    if os.path.exists(topic_model_path):
        print("load pretrained topic model")
        topicModel.load_state_dict(torch.load(topic_model_path))
    else:
        err_msg = "topic model path doesn't exist: {}".format(topic_model_path)
        logging.error(err_msg)
        raise Exception(err_msg)

    # set up optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr = LEARNING_RATE, betas = (0.8, 0.999), eps = 1e-8)
    if tran_resume_optimizer:
        if os.path.isfile(train_resume_optimizer_path):
            print("Load optimizer from {}".format(train_resume_optimizer_path))
            optimizer.load_state_dict(torch.load(train_resume_optimizer_path))
            LEARNING_RATE = optimizer.param_groups[0]['lr']
            print("Loaded learning rate is {}".format(LEARNING_RATE))

    ############################## training ##################################
    dataset.train()
    model.train()
    topicModel.eval()
    iteration = 0 # total number of iterations, regardless epochs
    best_val_score = None
    for epoch in range(0, EPOCHS):
        # Assign the scheduled sampling prob
        if epoch > scheduled_sampling_start and scheduled_sampling_start >= 0:
            frac = (epoch - scheduled_sampling_start) // scheduled_sampling_increase_every
            ss_prob = min(scheduled_sampling_increase_prob * frac, scheduled_sampling_max_prob)
            model.ss_prob = ss_prob
        # # increase rl weight
        # if rl_start_epoch >= 0 and (epoch + 1) % 5 == 0 and rl_weight < rl_weight_max:
        #     rl_weight += 0.05
        #     writer.add_scalar('train/rl_weight', rl_weight, iteration)

        start = time.time()

        for iter, batch in enumerate(train_loader):
            iteration += 1
            torch.cuda.synchronize()

            feature_fc = batch['feature_fc'].to(device) # batch_size*5*2048
            target = Variable(batch['split_story']).to(device) # batch_size*5*30
            index = batch['index']

            optimizer.zero_grad() # zero the parameter gradients

            with torch.no_grad():
                keywords = topicModel(feature_fc) # batch_size*5*NUM_TOPICS

            # cross entropy loss
            output = model(feature_fc, keywords, target) # 64*5*30*vocab_size
            loss = crit(output, target) # LanguageModelCriterion

            # # auxiliary loss
            # aloss_weights = dataset.get_ITFweights(target, 0.4) # ITF weights
            # aloss = crit(output, target, aloss_weights)
            # aloss = crit.hamming_diversity_crit(output, target, 2, 0.3)
            # loss = loss + 0.1 * aloss

            # reinforcement learning loss
            if rl_start_epoch >= 0 and epoch >= rl_start_epoch:
                # actor-critic
                seq, seq_log_probs, baseline = model.sample(feature_fc, keywords)
                rl_loss, avg_score, avg_baseline = rl_crit(seq, seq_log_probs, baseline, index)

                # # self-critical
                # seq, seq_log_probs, _ = model.sample(feature_fc, keywords)
                # rl_loss, avg_score, avg_baseline = rl_crit.selfcritical(model, feature_fc, keywords, seq, seq_log_probs, index)
                
                # rollout
                # seq, seq_log_probs, baseline = model.sample(feature_fc, keywords)
                # rl_loss, avg_score, avg_baseline = rl_crit.rollout(model, feature_fc, keywords, seq, seq_log_probs, baseline, index, rollout_num, rollout_step)

                if iteration % 10 == 0:
                    print("rl loss: {}".format(rl_loss.item()))
                    print("rl training average {} score: {}".format(reward_type, avg_score))
                    print("rl training average baseline: {}".format(avg_baseline))
                    writer.add_scalar('train/rl_loss', rl_loss.item(), iteration)
                    writer.add_scalar('train/rl_avg_score', avg_score, iteration)
                    writer.add_scalar('train/rl_avg_baseline', avg_baseline, iteration)
                loss = rl_loss

            loss.backward()
            train_loss = loss.item()

            nn.utils.clip_grad_norm(model.parameters(), grad_clip, norm_type=2)
            optimizer.step()
            torch.cuda.synchronize() # time

            # Write the training loss summary
            if iteration % 10 == 0:
                print("Epoch {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, iter,
                    len(train_loader), train_loss, time.time() - start))
                start = time.time()

                writer.add_scalar('train/loss', train_loss, iteration)
                writer.add_scalar('train/scheduled_sampling_prob', ss_prob, iteration)

            # val
            if iteration % VAL_STEP == 0:
                with torch.no_grad():
                    val_loss, predictions, metrics = evaluator.eval_story(model, topicModel, crit, dataset, val_loader, iteration, VAL_ON_TEST_SET)
                    if METRIC == 'XE':
                        current_score = -val_loss
                    else:
                        current_score = metrics[METRIC]
                    # write metrics
                    if not VAL_ON_TEST_SET:
                        writer.add_scalar('val/loss', val_loss, iteration)
                    writer.add_scalar('val/Bleu_1', metrics['Bleu_1'], iteration)
                    writer.add_scalar('val/Bleu_2', metrics['Bleu_2'], iteration)
                    writer.add_scalar('val/Bleu_3', metrics['Bleu_3'], iteration)
                    writer.add_scalar('val/Bleu_4', metrics['Bleu_4'], iteration)
                    writer.add_scalar('val/METEOR', metrics['METEOR'], iteration)
                    writer.add_scalar('val/ROUGE_L', metrics['ROUGE_L'], iteration)
                    writer.add_scalar('val/CIDEr', metrics['CIDEr'], iteration)

                    # save model
                    best_flag = False
                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    # save the model at current iteration
                    checkpoint_path = os.path.join(save_dir, 'model_iter_{}.pth'.format(iteration))
                    torch.save(model.state_dict(), checkpoint_path)
                    # save optimizer
                    if optimizer is not None:
                        optimizer_path = os.path.join(save_dir, 'optimizer_iter_{}.pth'.format(iteration))
                        torch.save(optimizer.state_dict(), optimizer_path)
                    # save as latest model
                    checkpoint_path = os.path.join(save_dir, 'model.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    if best_flag:
                        checkpoint_path = os.path.join(save_dir, 'model-best.pth')
                        torch.save(model.state_dict(), checkpoint_path)
                        print("model saved to {}".format(checkpoint_path))

                    # halve the learning rate if not improving for a long time
                    if LR_HALVING:
                        if best_val_score > current_score:
                            bad_valid += 1
                            if bad_valid >= 4:
                                LEARNING_RATE = LEARNING_RATE / 2.0
                                print("halve learning rate to {}".format(LEARNING_RATE))
                                writer.add_scalar('train/learning_rate', LEARNING_RATE, iteration)
                                # reload the best model and restart training from it             
                                checkpoint_path = os.path.join(save_dir, 'model-best.pth')
                                model.load_state_dict(torch.load(checkpoint_path))
                                utils.set_lr(optimizer, LEARNING_RATE)  # set the decayed rate
                                bad_valid = 0
                                print("bad valid : {}".format(bad_valid))
                        else:
                            print("achieving best {} score: {}".format(METRIC, current_score))
                            bad_valid = 0

def test():
    dataset = VISTDataset(num_topics = NUM_TOPICS)
    vocab_size = dataset.get_vocab_size()
    seq_length = dataset.get_story_length()

    dataset.test()

    test_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
    evaluator = Evaluator(save_dir, 'test', VAL_TEST_MODE, BEAM_SIZE)

    keywords_dist = dataset.keywords_dist['train']
    model = BuTdModel(dataset, vocab_size, seq_length, NUM_TOPICS, keywords_dist, MIN_LENGTH, HAMMING_DIVERSITY, HAMMING_F, HAMMING_N)
    model = model.to(device)
    if os.path.exists(test_start_from_model):
        print("Start test from pretrained model")
        model.load_state_dict(torch.load(test_start_from_model))
    else:
        err_msg = "model path doesn't exist: {}".format(test_start_from_model)
        logging.error(err_msg)
        raise Exception(err_msg)

    # load topic model
    topicModel = TopicModel(NUM_TOPICS)
    topicModel = topicModel.to(device)
    if os.path.exists(topic_model_path):
        print("load pretrained topic model")
        topicModel.load_state_dict(torch.load(topic_model_path))
    else:
        err_msg = "topic model path doesn't exist: {}".format(topic_model_path)
        logging.error(err_msg)
        raise Exception(err_msg)

    with torch.no_grad():
        predictions, metrics = evaluator.test_story(model, topicModel, dataset, test_loader, HAMMING_DIVERSITY, HAMMING_F, HAMMING_N)

if __name__ == "__main__":

    if OPTION == 'train':
        print('Begin training:')
        train()
    else:
        print('Begin testing:')
        test()
