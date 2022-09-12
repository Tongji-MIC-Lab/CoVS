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
from model import *

OPTION = 'train'
# OPTION = 'test'

# test_start_from_model = 'results-topic-16/topicModel.pth'

writer = SummaryWriter('tbruns-topic-128/')
save_dir = './results-topic-128'

NUM_TOPICS = 128

BATCH_SIZE = 64
SHUFFLE = True
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
EPOCHS = 30

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    global LEARNING_RATE

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = VISTDataset(num_topics = NUM_TOPICS)
    dataset.train()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    dataset.val()
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    crit = nn.KLDivLoss()

    model = TopicModel(NUM_TOPICS)
    model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr = LEARNING_RATE, betas = (0.8, 0.999), eps = 1e-8, weight_decay = 0)
    
    dataset.train()
    model.train()
    iteration = 0
    best_val_score = None

    for epoch in range(0, EPOCHS):
        start = time.time()
        for iter, batch in enumerate(train_loader):
            iteration += 1
            torch.cuda.synchronize()

            feature_fc = Variable(batch['feature_fc']).to(device) # batch_size*5*2048
            keywords = Variable(batch['keywords']).to(device) # batch_size*5*topic_num

            optimizer.zero_grad()

            output = model(feature_fc) # batch_size*5*topic_num
            
            loss = crit(output, keywords)

            loss.backward()
            train_loss = loss.item()

            optimizer.step()
            torch.cuda.synchronize()

            if iteration % 100 == 0:
                print("Epoch {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, iter,
                    len(train_loader), train_loss, time.time() - start))
                start = time.time()

                writer.add_scalar('train/loss', train_loss, iteration)
            
            if iteration % 1000 == 0:
                with torch.no_grad():
                    print("Evaluating...")
                    start = time.time()
                    model.eval()
                    dataset.val()
                    loss_sum = 0
                    loss_evals = 0
                    for eval_iter, eval_batch in enumerate(val_loader):
                        iter_start = time.time()

                        feature_fc = Variable(eval_batch['feature_fc']).cuda()
                        keywords = Variable(eval_batch['keywords']).cuda()

                        output = model(feature_fc)

                        loss = crit(output, keywords).item()
                        loss_sum += loss
                        loss_evals += 1

                        print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(eval_iter, len(val_loader), 
                                                        eval_iter * 100.0 / len(val_loader), time.time() - iter_start))

                    val_loss = loss_sum / loss_evals

                    model.train()
                    dataset.train()

                    current_score = -val_loss
                   
                    # write metrics
                    writer.add_scalar('val/loss', val_loss, iteration)

                    # save model
                    best_flag = False
                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    # # save the model at current iteration
                    # checkpoint_path = os.path.join(save_dir, 'topicModel_iter_{}.pth'.format(iteration))
                    # torch.save(model.state_dict(), checkpoint_path)
                    # save as latest model
                    checkpoint_path = os.path.join(save_dir, 'topicModel.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("topic model saved to {}".format(checkpoint_path))
                    # save optimizer
                    if optimizer is not None:
                        optimizer_path = os.path.join(save_dir, 'topicOptimizer.pth')
                        torch.save(optimizer.state_dict(), optimizer_path)
                    if best_flag:
                        checkpoint_path = os.path.join(save_dir, 'topicModel-best.pth')
                        torch.save(model.state_dict(), checkpoint_path)
                        print("topic model saved to {}".format(checkpoint_path))

def test():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = VISTDataset(num_topics = NUM_TOPICS)
    dataset.test()
    test_loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = NUM_WORKERS)

    crit = nn.KLDivLoss()

    model = TopicModel(NUM_TOPICS)
    model = model.to(device)
    if os.path.exists(test_start_from_model):
        print("Start test from pretrained model")
        model.load_state_dict(torch.load(test_start_from_model))
    else:
        err_msg = "model path doesn't exist: {}".format(test_start_from_model)
        logging.error(err_msg)
        raise Exception(err_msg)

    ### test on train set
    dataset.train()
    model.eval()
    for j in range(len(dataset.story_ids['train'])):
        story_id = dataset.story_ids['train'][j]
        story = dataset.story_line['train'][story_id]
        split_story = dataset.story_h5[story['text_index']]
        story_text = story['origin_text']
        
        if story['flickr_id'][0] == '17584390':
            print()
            print(story['flickr_id'])

            feature_fc = np.zeros((story['length'], feat_size), dtype='float32')
            for i in range(story['length']):
                # load fc feature
                path_resnet_features = '../datasets/resnet_features'
                fc_path = os.path.join(path_resnet_features, 'fc', 'train', '{}.npy'.format(story['flickr_id'][i]))
                feature_fc[i] = np.load(fc_path)
            
            total_len = 0
            for i in range(5):            
                s_len = len(list(filter(lambda x: x != 0, split_story[i])))
                s = ' '.join(story_text.split()[total_len: total_len+s_len])
                total_len += s_len
                print(s)
                print(np.argsort(dataset.keywords['train'][j][i])[::-1] + 1)
                print(dataset.keywords['train'][j][i])

            feature_fc = torch.tensor(feature_fc).cuda()

            output = model(feature_fc.unsqueeze(0))
            print(torch.exp(output[0, :, :]))

    # ### test on test set
    # with torch.no_grad():
    #     print("Evaluating...")
    #     start = time.time()
    #     model.eval()
    #     dataset.test()
    #     loss_sum = 0
    #     loss_evals = 0

    #     for test_iter, test_batch in enumerate(test_loader):
    #         iter_start = time.time()

    #         feature_fc = Variable(test_batch['feature_fc']).cuda()
    #         keywords = Variable(test_batch['keywords']).cuda()
    #         fid = test_batch['fid']

    #         output = model(feature_fc)

    #         loss = crit(output, keywords).item()
    #         loss_sum += loss
    #         loss_evals += 1

    #         # print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(test_iter, len(test_loader), 
    #         #                                 test_iter * 100.0 / len(test_loader), time.time() - iter_start))

    #     test_loss = loss_sum / loss_evals

    #     print('test result: ', test_loss)

if __name__ == "__main__":

    if OPTION == 'train':
        print('Begin training:')
        train()
    else:
        print('Begin testing:')
        test()