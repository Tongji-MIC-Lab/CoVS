import sys
import json
import h5py
import os
import os.path
import numpy as np
import random
import csv
import spacy
import re
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from torchvision import models as pre_trained_models

# import misc.utils as utils

path_story_h5 =        './datasets/VIST/story.h5'
path_story_line_json = './datasets/VIST/story_line.json'
path_ref_dir =         './vist_reference'
# path_img_dir =          '/data/gjj/vist_image'
path_resnet_features = './datasets/resnet_features'

feat_size = 2048

preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class VISTDataset(Dataset):
    def __init__(self, num_topics):
        self.mode = 'train'

        # open the hdf5 file
        print('DataLoader loading story h5 file: ', path_story_h5)
        self.story_h5 = h5py.File(path_story_h5, 'r', driver='core')['story']
        print("story's max sentence length is ", self.story_h5.shape[1])

        print('DataLoader loading story_line json file: ', path_story_line_json)
        self.story_line = json.load(open(path_story_line_json))

        self.id2word = self.story_line['id2words']
        self.word2id = self.story_line['words2id']
        self.vocab_size = len(self.id2word)
        print('vocab size is ', self.vocab_size)
        self.story_ids = {'train': [], 'val': [], 'test': []}
        self.story_ids['train'] = list(self.story_line['train'].keys())
        self.story_ids['val'] = list(self.story_line['val'].keys())
        self.story_ids['test'] = list(self.story_line['test'].keys())

        print('There are {} training data, {} validation data, and {} test data'.format(len(self.story_ids['train']),
                                                                                len(self.story_ids['val']),
                                                                                len(self.story_ids['test'])))

        # write reference files for storytelling
        if not os.path.exists(path_ref_dir):
            os.makedirs(path_ref_dir)
        
        # mode 1
        for split in ['train', 'val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                if story['album_id'] not in reference:
                    reference[story['album_id']] = [story['origin_text']]
                else:
                    reference[story['album_id']].append(story['origin_text'])
            with open(os.path.join(path_ref_dir, '{}_reference_m1.json'.format(split)), 'w') as f:
                json.dump(reference, f)
        # mode 2
        for split in ['train', 'val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                fid = '_'.join(story['flickr_id'])
                if fid not in reference:
                    reference[fid] = [story['origin_text']]
                else:
                    reference[fid].append(story['origin_text'])
            if split == 'train':
                self.train_reference = reference
            with open(os.path.join(path_ref_dir, '{}_reference_m2.json'.format(split)), 'w') as f:
                json.dump(reference, f)
        
        # # lda
        # # generate keywords
        # stop_word_file = os.path.join(path_ref_dir, 'stopwords_lda.txt')
        # self.init_keywords_lda(stop_word_file, num_topics)
        # print('generate keywords')
        # load
        print('load keywords')
        self.keywords_dist = {} # topic-word distrubution
        self.keywords = {} # doc-topic distribution
        for split in ['train', 'val', 'test']:
            self.keywords_dist[split] = np.load(os.path.join(path_ref_dir, 'keywords_' + str(num_topics), '{}_keywords_dist.npy'.format(split)))
            self.keywords[split] = np.load(os.path.join(path_ref_dir, 'keywords_' + str(num_topics), '{}_keywords.npy'.format(split))) # 5050*5*16

        # load id2pos
        self.nlp = spacy.load('en_core_web_sm')
        self.pos_set = ["$", "``", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX", "_SP"]
        self.pos_hamming = [""''"", "', '", "-", "-LRB-", "-RRB-", "ADD", "AFX", "GW", "HYPH", "SP", "SYM", "XX", "_SP", "NN", "DT", ".,", ".", "IN", "VBD", "JJ", "NNS", "PRP", "RB", "VB", ",", ", ", "TO", "CC", "VBG", "PRP$", "VBZ", "VBN", "VBP", "RP", "MD", "CD", "EX", "WRB", "POS", "WP", "WDT", "JJS", "JJR", "NNP", "PDT", "RBR", ":", "``", "RBS", "UH", ")", "FW", "(", "$", "''", "#"]
        # load id2pos
        print('load id2pos')
        with open(os.path.join(path_ref_dir, 'id2pos.json'), "r") as id2pos_file:
            self.id2pos = json.load(id2pos_file)
        # # save id2pos to json
        # self.id2pos = {}
        # for i in range(self.vocab_size):
        #     word = self.id2word[str(i)]
        #     token = self.nlp(word)[0]
        #     self.id2pos[i] = self.pos_set.index(token.tag_)
        # id2pos_json = json.dumps(self.id2pos)
        # f = open(os.path.join(path_ref_dir, 'id2pos.json'), "w")
        # f.write(id2pos_json)
        # f.close()

    def __getitem__(self, index):
        story_id = self.story_ids[self.mode][index]
        story = self.story_line[self.mode][story_id]

        # # load img
        # imgs = torch.zeros((story['length'], 3, 224, 224), dtype = torch.float32)
        # for i in range(story['length']):
        #     path_img = os.path.join(path_img_dir, self.mode, '{}.jpg'.format(story['flickr_id'][i]))
        #     img = Image.open(path_img)
        #     img = preprocess(img)
        #     imgs[i] = img
        # sample = {'imgs': imgs}

        # load feature
        feature_fc = np.zeros((story['length'], feat_size), dtype='float32')
        for i in range(story['length']):
            fc_path = os.path.join(path_resnet_features, 'fc', self.mode, '{}.npy'.format(story['flickr_id'][i]))
            feature_fc[i] = np.load(fc_path)
        sample = {'feature_fc': feature_fc}

        # load story
        split_story = self.story_h5[story['text_index']] # split story - 5*30; whole story - 1*?
        sample['split_story'] = np.int64(split_story)

        sample['index'] = np.int64(index)

        fid = '_'.join(story['flickr_id'])
        sample['fid'] = fid

        # load lda keywords distribution
        sample['keywords'] = self.keywords[self.mode][index, :, :].astype(np.float32)

        return sample

    def get_by_fid(self, _fid):
        counter = 0
        for index, story in enumerate(self.story_line['test'].values()):
            fid = '_'.join(story['flickr_id'])
            if fid == _fid:
                if counter >= 1:
                    # feature
                    feature_fc = np.zeros((story['length'], feat_size), dtype='float32')
                    for i in range(story['length']):
                        fc_path = os.path.join(path_resnet_features, 'fc', 'test', '{}.npy'.format(story['flickr_id'][i]))
                        feature_fc[i] = np.load(fc_path)
                    sample = {'feature_fc': feature_fc}
                    # gt
                    split_story = self.story_h5[story['text_index']] # split story - 5*30; whole story - 1*?
                    sample['split_story'] = np.int64(split_story)
                    # load lda keywords distribution
                    sample['keywords'] = self.keywords['test'][index, :, :].astype(np.float32)

                    return sample
                else:
                    counter += 1

    def __len__(self):
        return len(self.story_ids[self.mode])
    
    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def test(self):
        self.mode = 'test'
    
    def get_GT(self, index):
        """ get GT storys by batch index, used by criterion's rl reward
            train_reference is mode 2
        """

        story_id = self.story_ids[self.mode][index]
        story = self.story_line[self.mode][story_id]
        fid = '_'.join(story['flickr_id'])
        return self.train_reference[fid]
       
    def get_aid(self, index):
        """ get album_id by batch index
        """

        story_id = self.story_ids[self.mode][index]
        return self.story_line[self.mode][story_id]['album_id']

    def get_fid(self, index):
        """ get joint flickr_id by batch index
        """

        story_id = self.story_ids[self.mode][index]
        return '_'.join(self.story_line[self.mode][story_id]['flickr_id'])

    def get_all_id(self, index):
        story_id = self.story_ids[self.mode][index]
        return self.story_line[self.mode][story_id]['album_id'], self.story_line[self.mode][story_id]['flickr_id']

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.id2word

    def get_word2id(self):
        return self.word2id

    def get_whole_story_length(self):
        return self.full_story_h5.shape[1]

    def get_story_length(self):
        return self.story_h5.shape[1]

    def get_caption_length(self):
        return self.desc_h5.shape[1]

    def pos_ifhamming(self, index):
        """ if index in vocab is in pos hamming list
        """

        return self.pos_set[self.id2pos[str(index)]] in self.pos_hamming

    def init_keywords_lda(self, stop_word_file, num_topics):
        import lda
        import nltk

        emptyvocab = [0 for _ in range(self.vocab_size)]

        # stopwords
        stop_words = []
        for line in open(stop_word_file):
            if line.strip()[0:1] != "#":
                word = line.split()
                if len(word) == 0:
                    continue
                word = word[0]
                if word in self.word2id:
                    stop_words.append(self.word2id[word])

        for split in ['val', 'train', 'test']:
            docs = []
            for story_id, story in self.story_line[split].items():
                story_text = story['origin_text'] # text
                split_story = self.story_h5[story['text_index']] # ints, 5*30

                for j in range(5):
                    tokens = split_story[j, :]
                    freq = nltk.FreqDist(tokens)
                    doc_dict = dict(freq)

                    x = emptyvocab.copy() # vocab_size*1
                    for k, v in doc_dict.items():
                        if k not in stop_words:
                            x[k] = v

                    docs.append(x)
            
            docs = np.asarray(docs)

            model = lda.LDA(n_topics = num_topics, n_iter=1000, random_state=1)
            model.fit(docs)
            topic_word = model.topic_word_ # topic-word distribution
            doc_topic = model.doc_topic_ # doc-topic distribution
            doc_topic = doc_topic.reshape(len(self.story_ids[split]), 5, -1)

            n_top_words = 100
            keywords_dist = []
            # topic_words_all = {}
            for i, topic_dist in enumerate(topic_word):
                indices = np.argsort(topic_dist)[:-(n_top_words+1):-1]
                keywords_dist.append(indices)

                topic_words = []
                for ind in indices:
                    topic_words.append(self.id2word[str(ind)])
                print('Topic {}: {}'.format(i+1, ' '.join(topic_words)))
                # topic_words_all[i+1] = topic_words
            keywords_dist = np.asarray(keywords_dist, dtype = np.uint64)
            # with open('topic16_test.json', 'w') as f:
            #     json.dump(topic_words_all, f)

            # np.savetxt(os.path.join(path_ref_dir, '{}_keywords_dist.out'.format(split)), keywords_dist, fmt = '%d')
            # np.savetxt(os.path.join(path_ref_dir, '{}_keywords.out'.format(split)), doc_topic, fmt = '%.4f')
            with open(os.path.join(path_ref_dir, 'keywords_' + str(num_topics), '{}_keywords_dist.npy'.format(split)), 'wb') as f:
                np.save(f, keywords_dist)
            with open(os.path.join(path_ref_dir, 'keywords_' + str(num_topics), '{}_keywords.npy'.format(split)), 'wb') as f:
                np.save(f, doc_topic)


### preprocess ###
# remove album that has empty img in story_line
def vist_preprocess():

    # open the hdf5 file
    print('DataLoader loading story h5 file: ', path_story_h5)
    story_h5 = h5py.File(path_story_h5, 'r', driver='core')['story']
    print("story's max sentence length is ", story_h5.shape[1])

    print('DataLoader loading story_line json file: ', path_story_line_json)
    story_line = json.load(open(path_story_line_json))

    id2word = story_line['id2words']
    word2id = story_line['words2id']
    vocab_size = len(id2word)
    print('vocab size is ', vocab_size)
    story_ids = {'train': [], 'val': [], 'test': []}
    description_ids = {'train': [], 'val': [], 'test': []}


    story_ids['train'] = list(story_line['train'].keys())
    story_ids['val'] = list(story_line['val'].keys())
    story_ids['test'] = list(story_line['test'].keys())
    description_ids['train'] = list(story_line['image2caption']['train'].keys())
    description_ids['val'] = list(story_line['image2caption']['val'].keys())
    description_ids['test'] = list(story_line['image2caption']['test'].keys())

    print('There are {} training data, {} validation data, and {} test data'.format(len(story_ids['train']),
                                                                            len(story_ids['val']),
                                                                            len(story_ids['test'])))


    # get non-RGB jpgs
    # outers = {}
    for mode in ['val', 'test', 'train']:
        for s_id in story_ids[mode]:
            story = story_line[mode][s_id]

            for i in range(story['length']):
                path_img = os.path.join(path_img_dir, mode, '{}.jpg'.format(story['flickr_id'][i]))

                # image not exist
                if not os.path.isfile(path_img):
                    del story_line[mode][s_id]
                    break
                
                img = Image.open(path_img)

                # image not RGB
                if img.mode != 'RGB':
                    # print(story['flickr_id'], story['flickr_id'][i])
                    del story_line[mode][s_id]
                    # outers[s_id] = img.mode
                    break

    # with open(os.path.join('not_RGB_images.json'), 'w') as f:
    #     json.dump(outers, f)

    # with open('./datasets/VIST/story_line.json', 'w') as f:
    #     json.dump(story_line, f)


    id2word = story_line['id2words']
    word2id = story_line['words2id']
    vocab_size = len(id2word)
    print('vocab size is ', vocab_size)
    story_ids = {'train': [], 'val': [], 'test': []}
    description_ids = {'train': [], 'val': [], 'test': []}


    story_ids['train'] = list(story_line['train'].keys())
    story_ids['val'] = list(story_line['val'].keys())
    story_ids['test'] = list(story_line['test'].keys())
    description_ids['train'] = list(story_line['image2caption']['train'].keys())
    description_ids['val'] = list(story_line['image2caption']['val'].keys())
    description_ids['test'] = list(story_line['image2caption']['test'].keys())

    print('There are {} training data, {} validation data, and {} test data'.format(len(story_ids['train']),
                                                                            len(story_ids['val']),
                                                                            len(story_ids['test'])))

### resnet152 fc features ###
def resfc():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = pre_trained_models.resnet152(pretrained=True)

    # remove fc
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    resnet = resnet.to(device)
    resnet.eval()

    # open the hdf5 file
    print('DataLoader loading story h5 file: ', path_story_h5)
    story_h5 = h5py.File(path_story_h5, 'r', driver='core')['story']
    print("story's max sentence length is ", story_h5.shape[1])

    print('DataLoader loading story_line json file: ', path_story_line_json)
    story_line = json.load(open(path_story_line_json))

    id2word = story_line['id2words']
    word2id = story_line['words2id']
    vocab_size = len(id2word)
    print('vocab size is ', vocab_size)
    story_ids = {'train': [], 'val': [], 'test': []}
    description_ids = {'train': [], 'val': [], 'test': []}


    story_ids['train'] = list(story_line['train'].keys())
    story_ids['val'] = list(story_line['val'].keys())
    story_ids['test'] = list(story_line['test'].keys())
    description_ids['train'] = list(story_line['image2caption']['train'].keys())
    description_ids['val'] = list(story_line['image2caption']['val'].keys())
    description_ids['test'] = list(story_line['image2caption']['test'].keys())

    print('There are {} training data, {} validation data, and {} test data'.format(len(story_ids['train']),
                                                                            len(story_ids['val']),
                                                                            len(story_ids['test'])))

    for mode in ['test', 'train']:
        print(mode)
        for s_id in story_ids[mode]:
            story = story_line[mode][s_id]

            for i in range(story['length']):
                path_img = os.path.join(path_img_dir, mode, '{}.jpg'.format(story['flickr_id'][i]))
                img = Image.open(path_img)
                img = preprocess(img).unsqueeze(0)

                img = img.to(device)
                feature_fc = resnet(img).squeeze()
                feature_fc = np.array(feature_fc.cpu().detach())
                np.save(os.path.join(path_resnet_features, 'fc', mode, '{}.npy'.format(story['flickr_id'][i])), feature_fc)


###
# resfc()

# VISTDataset(num_topics = 16)