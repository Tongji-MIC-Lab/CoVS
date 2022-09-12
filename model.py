from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import logging
import time
import numpy as np

story_size = 5
word_embed_dim = 512 
hidden_dim = 512
encoder_rnn_type = 'gru' # 'gru, lstm, bilstm_attn, bi-gru'
decoder_rnn_type = 'gru'
encoder_num_layers = 1
decoder_num_layers = 2
visual_dropout = 0.2
decoder_dropout = 0.5
topicModel_dropout = 0.5
feat_size = 2048 # 2048 for resnet, 4096 for vgg
with_position = False # 'whether to use position embedding for the image feature' word

def _smallest(matrix, k, only_first_row=False):
    """ select smallest #beam_size costs in beam search matrix

    Args:
        matrix: beam_size*vocab_size
        k: beam_size
        only_first_row: True for seq's first step, all beams' probs the same

    Returns:
        indices: beam_size*1, beam index
        outputs: beam_size*1, vocab index
        chosen_costs: beam_size*1, cost values
    """ 

    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k] # beam_size*1, index of quicksort partition result
    args = args[np.argsort(flatten[args])] # beam_size*1, sorted index
    indices, outputs = np.unravel_index(args, matrix.shape)
    chosen_costs = flatten[args]
    return (indices, outputs), chosen_costs

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = feat_size
        self.embed_dim = word_embed_dim
        # rnn layer options
        self.rnn_type = encoder_rnn_type
        self.num_layers = encoder_num_layers
        self.hidden_dim = hidden_dim
        self.dropout = visual_dropout
        self.story_size = story_size
        self.with_position = with_position

        # visual embedding layer
        self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))

        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # max elementwise pooling
        self.elepool_fc = nn.Linear(self.embed_dim, self.embed_dim)

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU(True)

        if self.with_position: # bool
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else: # lstm
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def forward(self, input, hidden=None):
        """
        Args:
            input:  (batch_size, 5, feat_size) 64*5*2048
            hidden: (num_layers * num_dirs, batch_size, hidden_dim // 2)
        
        Returns:
            out: (batch_size, 5, rnn_size), serve as context 64*5*512
        """

        batch_size = input.size(0)

        # visual embeded
        emb = self.visual_emb(input.view(-1, self.feat_size)) # 320*512
        emb = emb.view(batch_size, self.story_size, -1)  # 64*5*512

        # # max elementwise pooling
        # emb = self.elepool_fc(emb) # 64*5*512
        # emb, _ = torch.max(emb, dim = 1) # 64*512
        # emb = emb.unsqueeze(1).expand(-1, self.story_size, -1) # 64*5*512
        # simple average
        emb = torch.mean(emb, 1, True).expand(-1, self.story_size, -1) # 64*5*512

        # visual rnn layer
        if hidden is None:
            hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        rnn_input = self.hin_dropout_layer(emb)
        houts, hidden = self.rnn(rnn_input, hidden) # 64*5*512

        # residual layer
        out = emb + self.project_layer(houts)
        out = self.relu(out)  # (batch_size, 5, embed_dim)
        # out = houts  # gjj-change

        if self.with_position:
            for i in range(self.story_size):
                position = Variable(input.data.new(batch_size).long().fill_(i))
                out[:, i, :] = out[:, i, :] + self.position_embed(position)

        return out, hidden

# ### bottom-up top-down attention ###
# class BuTdModel(nn.Module):
#     def __init__(self, dataset, vocab_size, seq_length, num_topics, keywords_dist, min_length, hamming_diversity, hamming_f, hamming_n):
#         super(BuTdModel, self).__init__()
#         self.dataset = dataset
#         self.vocab_size = vocab_size
#         self.story_size = story_size
#         self.word_embed_dim = word_embed_dim # 512
#         self.hidden_dim = hidden_dim # 512
#         self.dropout = decoder_dropout
#         self.seq_length = seq_length # 30
#         self.feat_size = feat_size
#         self.decoder_input_dim = self.word_embed_dim + self.word_embed_dim # 512*2
#         self.ss_prob = 0.0  # Schedule sampling probability
#         self.min_length = min_length
#         self.hamming_diversity = hamming_diversity
#         self.hamming_f = hamming_f
#         self.hamming_n = hamming_n
#         self.num_topics = num_topics
#         self.keywords_dist = keywords_dist

#         # Visual Encoder
#         self.encoder = VisualEncoder()

#         # word embedding layer
#         self.embed = nn.Embedding(self.vocab_size, self.word_embed_dim) # Embeddin matrix of vocab_size*512

#         # attention
#         self.image_attentionRNN = nn.LSTM(input_size=self.hidden_dim*3, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
#         self.topic_attentionRNN = nn.LSTM(input_size=self.hidden_dim*3, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

#         self.image_attn = nn.Sequential(nn.ReLU(),
#                                     nn.Dropout(p=self.dropout),
#                                     nn.Linear(self.hidden_dim * 2, self.story_size))
#         self.topic_attn = nn.Sequential(nn.ReLU(),
#                                     nn.Dropout(p=self.dropout),
#                                     nn.Linear(self.hidden_dim * 2, self.story_size))

#         self.languageRNN = nn.LSTM(input_size=self.hidden_dim * 4, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

#         # last linear layer
#         self.logit = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2),
#                                    nn.Tanh(),
#                                    nn.Dropout(p=self.dropout),
#                                    nn.Linear(self.hidden_dim // 2, self.vocab_size))

#         self.init_ha_proj = nn.Linear(self.feat_size, self.hidden_dim)
#         self.init_ca_proj = nn.Linear(self.feat_size, self.hidden_dim)
#         self.init_ha2_proj = nn.Linear(self.feat_size, self.hidden_dim)
#         self.init_ca2_proj = nn.Linear(self.feat_size, self.hidden_dim)
#         self.init_hl_proj = nn.Linear(self.feat_size, self.hidden_dim)
#         self.init_cl_proj = nn.Linear(self.feat_size, self.hidden_dim)

#         # topic project
#         self.topic_proj = nn.Sequential(nn.Linear(self.num_topics, self.hidden_dim),
#                                         nn.Tanh())

#         self.baseline_estimator = nn.Linear(self.hidden_dim * 2, 1)

#         self.init_weights(0.1)

#     def init_weights(self, init_range):
#         logging.info("Initialize the parameters of the model")
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data.uniform_(-init_range, init_range)
#                 if not m.bias is None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Embedding):
#                 m.weight.data.uniform_(-init_range, init_range)

#     def init_hidden_with_img(self, img):
#         """ init hidden state with un-encoded img features
#         """

#         output1 = self.init_ha_proj(img.expand(1, -1, -1))
#         output2 = self.init_ca_proj(img.expand(1, -1, -1))
#         output3 = self.init_hl_proj(img.expand(1, -1, -1))
#         output4 = self.init_cl_proj(img.expand(1, -1, -1))

#         return output1, output2, output3, output4

#     def init_hidden_with_img_2(self, img):
#         """ init hidden state with un-encoded img features
#         """

#         output1 = self.init_ha_proj(img.expand(1, -1, -1))
#         output2 = self.init_ca_proj(img.expand(1, -1, -1))
#         output12 = self.init_ha2_proj(img.expand(1, -1, -1))
#         output22 = self.init_ca2_proj(img.expand(1, -1, -1))
#         output3 = self.init_hl_proj(img.expand(1, -1, -1))
#         output4 = self.init_cl_proj(img.expand(1, -1, -1))

#         return output1, output2, output12, output22, output3, output4

#     def decode(self, imgs_emb, last_word, keywords_emb, image_ha, image_ca, topic_ha, topic_ca, hl, cl, ind, penalize_previous=False):
#         """
#         Args:
#             imgs_emb: batch_size*5*512
#             keywords_emb: batch_size*5*512
#             last_word: Variable contraining a word index, batch_size*1
#             ha, ca: attention RNN hidden state
#             hl, cl: language RNN hidden state
#             ind: sentence index [0, 5)
#         """

#         word_emb = self.embed(last_word) # 64*512

#         # attention
#         image_input_a = torch.cat((hl.squeeze(0), imgs_emb.mean(1, False), word_emb), 1).unsqueeze(1) # 64*1*(512*3)
#         image_output_a, (image_ha, image_ca) = self.image_attentionRNN(image_input_a, (image_ha, image_ca)) # 64*1*512

#         image_attn_weights = F.softmax(self.image_attn(torch.cat((imgs_emb[:, ind, :], torch.squeeze(image_output_a)), 1)), dim = 1) # batch_size*5
#         image_attn_applied = torch.bmm(image_attn_weights.unsqueeze(1), imgs_emb) # batch_size*1*512

#         topic_input_a = torch.cat((hl.squeeze(0), keywords_emb.mean(1, False), word_emb), 1).unsqueeze(1) # 64*1*(512*3)
#         topic_output_a, (topic_ha, topic_ca) = self.topic_attentionRNN(topic_input_a, (topic_ha, topic_ca)) # 64*1*512

#         topic_attn_weights = F.softmax(self.topic_attn(torch.cat((keywords_emb[:, ind, :], torch.squeeze(topic_output_a)), 1)), dim = 1) # batch_size*5
#         topic_attn_applied = torch.bmm(topic_attn_weights.unsqueeze(1), keywords_emb) # batch_size*1*512

#         input_l = torch.cat((image_attn_applied, image_output_a, topic_output_a, topic_attn_applied), 2)
#         output_l, (hl, cl) = self.languageRNN(input_l, (hl, cl)) # 64*1*512

#         log_probs = F.log_softmax(self.logit(output_l[:, 0, :]), dim = 1) # 64*vocab_size

#         if penalize_previous: # do not gen the same word as last word
#             last_word_onehot = torch.FloatTensor(last_word.size(0), self.vocab_size).zero_().cuda()
#             penalize_value = (last_word > 0).data.float() * -100
#             mask = Variable(last_word_onehot.scatter_(1, last_word.data[:, None], 1.) * penalize_value[:, None])
#             log_probs = log_probs + mask

#         return word_emb, log_probs, image_ha, image_ca, topic_ha, topic_ca, hl, cl

#     def forward(self, imgs, keywords, caption):
#         """ gen sentences with teacher forcing

#         Args:
#             imgs: (batch_size, 5, feat_size) 64*5*2048
#             keywords: batch_size*5*keyword_num
#             caption: (batch_size, 5, seq_length) 64*5*30
#         """

#         batch_size = imgs.shape[0]

#         ########################### encoding stage #############################
        
#         imgs_emb, _ = self.encoder(imgs) # 64*5*512

#         ########################### decoding stage #############################

#         outputs = torch.FloatTensor(batch_size, self.story_size, self.seq_length, self.vocab_size).zero_().cuda() # 64*5*30*vocab_size

#         keywords_emb = self.topic_proj(keywords) # batch_size*5*512

#         for j in range(self.story_size):
#             last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda() # 64*1, <EOS>
            
#             ha, ca, ha2, ca2, hl, cl = self.init_hidden_with_img_2(imgs[:, j, :])
           
#             for i in range(self.seq_length):
#                 word_emb, log_probs, ha, ca, ha2, ca2, hl, cl = self.decode(imgs_emb, last_word, keywords_emb, ha, ca, ha2, ca2, hl, cl, j) # 64*vocab_size, 1*64*512

#                 outputs[:, j, i, :] = log_probs

#                 # ss choose the word
#                 if self.ss_prob > 0.0: # default range [0, 0.25]
#                     sample_prob = torch.FloatTensor(batch_size).uniform_(0, 1).cuda() # 64*1
#                     sample_mask = sample_prob < self.ss_prob # 64*1
#                     if sample_mask.sum() == 0:
#                         last_word = caption[:, j, i].clone() # use GT's final word as lastword of sentence
#                     else:
#                         sample_ind = sample_mask.nonzero().view(-1) # nonzero index
#                         last_word = caption[:, j, i].data.clone()
#                         # fetch prev distribution: shape Nx(M+1)
#                         prob_prev = torch.exp(log_probs.data) # 64*vocab_size
#                         last_word.index_copy_(0, sample_ind, # select the highest prob word in sample_ind indices
#                                             torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
#                         last_word = Variable(last_word)
#                 else:
#                     last_word = caption[:, j, i].clone()

#                 # break when whole batch's sentences end
#                 if i >= 1 and caption[:, j, i].data.sum() == 0:
#                     break

#         return outputs

#     def sample(self, imgs, keywords):
#         """ gen sentences by sampling

#         Args:
#             imgs: batch_size*5*2048
#             seq, seq_log_probs: output story, each word selected by prb
#         """

#         batch_size = imgs.shape[0]

#         ########################### encoding stage #############################
#         imgs_emb, _ = self.encoder(imgs) # batch_size*5*512

#         ########################### decoding stage #############################

#         seq = torch.LongTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()
#         seq_log_probs = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()
#         baseline = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()

#         keywords_emb = self.topic_proj(keywords)

#         for j in range(self.story_size):
#             last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda() # 64*1, <EOS>
#             ha, ca, ha2, ca2, hl, cl = self.init_hidden_with_img_2(imgs[:, j, :])
            
#             for i in range(self.seq_length):
#                 word_emb, log_probs, ha, ca, ha2, ca2, hl, cl = self.decode(imgs_emb, last_word, keywords_emb, ha, ca, ha2, ca2, hl, cl, j)
                
#                 if i < self.min_length: # make <EOS> not appear in first min_length words
#                     mask = np.zeros((batch_size, log_probs.size(-1)), 'float32') # 64*vocab_size
#                     mask[:, 0] = -1000
#                     mask = Variable(torch.from_numpy(mask)).cuda()
#                     log_probs = log_probs + mask

#                 # 'sample' select by nomial
#                 # fetch prev distribution: shape Nx(M+1)
#                 prob_prev = torch.exp(log_probs.data).cpu()
#                 last_word = torch.multinomial(prob_prev, 1).cuda() # 64*1
#                 # gather the logprobs at sampled positions
#                 sample_log_prob = log_probs.gather(1, Variable(last_word))
#                 # flatten indices for downstream processing
#                 last_word = last_word.view(-1).long()

#                 if (last_word > 0).sum() == 0 and i >= 1:
#                     break

#                 seq[:, j, i] = last_word  # 64*1
#                 seq_log_probs[:, j, i] = sample_log_prob.view(-1) # 64*1

#                 # NN as baseline, actor-critic
#                 value = self.baseline_estimator(torch.cat((hl.squeeze(0).detach(), cl.squeeze(0).detach()), dim = 1)) # batch_size*1
#                 baseline[:, j, i] = value.view(-1)

#         return seq, seq_log_probs, baseline

#     def sample_greedy(self, imgs, keywords):
#         """ gen sentences by greedy search

#         Args:
#             imgs: 64*5*2048
#             seq, seq_log_probs: output story, each word selected by prb
#         """
#         batch_size = imgs.shape[0]

#         ########################### encoding stage #############################
#         imgs_emb, _ = self.encoder(imgs) # 64*5*512

#         ########################### decoding stage #############################

#         seq = torch.LongTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()
#         seq_log_probs = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()

#         keywords_emb = self.topic_proj(keywords)

#         for j in range(self.story_size):
#             last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda() # 64*1, <EOS>
#             ha, ca, ha2, ca2, hl, cl = self.init_hidden_with_img_2(imgs[:, j, :])
            
#             for i in range(self.seq_length):
#                 word_emb, log_probs, ha, ca, ha2, ca2, hl, cl = self.decode(imgs_emb, last_word, keywords_emb, ha, ca, ha2, ca2, hl, cl, j)
                
#                 if i < self.min_length: # make <EOS> not appear in first min_length words
#                     mask = np.zeros((batch_size, log_probs.size(-1)), 'float32') # 64*vocab_size
#                     mask[:, 0] = -1000
#                     mask = Variable(torch.from_numpy(mask)).cuda()
#                     log_probs = log_probs + mask

#                 # greedy select max prob
#                 sample_log_prob, last_word = torch.max(log_probs, 1) # 64*1, 64*1
#                 last_word = last_word.data.view(-1).long()

#                 if (last_word > 0).sum() == 0 and i >= 1:
#                     break

#                 seq[:, j, i] = last_word  # 64*1
#                 seq_log_probs[:, j, i] = sample_log_prob.view(-1) # 64*1

#         return seq, seq_log_probs

#     def predict(self, imgs, keywords, beam_size=5):
#         """ gen sentences with beam search

#         Args:
#             imgs: 64*5*2048
#             keywords: batch_size*5*num_topics
#         """

#         assert beam_size <= self.vocab_size and beam_size > 0
#         if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
#             return self.sample_greedy(imgs, keywords)

#         batch_size = imgs.shape[0]

#         # encode the visual features
#         imgs_emb, _ = self.encoder(imgs) # 64*5*512

#         ####################### decoding stage ##################################

#         seq = torch.LongTensor(batch_size, self.story_size, self.seq_length).zero_()
#         seq_log_probs = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_()

#         keywords_emb = self.topic_proj(keywords)

#         # lets process the videos independently for now, for simplicity
#         for k in range(batch_size):
#             imgs_emb_k = imgs_emb[k, :, :].unsqueeze(0).expand(beam_size, imgs_emb.size(1), imgs_emb.size(2)).contiguous() # beam_size*5*512
#             keywords_emb_k = keywords_emb[k, :, :].unsqueeze(0).expand(beam_size, keywords_emb.size(1), keywords_emb.size(2))

#             # hamming diversity
#             if self.hamming_diversity:
#                 # n-gram freq
#                 hamming = {}

#             for j in range(self.story_size):
#                 last_word = Variable(torch.FloatTensor(beam_size).long().zero_().cuda()) # beam_size*1, <EOS>

#                 ha, ca, ha2, ca2, hl, cl = self.init_hidden_with_img_2(imgs[:, j, :])
#                 ha_k = ha[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
#                 ca_k = ca[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
#                 hl_k = hl[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
#                 cl_k = cl[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
#                 ha2_k = ha2[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
#                 ca2_k = ca2[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()

#                 all_outputs = np.ones((1, beam_size), dtype='int32') # 1*beam_size
#                 all_masks = np.ones_like(all_outputs, dtype="float32")
#                 all_costs = np.zeros_like(all_outputs, dtype="float32")
                
#                 for i in range(self.seq_length):
#                     if all_masks[-1].sum() == 0:
#                         break

#                     ### decode ###
#                     word_emb, log_probs, ha_k, ca_k, ha2_k, ca2_k, hl_k, cl_k = self.decode(imgs_emb_k, last_word, keywords_emb_k, ha_k, ca_k, ha2_k, ca2_k, hl_k, cl_k, j, True)

#                     # hamming diversity
#                     if self.hamming_diversity and i > self.hamming_n - 2:
#                         for tmpb in range(beam_size):
#                             if self.hamming_n == 1:
#                                 for tmpk, tmpv in hamming.items():
#                                     log_probs[tmpb, int(tmpk)] -= self.hamming_f * tmpv
#                             else:
#                                 tmpoutput = all_outputs[-(self.hamming_n-1):, tmpb]
#                                 tmpoutput = [str(_) for _ in tmpoutput]
#                                 tmpngram = ','.join(tmpoutput)
#                                 for tmpk, tmpv in hamming.items():
#                                     tmpkl = tmpk.split(',')
#                                     if ','.join(tmpkl[:-1]) == tmpngram:
#                                         log_probs[tmpb, int(tmpkl[-1])] -= self.hamming_f * tmpv

#                         # log_probs -= self.hamming_f * hamming
                    
#                     log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token

#                     neg_log_probs = -log_probs

#                     ### beam search, log_pros are added up every step ###

#                     # all_costs[-1, :, None] equals to all_costs[-1, :].unsqueeze(1)
#                     next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None]) # beam_size*vocab_size
#                     (finished,) = np.where(all_masks[-1] == 0)
#                     next_costs[finished, 1:] = np.inf

#                     (indices, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)

#                     all_outputs = all_outputs[:, indices]
#                     all_masks = all_masks[:, indices]
#                     all_costs = all_costs[:, indices]

#                     all_outputs = np.vstack([all_outputs, outputs[None, :]])
#                     all_costs = np.vstack([all_costs, chosen_costs[None, :]])
#                     mask = outputs != 0
#                     all_masks = np.vstack([all_masks, mask[None, :]])

#                     last_word = Variable(torch.from_numpy(outputs)).cuda()

#                     ha_k = Variable(torch.from_numpy(ha_k.data.cpu().numpy()[:, indices, :])).cuda()
#                     ca_k = Variable(torch.from_numpy(ca_k.data.cpu().numpy()[:, indices, :])).cuda()
#                     hl_k = Variable(torch.from_numpy(hl_k.data.cpu().numpy()[:, indices, :])).cuda()
#                     cl_k = Variable(torch.from_numpy(cl_k.data.cpu().numpy()[:, indices, :])).cuda()
#                     ha2_k = Variable(torch.from_numpy(ha2_k.data.cpu().numpy()[:, indices, :])).cuda()
#                     ca2_k = Variable(torch.from_numpy(ca2_k.data.cpu().numpy()[:, indices, :])).cuda()

#                 all_outputs = all_outputs[1:] # seq_length*beam_size, all_outputs[0] are 1
#                 all_costs = all_costs[1:] - all_costs[:-1] # restore step costs
#                 all_masks = all_masks[:-1]
#                 costs = all_costs.sum(axis=0) # beam_size*1
#                 lengths = all_masks.sum(axis=0)
#                 normalized_cost = costs / lengths
#                 best_idx = np.argmin(normalized_cost)
#                 seq[k, j, :all_outputs.shape[0]] = torch.from_numpy(all_outputs[:, best_idx])
#                 seq_log_probs[k, j, :all_outputs.shape[0]] = torch.from_numpy(all_costs[:, best_idx])
                
#                 # update previous sentences' hamming freq with beam search result
#                 if self.hamming_diversity:
#                     tmpoutput = all_outputs[:, best_idx]
#                     tmpoutput = [str(_) for _ in tmpoutput]
#                     if self.hamming_n == 1:
#                         for token in tmpoutput:
#                             if token == '0':
#                                 break
#                             # if self.dataset.pos_ifhamming(token):
#                             if token not in hamming:
#                                 hamming[token] = 1
#                             else:
#                                 hamming[token] += 1
#                     else:
#                         for tmpi in range(len(tmpoutput) - self.hamming_n + 1):
#                             tmpngram = ','.join(tmpoutput[tmpi:tmpi + self.hamming_n])
#                             if tmpngram not in hamming:
#                                 hamming[tmpngram] = 1
#                             else:
#                                 hamming[tmpngram] += 1

#         # return the samples and their log likelihoods
#         seq = seq.contiguous()
#         seq_log_probs = seq_log_probs.contiguous()

#         return seq, seq_log_probs # 64*5*30, 64*5*30

# GRU
class BuTdModel(nn.Module):
    def __init__(self, dataset, vocab_size, seq_length, num_topics, keywords_dist, min_length, hamming_diversity, hamming_f, hamming_n):
        super(BuTdModel, self).__init__()
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.story_size = story_size
        self.word_embed_dim = word_embed_dim # 512
        self.hidden_dim = hidden_dim # 512
        self.dropout = decoder_dropout
        self.seq_length = seq_length # 30
        self.feat_size = feat_size
        self.decoder_input_dim = self.word_embed_dim + self.word_embed_dim # 512*2
        self.ss_prob = 0.0  # Schedule sampling probability
        self.min_length = min_length
        self.hamming_diversity = hamming_diversity
        self.hamming_f = hamming_f
        self.hamming_n = hamming_n
        self.num_topics = num_topics
        self.keywords_dist = keywords_dist

        # Visual Encoder
        self.encoder = VisualEncoder()

        # word embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.word_embed_dim) # Embeddin matrix of vocab_size*512

        # attention
        self.image_attentionRNN = nn.GRU(input_size=self.hidden_dim*3, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.topic_attentionRNN = nn.GRU(input_size=self.hidden_dim*3, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

        self.image_attn = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(self.hidden_dim * 2, self.story_size))
        self.topic_attn = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(self.hidden_dim * 2, self.story_size))

        self.languageRNN = nn.GRU(input_size=self.hidden_dim * 4, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        # self.languageRNN = nn.GRU(input_size=self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

        # last linear layer
        self.logit = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                   nn.Tanh(),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(self.hidden_dim // 2, self.vocab_size))

        self.init_ha_proj = nn.Linear(self.feat_size, self.hidden_dim)
        self.init_ha2_proj = nn.Linear(self.feat_size, self.hidden_dim)
        self.init_hl_proj = nn.Linear(self.feat_size, self.hidden_dim)

        # topic project
        self.topic_proj = nn.Sequential(nn.Linear(self.num_topics, self.hidden_dim),
                                        nn.Tanh())

        # self.baseline_estimator = nn.Linear(self.hidden_dim * 3, 1)

        self.init_weights(0.1)

    def init_weights(self, init_range):
        logging.info("Initialize the parameters of the model")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-init_range, init_range)

    def init_hidden_with_img(self, img):
        """ init hidden state with un-encoded img features
        """

        output1 = self.init_ha_proj(img.expand(1, -1, -1))
        output3 = self.init_hl_proj(img.expand(1, -1, -1))

        return output1, output3

    def init_hidden_with_img_2(self, img):
        """ init hidden state with un-encoded img features
        """

        output1 = self.init_ha_proj(img.expand(1, -1, -1))
        output12 = self.init_ha2_proj(img.expand(1, -1, -1))
        output3 = self.init_hl_proj(img.expand(1, -1, -1))

        return output1,output12, output3

    def decode(self, imgs_emb, last_word, keywords_emb, image_ha, topic_ha, hl, ind, penalize_previous=False):
        """
        Args:
            imgs_emb: batch_size*5*512
            keywords_emb: batch_size*5*512
            last_word: Variable contraining a word index, batch_size*1
            ha, ca: attention RNN hidden state
            hl, cl: language RNN hidden state
            ind: sentence index [0, 5)
        """

        word_emb = self.embed(last_word) # 64*512

        ### attention RNN
        image_input_a = torch.cat((hl.squeeze(0), imgs_emb.mean(1, False), word_emb), 1).unsqueeze(1) # 64*1*(512*3)
        image_output_a, image_ha = self.image_attentionRNN(image_input_a, image_ha) # 64*1*512

        image_attn_weights = F.softmax(self.image_attn(torch.cat((imgs_emb[:, ind, :], torch.squeeze(image_output_a)), 1)), dim = 1) # batch_size*5
        image_attn_applied = torch.bmm(image_attn_weights.unsqueeze(1), imgs_emb) # batch_size*1*512
        # # no attention
        # image_attn_applied = imgs_emb[:, ind, :].unsqueeze(1)

        topic_input_a = torch.cat((hl.squeeze(0), keywords_emb.mean(1, False), word_emb), 1).unsqueeze(1) # 64*1*(512*3)
        topic_output_a, topic_ha = self.topic_attentionRNN(topic_input_a, topic_ha) # 64*1*512

        topic_attn_weights = F.softmax(self.topic_attn(torch.cat((keywords_emb[:, ind, :], torch.squeeze(topic_output_a)), 1)), dim = 1) # batch_size*5
        topic_attn_applied = torch.bmm(topic_attn_weights.unsqueeze(1), keywords_emb) # batch_size*1*512
        # # no attention
        # topic_attn_applied = keywords_emb[:, ind, :].unsqueeze(1)

        input_l = torch.cat((image_attn_applied, image_output_a, topic_output_a, topic_attn_applied), 2)
        # input_l = torch.cat((image_attn_applied, image_output_a), 2)
        
        output_l, hl = self.languageRNN(input_l, hl) # 64*1*512

        log_probs = F.log_softmax(self.logit(output_l[:, 0, :]), dim = 1) # 64*vocab_size

        if penalize_previous: # do not gen the same word as last word
            last_word_onehot = torch.FloatTensor(last_word.size(0), self.vocab_size).zero_().cuda()
            penalize_value = (last_word > 0).data.float() * -100
            mask = Variable(last_word_onehot.scatter_(1, last_word.data[:, None], 1.) * penalize_value[:, None])
            log_probs = log_probs + mask

        return word_emb, log_probs, image_ha, topic_ha, hl

    def forward(self, imgs, keywords, caption):
        """ gen sentences with teacher forcing

        Args:
            imgs: (batch_size, 5, feat_size) 64*5*2048
            keywords: batch_size*5*keyword_num
            caption: (batch_size, 5, seq_length) 64*5*30
        """

        batch_size = imgs.shape[0]

        ########################### encoding stage #############################
        
        imgs_emb, _ = self.encoder(imgs) # 64*5*512

        ########################### decoding stage #############################

        outputs = torch.FloatTensor(batch_size, self.story_size, self.seq_length, self.vocab_size).zero_().cuda() # 64*5*30*vocab_size

        keywords_emb = self.topic_proj(keywords) # batch_size*5*512

        for j in range(self.story_size):
            last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda() # 64*1, <EOS>
            
            # ha, ca, hl, cl = self.init_hidden_with_img(imgs[:, j, :])
            ha, ha2, hl = self.init_hidden_with_img_2(imgs[:, j, :])
           
            for i in range(self.seq_length):
                word_emb, log_probs, ha, ha2, hl = self.decode(imgs_emb, last_word, keywords_emb, ha, ha2, hl, j) # 64*vocab_size, 1*64*512

                outputs[:, j, i, :] = log_probs

                # ss choose the word
                if self.ss_prob > 0.0: # default range [0, 0.25]
                    sample_prob = torch.FloatTensor(batch_size).uniform_(0, 1).cuda() # 64*1
                    sample_mask = sample_prob < self.ss_prob # 64*1
                    if sample_mask.sum() == 0:
                        last_word = caption[:, j, i].clone() # use GT's final word as lastword of sentence
                    else:
                        sample_ind = sample_mask.nonzero().view(-1) # nonzero index
                        last_word = caption[:, j, i].data.clone()
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(log_probs.data) # 64*vocab_size
                        last_word.index_copy_(0, sample_ind, # select the highest prob word in sample_ind indices
                                            torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        last_word = Variable(last_word)
                else:
                    last_word = caption[:, j, i].clone()

                # break when whole batch's sentences end
                if i >= 1 and caption[:, j, i].data.sum() == 0:
                    break

        return outputs

    def sample(self, imgs, keywords):
        """ gen sentences by sampling

        Args:
            imgs: batch_size*5*2048
            seq, seq_log_probs: output story, each word selected by prb
        """

        batch_size = imgs.shape[0]

        ########################### encoding stage #############################
        imgs_emb, _ = self.encoder(imgs) # batch_size*5*512

        ########################### decoding stage #############################

        seq = torch.LongTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()
        seq_log_probs = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()
        baseline = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()

        keywords_emb = self.topic_proj(keywords)

        for j in range(self.story_size):
            last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda() # 64*1, <EOS>
            ha, ha2, hl = self.init_hidden_with_img_2(imgs[:, j, :])
            
            for i in range(self.seq_length):
                word_emb, log_probs, ha, ha2, hl = self.decode(imgs_emb, last_word, keywords_emb, ha, ha2, hl, j) # 64*vocab_size, 1*64*512
                
                if i < self.min_length: # make <EOS> not appear in first min_length words
                    mask = np.zeros((batch_size, log_probs.size(-1)), 'float32') # 64*vocab_size
                    mask[:, 0] = -1000
                    mask = Variable(torch.from_numpy(mask)).cuda()
                    log_probs = log_probs + mask

                # 'sample' select by nomial
                # fetch prev distribution: shape Nx(M+1)
                prob_prev = torch.exp(log_probs.data).cpu()
                last_word = torch.multinomial(prob_prev, 1).cuda() # 64*1
                # gather the logprobs at sampled positions
                sample_log_prob = log_probs.gather(1, Variable(last_word))
                # flatten indices for downstream processing
                last_word = last_word.view(-1).long()

                if (last_word > 0).sum() == 0 and i >= 1:
                    break

                seq[:, j, i] = last_word  # 64*1
                seq_log_probs[:, j, i] = sample_log_prob.view(-1) # 64*1

                # # NN as baseline, actor-critic
                # value = self.baseline_estimator(torch.cat((ha.squeeze(0).detach(), ha2.squeeze(0).detach(), hl.squeeze(0).detach()), dim = 1)) # batch_size*1
                # baseline[:, j, i] = value.view(-1)

        return seq, seq_log_probs, baseline

    def sample_greedy(self, imgs, keywords):
        """ gen sentences by greedy search

        Args:
            imgs: 64*5*2048
            seq, seq_log_probs: output story, each word selected by prb
        """
        batch_size = imgs.shape[0]

        ########################### encoding stage #############################
        imgs_emb, _ = self.encoder(imgs) # 64*5*512

        ########################### decoding stage #############################

        seq = torch.LongTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()
        seq_log_probs = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_().cuda()

        keywords_emb = self.topic_proj(keywords)

        for j in range(self.story_size):
            last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda() # 64*1, <EOS>
            ha, ha2, hl = self.init_hidden_with_img_2(imgs[:, j, :])
            
            for i in range(self.seq_length):
                word_emb, log_probs, ha, ha2, hl = self.decode(imgs_emb, last_word, keywords_emb, ha, ha2, hl, j) # 64*vocab_size, 1*64*512
                
                if i < self.min_length: # make <EOS> not appear in first min_length words
                    mask = np.zeros((batch_size, log_probs.size(-1)), 'float32') # 64*vocab_size
                    mask[:, 0] = -1000
                    mask = Variable(torch.from_numpy(mask)).cuda()
                    log_probs = log_probs + mask

                # greedy select max prob
                sample_log_prob, last_word = torch.max(log_probs, 1) # 64*1, 64*1
                last_word = last_word.data.view(-1).long()

                if (last_word > 0).sum() == 0 and i >= 1:
                    break

                seq[:, j, i] = last_word  # 64*1
                seq_log_probs[:, j, i] = sample_log_prob.view(-1) # 64*1

        return seq, seq_log_probs

    def predict(self, imgs, keywords, beam_size=5):
        """ gen sentences with beam search

        Args:
            imgs: 64*5*2048
            keywords: batch_size*5*num_topics
        """

        assert beam_size <= self.vocab_size and beam_size > 0
        if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
            return self.sample_greedy(imgs)

        batch_size = imgs.shape[0]

        # encode the visual features
        imgs_emb, _ = self.encoder(imgs) # 64*5*512

        ####################### decoding stage ##################################

        seq = torch.LongTensor(batch_size, self.story_size, self.seq_length).zero_()
        seq_log_probs = torch.FloatTensor(batch_size, self.story_size, self.seq_length).zero_()

        keywords_emb = self.topic_proj(keywords)

        # lets process the videos independently for now, for simplicity
        for k in range(batch_size):
            imgs_emb_k = imgs_emb[k, :, :].unsqueeze(0).expand(beam_size, imgs_emb.size(1), imgs_emb.size(2)).contiguous() # beam_size*5*512
            keywords_emb_k = keywords_emb[k, :, :].unsqueeze(0).expand(beam_size, keywords_emb.size(1), keywords_emb.size(2))

            # hamming diversity
            if self.hamming_diversity:
                # n-gram freq
                hamming = {}

            for j in range(self.story_size):
                
                # # BS display
                # print()
                # print('start BS')

                last_word = Variable(torch.FloatTensor(beam_size).long().zero_().cuda()) # beam_size*1, <EOS>

                # ha, ca, hl, cl = self.init_hidden_with_img(imgs[:, j, :])
                ha, ha2, hl = self.init_hidden_with_img_2(imgs[:, j, :])
                ha_k = ha[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
                hl_k = hl[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()
                ha2_k = ha2[:, k, :].unsqueeze(1).expand(-1, beam_size, ha.size(2)).contiguous()

                all_outputs = np.ones((1, beam_size), dtype='int32') # 1*beam_size
                all_masks = np.ones_like(all_outputs, dtype="float32")
                all_costs = np.zeros_like(all_outputs, dtype="float32")
                
                for i in range(self.seq_length):
                    if all_masks[-1].sum() == 0:
                        break

                    ### decode ###
                    word_emb, log_probs, ha_k, ha2_k, hl_k = self.decode(imgs_emb_k, last_word, keywords_emb_k, ha_k, ha2_k, hl_k, j, True)

                    # hamming diversity
                    if self.hamming_diversity and i > self.hamming_n - 2:
                        for tmpb in range(beam_size):
                            if self.hamming_n == 1:
                                for tmpk, tmpv in hamming.items():
                                    log_probs[tmpb, int(tmpk)] -= self.hamming_f * tmpv
                                    # log_probs[tmpb, int(tmpk)] -= np.log(2)
                            else:
                                tmpoutput = all_outputs[-(self.hamming_n-1):, tmpb]
                                tmpoutput = [str(_) for _ in tmpoutput]
                                tmpngram = ','.join(tmpoutput)
                                for tmpk, tmpv in hamming.items():
                                    tmpkl = tmpk.split(',')
                                    if ','.join(tmpkl[:-1]) == tmpngram:
                                        log_probs[tmpb, int(tmpkl[-1])] -= self.hamming_f * tmpv
                                        # print('hm: ', i, self.dataset.id2word[str(tmpkl[0])], self.dataset.id2word[str(tmpkl[1])])
                    
                    log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token

                    neg_log_probs = -log_probs

                    ### beam search, log_pros are added up every step ###

                    # all_costs[-1, :, None] equals to all_costs[-1, :].unsqueeze(1)
                    next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None]) # beam_size*vocab_size
                    (finished,) = np.where(all_masks[-1] == 0)
                    next_costs[finished, 1:] = np.inf

                    (indices, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)

                    all_outputs = all_outputs[:, indices]
                    all_masks = all_masks[:, indices]
                    all_costs = all_costs[:, indices]

                    all_outputs = np.vstack([all_outputs, outputs[None, :]])

                    # # BS display
                    # tmpcosts = chosen_costs[None, :] - all_costs[-1, :]

                    all_costs = np.vstack([all_costs, chosen_costs[None, :]])
                    mask = outputs != 0
                    all_masks = np.vstack([all_masks, mask[None, :]])

                    last_word = Variable(torch.from_numpy(outputs)).cuda()

                    # # BS display
                    # # print(all_outputs)
                    # # print(all_costs)
                    # # print(last_word)
                    # print("{} {:.2f}".format(self.dataset.id2word[str(last_word[0].item())], np.exp(-tmpcosts[0][0])), 
                    #     "{} {:.2f}".format(self.dataset.id2word[str(last_word[1].item())], np.exp(-tmpcosts[0][1])), 
                    #     "{} {:.2f}".format(self.dataset.id2word[str(last_word[2].item())], np.exp(-tmpcosts[0][2])))
                    # print(indices)

                    ha_k = Variable(torch.from_numpy(ha_k.data.cpu().numpy()[:, indices, :])).cuda()
                    hl_k = Variable(torch.from_numpy(hl_k.data.cpu().numpy()[:, indices, :])).cuda()
                    ha2_k = Variable(torch.from_numpy(ha2_k.data.cpu().numpy()[:, indices, :])).cuda()

                all_outputs = all_outputs[1:] # seq_length*beam_size, all_outputs[0] are 1
                all_costs = all_costs[1:] - all_costs[:-1] # restore step costs
                all_masks = all_masks[:-1]
                costs = all_costs.sum(axis=0) # beam_size*1
                lengths = all_masks.sum(axis=0)
                normalized_cost = costs / lengths
                best_idx = np.argmin(normalized_cost)
                seq[k, j, :all_outputs.shape[0]] = torch.from_numpy(all_outputs[:, best_idx])
                seq_log_probs[k, j, :all_outputs.shape[0]] = torch.from_numpy(all_costs[:, best_idx])

                # BS display
                # print(np.exp(-all_costs))
                # print(best_idx)
                # for step_outputs in all_outputs:
                #     print(self.dataset.id2word[str(step_outputs[0])], 
                #         self.dataset.id2word[str(step_outputs[1])], 
                #         self.dataset.id2word[str(step_outputs[2])])
                
                # update previous sentences' hamming freq with beam search result
                if self.hamming_diversity:
                    tmpoutput = all_outputs[:, best_idx]
                    tmpoutput = [str(_) for _ in tmpoutput]
                    if self.hamming_n == 1:
                        for token in tmpoutput:
                            if token == '0':
                                break

                            # # pos hamming
                            # if not self.dataset.pos_ifhamming(token):
                            #     continue

                            if token not in hamming:
                                hamming[token] = 1
                            else:
                                hamming[token] += 1
                    else:
                        for tmpi in range(len(tmpoutput) - self.hamming_n + 1):
                            tmpngram = ','.join(tmpoutput[tmpi:tmpi + self.hamming_n])
                            if tmpngram not in hamming:
                                hamming[tmpngram] = 1
                            else:
                                hamming[tmpngram] += 1

        # return the samples and their log likelihoods
        seq = seq.contiguous()
        seq_log_probs = seq_log_probs.contiguous()

        return seq, seq_log_probs # 64*5*30, 64*5*30


class TopicModel(nn.Module):
    def __init__(self, num_topics):
        super(TopicModel, self).__init__()    
        self.num_topics = num_topics   
        self.dropout = topicModel_dropout
        self.feat_size = feat_size

        self.linear = nn.Sequential(nn.Linear(self.feat_size, 512),
                                    nn.Tanh(),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(512, self.num_topics))
    
    def forward(self, imgs):
        """
        Args:
            imgs: batch_size*5*2048
        
        Returns:
            topic distribution: batch_size*5*num_topics
        """

        return F.log_softmax(self.linear(imgs), dim = 2)