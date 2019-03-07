import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

from classifier import Classifier
from decoder import Decoder


class TransferNet(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 attn_dim, attn_hops, classifier_dim,
                 padding_idx=0, fc_init_range=0.1, pretrained_word_vec=None):
        super(TransferNet, self).__init__()
        
        self.classifier = Classifier(emb_size, vocab_size, rnn_size,
                                     attn_dim, attn_hops, classifier_dim,
                                     padding_idx, fc_init_range, pretrained_word_vec)
        
        self.decoder = Decoder(emb_size, vocab_size, rnn_size,
                               padding_idx, pretrained_word_vec)
        
        
    def forward(src, tgt, src_lengths, tgt_lengths):
        hidden, w_score = self.classifier.encode(self, src, src_lengths, reverse=True)
        
        self.decoder.init_state(src, None, hidden)
        decoded, attns = self.decoder(self, tgt, lengths):
        return decoded, attns
        
        