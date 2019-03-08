import torch
import torch.nn as nn

from encoder import EncoderRNN
from decoder import DecoderRNN
from classifier import Classifier

class TransferNet(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 attn_dim, attn_hops, classifier_dim,
                 padding_idx=0, pretrained_word_vec=None):
        
        super(TransferNet, self).__init__()
        
        self.encoder = EncoderRNN(emb_size, vocab_size, rnn_size,
                                  padding_idx=padding_idx, pretrained_word_vec=pretrained_word_vec)
        
        self.classifier = Classifier(rnn_size, attn_dim, attn_hops, classifier_dim)
        
        self.decoder1 = DecoderRNN(emb_size, vocab_size, rnn_size,
                                   padding_idx=padding_idx, pretrained_word_vec=pretrained_word_vec)
        self.decoder2 = DecoderRNN(emb_size, vocab_size, rnn_size,
                                   padding_idx=padding_idx, pretrained_word_vec=pretrained_word_vec)
        
        self.choose_decoder = lambda idx: self.decoder1 if idx == 1 else self.decoder2 
        
        
    def forward(self, inputs, input_lengths, decoder_idx):
        assert decoder_idx == 1 or decoder_idx == 2
        
        outputs, hidden = self.encoder(inputs, input_lengths)
        content_hidden, w_score = self.classifier.generate(outputs, input_lengths, reverse=True)
        
        decoded = self.choose_decoder(decoder_idx)(inputs, input_lengths, content_hidden)
        return decoded, w_score
    
    def classify(self, inputs, input_lengths):
        outputs, hidden = self.encoder(inputs, input_lengths)
        cls, w_score = self.classifier(outputs, input_lengths)
        return cls, w_score
        