import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 padding_idx=0, pretrained_word_vec=None):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size, rnn_size)
        
        if pretrained_word_vec is not None:
            self.embedding.emb_luts = nn.Embedding.from_pretrained(pretrained_word_vec, freeze=False)
        
    def forward(self, src, lengths):
        src = src[1:]
        
        embedded = self.embedding(src)
        packed = pack_padded_sequence(embedded, lengths)
        outputs, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(outputs)
        
        return outputs, hidden