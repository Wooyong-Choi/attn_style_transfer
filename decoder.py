import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 padding_idx=0, pretrained_word_vec=None):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size, rnn_size)
        
        if pretrained_word_vec is not None:
            self.embedding.emb_luts = nn.Embedding.from_pretrained(pretrained_word_vec, freeze=False)

        self.generator = nn.Sequential(
            nn.Linear(rnn_size, vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, tgt, lengths, hidden):
        tgt = tgt[:-1]
        
        embeded = self.embedding(tgt)
        dec_out, _ = self.rnn(embeded, hidden)
        decoded = self.generator(dec_out)
        
        return decoded