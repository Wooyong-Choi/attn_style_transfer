import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

# Copy 붙여서 Attention?
class Decoder(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 padding_idx=0, pretrained_word_vec=None):
        super(Decoder, self).__init__()
        
        tgt_emb = Embeddings(
            word_vec_size=emb_size,
            word_vocab_size=vocab_size,
            word_padding_idx=padding_idx,
            dropout=0
        )
        decoder = StdRNNDecoder(
            "LSTM", False, 1, rnn_size,
            attn_type="general", attn_func="softmax", embeddings=tgt_emb
        )
        
        if pretrained_word_vec is not None:
            encoder.embeddings.emb_luts = nn.Embedding.from_pretrained(pretrained_word_vec, freeze=False)
            
        self.decoder = decoder
        self.generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(tgt_field.vocab)),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, tgt, lengths):
        tgt = tgt[:-1]
        
        dec_out, attns = self.decoder(tgt, lengths)
        decoded = self.generator(dec_out)
        return decoded