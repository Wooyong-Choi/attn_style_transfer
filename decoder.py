import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder
from onmt.decoders.decoder import InputFeedRNNDecoder

# Copy 붙여서 Attention?
class Decoder(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 padding_idx=0, pretrained_word_vec=None):
        super(Decoder, self).__init__()
        
#         tgt_emb = Embeddings(
#             word_vec_size=emb_size,
#             word_vocab_size=vocab_size,
#             word_padding_idx=padding_idx,
#             dropout=0
#         )
#         decoder = InputFeedRNNDecoder(
#             "GRU", False, 1, rnn_size,
#             attn_type="general", attn_func="softmax", embeddings=tgt_emb
#         )
#        
#        if pretrained_word_vec is not None:
#            encoder.embeddings.emb_luts = nn.Embedding.from_pretrained(pretrained_word_vec, freeze=False)
#            
#        self.decoder = decoder

        self.state = None
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.decoder = nn.GRU(emb_size, rnn_size)
        
        self.generator = nn.Sequential(
            nn.Linear(rnn_size, vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, tgt, lengths, memory_bank):
        tgt = tgt[:-1]
        
        #dec_out, attns = self.decoder(tgt, memory_bank, lengths)
        #return decoded, attns
        tgt = tgt.squeeze()
        embeded = self.embedding(tgt)
        dec_out, _ = self.decoder(embeded, self.state)
        decoded = self.generator(dec_out)
        
        return decoded
        
    
    def init_state(self, src, mem_bank, state):
        #self.decoder.init_state(src, mem_bank, state)
        self.state = state