import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

from utils import sequence_mask

class Classifier(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 attn_dim, attn_hops, classifier_dim,
                 padding_idx=0, fc_init_range=0.1, pretrained_word_vec=None):
        super(Classifier, self).__init__()
        
        src_emb = Embeddings(
            word_vec_size=emb_size,
            word_vocab_size=vocab_size,
            word_padding_idx=padding_idx,
            dropout=0
        )
        encoder = RNNEncoder(
            rnn_type="LSTM",
            bidirectional=True,
            num_layers=1,
            hidden_size=rnn_size,
            embeddings=src_emb
        )
        
        if pretrained_word_vec is not None:
            encoder.embeddings.emb_luts = nn.Embedding.from_pretrained(pretrained_word_vec, freeze=False)

        self.encoder = encoder
        
        self.attn_hops = attn_hops
        
        self.w_s1 = nn.Linear(rnn_size, attn_dim, bias=False)
        self.w_s2 = nn.Linear(attn_dim, attn_hops, bias=False)
        
        self.w_cls = nn.Linear(rnn_size * attn_hops, classifier_dim)
        self.w_out = nn.Linear(classifier_dim, 2)
        
        self.w_s1.weight.data.uniform_(-fc_init_range, fc_init_range)
        self.w_s2.weight.data.uniform_(-fc_init_range, fc_init_range)
        
    def forward(self, src, lengths, reverse=False):
        _, memory_bank, _ = self.encoder(src[1:], lengths)
        
        # 안에서 masking하므로 상관없음 1은 sos, -1은 pad or eos
        output, w_score = self._attn_word(src[1:-1], memory_bank, lengths, reverse)
        
        output = output.view(output.size(0), -1)
        output = torch.tanh(self.w_cls(output))
        cls = self.w_out(output)
        
        return cls, w_score
    
    def encode(self, src, lengths, reverse=False):
        _, memory_bank, _ = self.encoder(src[1:], lengths)
        output, w_score = self._attn_word(src[1:-1], memory_bank, lengths, reverse)        
        return output, w_score
    
    def _attn_word(self, src, mem_bank, lengths, reverse):
        src_len, batch_size, hidden_size = mem_bank.size()
        
        mem_bank = mem_bank.transpose(0, 1).contiguous()
        mem_bank = mem_bank.view(-1, hidden_size)
        
        src = src.transpose(0, 1).contiguous()
        src = src.view(batch_size, 1, src_len)  # [bsz, 1, len]
        src = torch.cat([src for i in range(self.attn_hops)], dim=1)
        
        hbar = torch.tanh(self.w_s1(mem_bank)) # [bsz*len, attention-unit]
        align = self.w_s2(hbar).view(batch_size, src_len, -1) # [bsz, len, hop]
        align = align.transpose(1, 2).contiguous() # [bsz, hop, len]
        
        mask = sequence_mask(lengths, self.attn_hops)
        align = align.masked_fill_(1-mask, float("-inf"))
        
        align = torch.softmax(align.view(-1, src_len), dim=-1)
        align = align.view(batch_size, self.attn_hops, src_len)
        
        if reverse:
            align = self.attn_hops-align
            align = align.masked_fill_(1-mask, float("-inf"))
            align = torch.softmax(align.view(-1, src_len), dim=-1)
            align = align.view(batch_size, self.attn_hops, src_len) # [bsz, hop, len]
         
        mem_bank = mem_bank.view(batch_size, src_len, hidden_size)
        return torch.bmm(align, mem_bank), align