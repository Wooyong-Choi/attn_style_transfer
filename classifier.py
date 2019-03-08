import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

from utils import sequence_mask


class Classifier(nn.Module):
    
    def __init__(self, rnn_size, attn_dim, attn_hops, classifier_dim,
                 padding_idx=0, fc_init_range=0.1, pretrained_word_vec=None):
        super(Classifier, self).__init__()
        
        self.attn_hops = attn_hops
        
        self.w_s1 = nn.Linear(rnn_size, attn_dim, bias=False)
        self.w_s2 = nn.Linear(attn_dim, attn_hops, bias=False)
        
        self.w_cls = nn.Linear(rnn_size * attn_hops, classifier_dim)
        self.w_out = nn.Linear(classifier_dim, 2)
        
        # Initialize weights
        self.w_s1.weight.data.uniform_(-fc_init_range, fc_init_range)
        self.w_s2.weight.data.uniform_(-fc_init_range, fc_init_range)
        
    def forward(self, memory_bank, lengths):
        outputs, w_score = self._attn_word(memory_bank, lengths)
        
        outputs = outputs.view(outputs.size(0), -1)
        outputs = torch.tanh(self.w_cls(outputs))
        cls = self.w_out(outputs)
        
        return cls, w_score
    
    def generate(self, memory_bank, lengths, reverse):
        outputs, w_score = self._attn_word(memory_bank, lengths, reverse=reverse)
        
        # sum by hops
        outputs = outputs.sum(dim=1).unsqueeze(dim=0)
        return outputs, w_score
    
    def _attn_word(self, mem_bank, lengths, reverse=False):
        src_len, batch_size, hidden_size = mem_bank.size()
        
        mem_bank = mem_bank.transpose(0, 1).contiguous()
        mem_bank = mem_bank.view(-1, hidden_size)
        
        hbar = torch.tanh(self.w_s1(mem_bank)) # [bsz*len, attention-unit]
        align = self.w_s2(hbar).view(batch_size, src_len, -1) # [bsz, len, hop]
        align = align.transpose(1, 2).contiguous() # [bsz, hop, len]
        
        mask = sequence_mask(lengths, self.attn_hops)
        align = align.masked_fill_(1-mask, float("-inf"))
        
        align = torch.softmax(align.view(-1, src_len), dim=-1)
        align = align.view(batch_size, self.attn_hops, src_len)
        
        mem_bank = mem_bank.view(batch_size, src_len, hidden_size)
        
        if reverse:
            align = self.attn_hops-align
            align = align.masked_fill_(1-mask, float("-inf"))
            align = torch.softmax(align.view(-1, src_len), dim=-1)
            align = align.view(batch_size, self.attn_hops, src_len) # [bsz, hop, len]
         
        return torch.bmm(align, mem_bank), align