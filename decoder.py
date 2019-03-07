import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

from utils import sequence_mask

class Decoder(nn.Module):
    def __init__(self, emb_size, vocab_size, rnn_size,
                 padding_idx=0, fc_init_range=0.1, pretrained_word_vec=None):