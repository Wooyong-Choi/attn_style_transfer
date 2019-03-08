import torch
import torch.nn as nn

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder

from classifier import Classifier
from decoder import Decoder


class MultiDecoderNet(nn.Module):
    
    def __init__(self, emb_size, vocab_size, rnn_size,
                 padding_idx=0, fc_init_range=0.1):
        super(MultiDecoderNet, self).__init__()
        
        src_emb = Embeddings(
            word_vec_size=emb_size,
            word_vocab_size=vocab_size,
            word_padding_idx=padding_idx,
            dropout=0
        )
        self.encoder = RNNEncoder(
            rnn_type="GRU",
            bidirectional=True,
            num_layers=1,
            hidden_size=rnn_size,
            embeddings=src_emb
        )
        
        self.decoder1 = Decoder(emb_size, vocab_size, rnn_size, padding_idx)
        self.decoder2 = Decoder(emb_size, vocab_size, rnn_size, padding_idx)
        
        self.choose_decoder = lambda idx: self.decoder1 if idx == 1 else self.decoder2 
        
        
    def forward(self, inputs, input_lengths, decoder_idx, only_encode=False):
        assert decoder_idx == 1 or decoder_idx == 2
        
        src_state, memory_bank, _ = self.encoder(inputs[1:], input_lengths)
        src_state = self._cat_direction(src_state)
        
        if only_encode:
            return src_state
        
        src_state = src_state.unsqueeze(0)
        
        self.choose_decoder(decoder_idx).init_state(inputs, memory_bank, src_state)
        decoded = self.choose_decoder(decoder_idx)(inputs, input_lengths, memory_bank)
        
        return decoded
        
    def _cat_direction(self, state):
        return torch.cat([state[0:state.size(0):2], state[1:state.size(0):2]], 2).squeeze(0).contiguous()

        
class Discriminator(nn.Module):
    def __init__(self, ninput, noutput=2, layers='128-128', activation=nn.ReLU(), device=torch.device("cpu")):
        super(Discriminator, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1]).to(device)
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization in first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1]).to(device)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput).to(device)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
        