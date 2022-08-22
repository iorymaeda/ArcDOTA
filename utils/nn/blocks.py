from codecs import raw_unicode_escape_decode
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ConfigBase
from .._typing.property import FEATURES


# ----------------------------------------------------------------------------------------------- #
# Stuff
class SwiGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""
    def forward(self, x:torch.Tensor):
        # |x| : (..., Any)
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
        # |x| : (..., Any//2)

class LayerNorm(nn.Module):
    """LayerNorm without bias"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x:torch.Tensor):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# ----------------------------------------------------------------------------------------------- #
# RNN
class __SelfAttention(nn.Module):
    """Self-attention for modded RNN

    source: https://github.com/gucci-j/imdb-classification-gru/blob/master/src/model_with_self_attention.py
    """
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(__SelfAttention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # query == hidden: (batch_size, hidden_dim)
        # key/value == gru_output: (batch_size, sentence_length, hidden_dim)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim)
        key = key.transpose(1, 2) # (batch_size, hidden_dim, sentence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sentence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2) # normalize sentence_length's dimension

        value = value.transpose(0, 1) # (batch_size, sentence_length, hidden_dim)
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim)

        # (batch_size, hidden_dim)
        return attention_output.squeeze(1), attention_weight.squeeze(1)


class RNN(nn.Module):
    RNNs = {'GRU': nn.GRU, 'LSTM': nn.LSTM}

    def __init__(
        self, input_size:int, embed_dim:int, 
        num_layers:int, dropout: float,
        attention:bool, bidirectional=False,
        RNN_type:Literal['GRU', 'LSTM']='GRU',
        output_hidden:bool=True, **kwargs):

        super(RNN, self).__init__()
        RNN_type = RNN_type.upper()

        if  RNN_type == 'LSTM': raise NotImplementedError

        if not output_hidden and attention: raise Exception("We outputs hidden while use attention, but `output_hidden==False`")

        self.output_hidden = output_hidden
        self.bidirectional = bidirectional
        self.attention = __SelfAttention(2*embed_dim if bidirectional else embed_dim) if attention else False
        self.rnn = self.RNNs[RNN_type](
            input_size=input_size , hidden_size=embed_dim, 
            num_layers=num_layers, bidirectional=bidirectional, 
            dropout=dropout, batch_first=True
        )
        
    def forward(self, x: torch.Tensor):
        output, hidden = self.rnn(x)
        output: torch.Tensor # (batch_size, sentence_length, embed_dim (*2 if bidirectional)) 
        hidden: torch.Tensor # (num_layers (*2 if bidirectional), batch_size, embed_dim)
        ## ordered: [f_layer_0, b_layer_0, ...f_layer_n, b_layer n]

        # concat the final output of forward direction and backward direction
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            # | hidden | : (batch_size, embed_dim * 2)
        else:
            hidden = hidden[-1,:,:]
            # | hidden | : (batch_size, embed_dim)

        if self.attention is not False:
            rescaled_hidden, attention_weight = self.attention(query=hidden, key=output, value=output)
            return rescaled_hidden
        else:
            return hidden if self.output_hidden else output[:,-1:,:]

# ----------------------------------------------------------------------------------------------- #
# Transformer stuff
class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        assert ff_dim%2 == 0

        self.linear1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.linear2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        outputs = self.activation(self.linear1(inputs))
        outputs = self.dropout(outputs)
        # |outputs| : (batch_size, seq_len, d_ff)
        
        outputs = self.linear2(outputs)
        # |outputs| : (batch_size, seq_len, d_model)

        return outputs


class AttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.15, **kwargs):
        super(AttentionBase, self).__init__()
        
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        self.ffn = PositionWiseFeedForwardNetwork(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # TODO
        self.config = {
            # pre | pre
            'layer_norm': 'post' 
        }


class CrossAttentionEncoder(AttentionBase):
    def forward(self, x1, x2, key_padding_mask=None, attn_mask=None):
        if self.config['layer_norm'] == 'pre':
            # ---------------------- #
            x1_ = self.layernorm1(x1)
            x2_ = self.layernorm1(x2)
            
            x1_attn_output, x1_attn_weights = self.attn(
                query=x1_, key=x2_, value=x2_, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            x2_attn_output, x2_attn_weights = self.attn(
                query=x2_, key=x1_, value=x1_, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            
            x1 = x1 + x1_attn_output
            x2 = x2 + x2_attn_output
            
            # ---------------------- #
            x1_ = self.layernorm2(x1)
            x2_ = self.layernorm2(x2)
            
            x1_fnn_output = self.ffn(x1_)
            x2_fnn_output = self.ffn(x2_)
            
            x1 = x1 + x1_fnn_output
            x2 = x2 + x2_fnn_output
            
            return x1, x2
        
        elif self.config['layer_norm'] == 'post':
            x1_attn_output, x1_attn_weights = self.attn(
                query=x1, key=x2, value=x2, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            x2_attn_output, x2_attn_weights = self.attn(
                query=x2, key=x1, value=x1, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            
            x1 = self.layernorm1(x1 + x1_attn_output)
            x2 = self.layernorm1(x2 + x2_attn_output)
            
            x1 = self.layernorm2(x1 + self.ffn(x1))
            x2 = self.layernorm2(x2 + self.ffn(x2))
            
            return x1, x2
        
        else: raise Exception


class TransformerEncoder(AttentionBase):
    def forward(self, inputs, key_padding_mask=None, attn_mask=None):
        if self.config['layer_norm'] == 'pre':
            inputs_ = self.layernorm1(inputs)
            
            attn_output, attn_weights = self.attn(
                query=inputs_, key=inputs_, value=inputs_, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

            x = inputs + attn_output
            x = x + self.ffn(self.layernorm2(x))
            return x
        
        elif self.config['layer_norm'] == 'post':
            attn_output, attn_weights = self.attn(
                query=inputs, key=inputs, value=inputs, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

            x = self.layernorm1(inputs + attn_output)
            x = self.layernorm2(x + self.ffn(x))
            return x
        
        else: raise Exception

# ----------------------------------------------------------------------------------------------- #
# Feature exctrators || Prematch
class StatsEncoder(nn.Module):
    def __init__(self, in_dim, ff_dim, out_dim, num_layers, dropout, norm='layer'):
        super(StatsEncoder, self).__init__()
        assert num_layers >= 1

        norm = nn.LayerNorm if norm == 'layer' else nn.BatchNorm1d
        # -------------------------------- #
        modules = [nn.Linear(in_dim, ff_dim, bias=True), nn.GELU(), nn.Dropout(dropout)]
        for n in range(num_layers - 1):
            if n == num_layers - 1:
                modules += [nn.Linear(ff_dim, ff_dim, bias=True), norm(ff_dim), nn.GELU(), nn.Dropout(dropout)]
            else:
                modules += [nn.Linear(ff_dim, ff_dim, bias=True), nn.GELU(), nn.Dropout(dropout)]

        self.stats1 = nn.Linear(ff_dim, out_dim, bias=True)
        self.stats2 = nn.Linear(ff_dim, out_dim, bias=True)
        self.stats_encoder = nn.ModuleList(modules)


    def forward(self, inputs: torch.Tensor, opponent_stats=False) -> torch.Tensor:
        # |inputs| : (batch_size, seq_len, d_model)

        output = inputs
        for layer in self.stats_encoder:
            layer: nn.Linear | LayerNorm | nn.BatchNorm1d | nn.GELU | nn.Dropout
            output: torch.Tensor

            if isinstance(layer, nn.BatchNorm1d):
                output = layer(output.transpose(1, 2))
                output = output.transpose(1, 2)
            else:
                output = layer(output)

        output = self.stats2(output) if opponent_stats else self.stats1(output)

        return output


class ResultEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ResultEncoder, self).__init__()
        self.embeddings = nn.Embedding(2, embed_dim)

    def forward(self, inputs: torch.IntTensor | torch. FloatTensor) -> torch.Tensor:
        # |inputs| : (batch_size, seq_len)
        if inputs.dtype is torch.int64:
            outputs = self.embeddings(inputs)
            # |outputs| : (batch_size, seq_len, embed_dim)
            return outputs

        else: raise NotImplementedError


class WindowedGamesFeatureEncoder(ConfigBase, nn.Module):
    def __init__(self):
        super(WindowedGamesFeatureEncoder, self).__init__()
        self.features_config: dict = self._get_config('features')['league']
        self.models_config: dict = self._get_config('models')['preamtch']['windowedGamesFeatureEncoder']

        self.statsEncoder = StatsEncoder(   
            in_dim=len(FEATURES), 
            ff_dim=self.models_config['statsEncoder']['ff_dim'], 
            num_layers=self.models_config['statsEncoder']['num_layers'], 
            norm=self.models_config['statsEncoder']['norm'],
            out_dim=self.models_config['embed_dim'], 
            dropout=self.models_config['dropout'],
        )

        self.resultEncoder = ResultEncoder(self.models_config['embed_dim'])
        self.pos_embedding = nn.Embedding(self.features_config['window_size']+1, self.models_config['embed_dim'])
        

    def forward(self, inputs: dict) -> torch.Tensor:
        """Preprocces and encode input data
        inputs is raw dict output from dataset"""

        r_window = self.encode_features(inputs['r_window'])
        d_window = self.encode_features(inputs['d_window'])

        # position encoding
        r_window = r_window + self.pos_embedding(self.generate_pos_tokens(r_window))
        d_window = d_window + self.pos_embedding(self.generate_pos_tokens(d_window))

        return r_window, d_window


    def generate_pos_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.arange(inputs.size(1), device=inputs.device, dtype=torch.int64).repeat(inputs.size(0), 1)


    def encode_features(self, window: dict) -> torch.Tensor:
        output: torch.Tensor = None

        for feature in window:
            f_output: torch.Tensor = None
            match feature:
                case 'stats':
                    f_output = self.statsEncoder(window[feature], opponent_stats=False)

                case 'opponent_stats':
                    f_output = self.statsEncoder(window[feature], opponent_stats=True)

                case 'result':
                    f_output = self.resultEncoder(window[feature])

                case 'opponent': 
                    raise NotImplementedError

                case _: 
                    pass
                    
            if f_output is not None:
                if output is None: output = f_output
                else: output += f_output

        return output


class OutputHead(ConfigBase, nn.Module):
    def __init__(self, in_dim:int, regression:bool):
        super().__init__()
        self.regression = regression
        self.model_config: dict = self._get_config('models')['preamtch']

        if self.model_config['compare_encoder_type'] == 'transformer':
            conf = self.model_config['compare_encoder']['transformer']
            self.in_fnn = nn.Linear(
                in_features=in_dim, 
                out_features=conf['embed_dim'],
                bias=False)
            self.compare_embedding = nn.Embedding(2, conf['embed_dim'])
            self.compare = nn.Sequential(
                *([
                    TransformerEncoder(**conf
                    ) for _ in range(conf['num_encoder_layers'])
                    ])
            )
            self.out_fnn = nn.Linear(
                in_features=conf['embed_dim'], 
                out_features=len(FEATURES) if regression else 1, 
                bias=False)

        elif self.model_config['compare_encoder_type'] == 'linear':
            assert not regression, 'regression only for transformer'
            conf = self.model_config['compare_encoder']['linear']

            modules = []
            for _in, _out in zip( [in_dim] + conf['in_fnn_dims'][:-1], conf['in_fnn_dims'] ):
                modules += [nn.Linear(_in, _out, bias=conf['bias']), nn.GELU(), nn.Dropout(conf['dropout'])]
            self.in_fnn = nn.Sequential(*modules)

            modules = []
            for _in, _out in zip():
                modules += [nn.Linear(_in, _out, bias=True), nn.GELU(), nn.Dropout(conf['dropout'])]


            self.out_fnn = nn.Linear(
                in_features=conf['embed_dim'], 
                out_features=len(FEATURES) if regression else 1, 
                bias=False)


    def forward(self, radiant: torch.Tensor, dire: torch.Tensor):
        # |radiant| : (batch_size, in_dim)
        # |dire|    : (batch_size, in_dim)
        if self.model_config['compare_encoder_type'] == 'transformer':
            radiant = self.in_fnn(radiant)
            # |radiant| : (batch_size, 1, embed_dim)
            dire = self.in_fnn(dire)
            # |dire| : (batch_size, 1, embed_dim)

            radiant = radiant.unsqueeze(1)
            # |radiant| : (batch_size, 1, embed_dim)
            dire = dire.unsqueeze(1)
            # |dire| : (batch_size, 1, embed_dim)

            pooled = torch.cat([dire, radiant], dim=1)
            # |pooled| : (batch_size, 2, embed_dim)