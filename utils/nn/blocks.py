import torch
import torch.nn as nn

from ..base import ConfigBase
from .._typing.property import FEATURES

"""TODO: implement config for all modules arhitecture"""


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        outputs = self.gelu(self.linear1(inputs))
        outputs = self.dropout(outputs)
        # |outputs| : (batch_size, seq_len, d_ff)
        
        outputs = self.linear2(outputs)
        # |outputs| : (batch_size, seq_len, d_model)

        return outputs


class AttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.15):
        super(AttentionBase, self).__init__()
        
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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
class StatsEncoder(nn.Module):
    def __init__(self, in_dim, ff_dim, out_dim, num_layers, dropout):
        super(StatsEncoder, self).__init__()
        assert num_layers >= 1

        # -------------------------------- #
        # TODO
        self.config = {'norm': 'batch'}
        norm = nn.LayerNorm if self.config['norm'] == 'layer' else nn.BatchNorm1d
        
        # -------------------------------- #
        modules = [nn.Linear(in_dim, ff_dim), norm(ff_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            modules += [nn.Linear(ff_dim, ff_dim), norm(ff_dim), nn.GELU(), nn.Dropout(dropout)]

        self.stats1 = nn.Linear(ff_dim, out_dim)
        self.stats2 = nn.Linear(ff_dim, out_dim)
        self.stats_encoder = nn.ModuleList(modules)


    def forward(self, inputs: torch.Tensor, opponent_stats=False) -> torch.Tensor:
        # |inputs| : (batch_size, seq_len, d_model)

        output = inputs
        for layer in self.stats_encoder:
            layer: nn.Linear | nn.BatchNorm1d | nn.GELU | nn.Dropout
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
    def __init__(self, embed_dim, dropout):
        super(WindowedGamesFeatureEncoder, self).__init__()
        self.config = self.get_config('features')

        self.statsEncoder = StatsEncoder(len(FEATURES), embed_dim, embed_dim, num_layers=2, dropout=dropout)
        self.resultEncoder = ResultEncoder(embed_dim)
        self.pos_embedding = nn.Embedding(self.config['league']['window_size']+1, embed_dim)
        

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

