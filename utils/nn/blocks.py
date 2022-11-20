import math
import time

import torch
import torch.nn as nn

from .modules import (
    GATED_ACT,
    LayerNorm, AttentionBase,
    SelfAttention, RNN, SparseConnectedLayer, 
    SeqMasking, SeqPermutation, Embedding, 
    apply_atcivation, apply_seq_batchnorm, get_norm, get_activation
    )
from ..base import ConfigBase
from .._typing.property import FEATURES


# ----------------------------------------------------------------------------------------------- #
# Initialization
def set_init(func):
    nn.Linear.reset_parameters = func
    SparseConnectedLayer.reset_parameters = func

# ----------------------------------------------------------------------------------------------- #
# Prematch blocks
class StatsEncoder(nn.Module):
    def __init__(self, 
        in_dim:int, ff_dim:int, 
        out_dim:int, bias: bool, 
        num_layers:int, dropout:float, 
        wdropoout:float, bdropoout:float, 
        prenorm:bool, predropout:bool, 
        norm:str, activation: str,
        last_layer_activation: bool
        ):

        super(StatsEncoder, self).__init__()
        assert num_layers >= 1
        
        self.norm = norm
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        # -------------------------------- #
        modules = []
        for l_num in range(num_layers):
            modules += [
                SparseConnectedLayer(
                    in_dim if l_num == 0 else ff_dim, 
                    ff_dim*2 if activation in GATED_ACT else ff_dim, 
                    bias=bias, 
                    pw=wdropoout, 
                    pb=bdropoout
                    ), 
                ]

            if predropout:
                modules += [nn.Dropout(dropout)]

            if prenorm:
                modules += [
                    get_norm(norm)(ff_dim*2 if activation in GATED_ACT else ff_dim), 
                    get_activation(activation), 
                    ]
            else:
                modules += [
                    get_activation(activation), 
                    get_norm(norm)(ff_dim), 
                    ]

            if not predropout:
                modules += [nn.Dropout(dropout)]

        self.stats1 = SparseConnectedLayer(ff_dim, out_dim*2 if ((activation in GATED_ACT) and last_layer_activation) else out_dim, bias=bias, pw=wdropoout, pb=bdropoout)
        self.stats2 = SparseConnectedLayer(ff_dim, out_dim*2 if ((activation in GATED_ACT) and last_layer_activation) else out_dim, bias=bias, pw=wdropoout, pb=bdropoout)
        self.stats_encoder = nn.ModuleList(modules)


    def forward(self, inputs: torch.Tensor, opponent_stats=False, mask: torch.BoolTensor=None) -> torch.Tensor:
        # |inputs| : (batch_size, seq_len, d_model)

        output = inputs
        for layer in self.stats_encoder:
            layer: nn.Linear | LayerNorm | nn.BatchNorm1d | nn.GELU | nn.Dropout | SparseConnectedLayer
            output: torch.Tensor

            if isinstance(layer, nn.BatchNorm1d):
                output = apply_seq_batchnorm(output, layer, mask if self.norm == 'masked_batch' else None)
            else:
                output = layer(output)

        output = self.stats2(output) if opponent_stats else self.stats1(output)

        if self.last_layer_activation:
            output = apply_atcivation(self.activation, output)

        return output

class WindowGamesFeatureEncoder(ConfigBase, nn.Module):
    def __init__(self):
        super(WindowGamesFeatureEncoder, self).__init__()
        self.features_config: dict = self._get_config('features')['league']
        self.models_config: dict = self._get_config('models')['prematch']['windowGamesFeatureEncoder']

        self.statsEncoder = StatsEncoder(   
            in_dim=len(FEATURES), 
            out_dim=self.models_config['embed_dim'], 
            **self.models_config['statsEncoder']
            )

        if self.models_config['splitStatsEncoder']:
            self.statsEncoder2 = StatsEncoder(   
                in_dim=len(FEATURES), 
                out_dim=self.models_config['embed_dim'], 
                **self.models_config['statsEncoder']
                )

        if self.models_config['pos_encoding']:
            self.pos_embedding = Embedding(
                self.features_config['window_size']+1, 
                self.models_config['embed_dim'], 
                **self.models_config['posEncoder'], padding_idx=0
                )

        self.resultEncoder = Embedding(
            num_embeddings=2,
            embedding_dim=self.models_config['embed_dim'], 
            **self.models_config['resultEncoder'], padding_idx=0
            )

        self.norm = get_norm(self.models_config['norm'])
        self.norm = self.norm(self.models_config['embed_dim'])

        self.masking = SeqMasking(**self.models_config['seq_masking'])
        self.permutation = SeqPermutation(**self.models_config['seq_permutation'])

    def forward(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Preprocces and encode input data
        inputs is raw dict output from dataset"""

        inputs['r_window'], inputs['r_window']['padded_mask'], inputs['r_window']['seq_len'] =\
            self.masking(inputs['r_window'], inputs['r_window']['padded_mask'], inputs['r_window']['seq_len'])
        inputs['r_window'] = self.permutation(inputs['r_window'], inputs['r_window']['padded_mask'], inputs['r_window']['seq_len'])
        
        inputs['d_window'], inputs['d_window']['padded_mask'], inputs['d_window']['seq_len'] =\
            self.masking(inputs['d_window'], inputs['d_window']['padded_mask'], inputs['d_window']['seq_len'])
        inputs['d_window'] = self.permutation(inputs['d_window'], inputs['d_window']['padded_mask'], inputs['d_window']['seq_len'])

        r_window = self.encode_features(inputs['r_window'], inputs['r_window']['padded_mask'])
        d_window = self.encode_features(inputs['d_window'], inputs['d_window']['padded_mask'])

        # position encoding
        if self.models_config['pos_encoding']:
            r_window = r_window + self.pos_embedding(self.generate_pos_tokens(r_window))
            d_window = d_window + self.pos_embedding(self.generate_pos_tokens(d_window))
        
        return r_window, d_window, inputs

    def generate_pos_tokens(self, inputs: torch.Tensor, mask: torch.BoolTensor=None) -> torch.Tensor:
        pos = torch.arange(1, inputs.size(1), device=inputs.device, dtype=torch.int64).repeat(inputs.size(0), 1)
        # |pos|: (B, S)

        if mask is not None:
            pos[mask] = 0

        return pos

    def encode_features(self, window: dict, mask:torch.Tensor=None) -> torch.Tensor:
        output: torch.Tensor = None
        for feature in window:
            f_output: torch.Tensor = None
            match feature:
                case 'stats':
                    f_output = self.statsEncoder(window[feature], opponent_stats=False, mask=mask)

                case 'opponent_stats':
                    if self.models_config['splitStatsEncoder']:
                        f_output = self.statsEncoder2(window[feature], opponent_stats=False, mask=mask)
                    else:
                        f_output = self.statsEncoder(window[feature], opponent_stats=True, mask=mask)

                case 'result':
                    f_output = self.resultEncoder(window[feature])

                case 'opponent': 
                    raise NotImplementedError

                case _: 
                    pass
                    
            if f_output is not None:
                if output is None: 
                    output = f_output
                else: 
                    output = output + f_output

        output = apply_seq_batchnorm(output, self.norm, mask if self.models_config['norm'] == 'masked_batch' else None)
        return output

class WindowSeqEncoder(ConfigBase, nn.Module):
    def __init__(self):
        super(WindowSeqEncoder, self).__init__()
        self.model_config: dict = self._get_config('models')['prematch']
        
        modules = []
        output_dim = 0
        encoder_type = self.model_config['windows_seq_encoder_type']
        if isinstance(encoder_type, str):
            types = encoder_type = [encoder_type]
        else:
            types = encoder_type

        in_dim = self.model_config['windowGamesFeatureEncoder']['embed_dim']
        for t in types:

            # windows_seq_encoder
            if t == 'transformer':
                out_dim = self.model_config['windows_seq_encoder']['transformer']['embed_dim']

                # Compare size
                if output_dim != 0:
                    if output_dim != out_dim:
                        modules += [nn.Linear(in_features=output_dim, out_features=out_dim)]

                elif in_dim != out_dim:
                    modules += [nn.Linear(in_features=in_dim, out_features=out_dim)]

                modules += [
                    SelfAttention(**self.model_config['windows_seq_encoder']['transformer'])
                    for _ in range(self.model_config['windows_seq_encoder']['transformer']['num_encoder_layers'])
                ]
                output_dim = out_dim

            elif t in ['LSTM', 'LSTMN', 'GRU', 'IRNN']:
                config = self.model_config['windows_seq_encoder'][t]
                self.pool_output = config['pool_output']

                assert not (config['output_hidden'] and self.pool_output), 'You cant pool hidden states'

                input_size = output_dim if output_dim != 0 else in_dim
                modules += [RNN(rnn_type=t, input_size=input_size, **config)]

                output_dim = config['embed_dim']
                if 'bidirectional' in config and config['bidirectional']:
                    output_dim *= 2

            else: raise Exception('Unexcpeted windows seq encoder type')

        self.windows_seq_encoder = nn.ModuleList(modules)
        self.output_dim = output_dim

    def forward(self, window: torch.Tensor, key_padding_mask:torch.Tensor, seq_len:torch.Tensor) -> torch.FloatTensor:
        skip_connection = None
        for idx, layer in enumerate(self.windows_seq_encoder):
            if isinstance(layer, nn.Linear):
                window = layer(window)

            elif isinstance(layer, AttentionBase):
                if idx == 0:
                    window = skip_connection = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)

                elif (idx+1)%self.model_config['windows_seq_encoder']['transformer']['skip_connection'] == 0:
                    window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
                    window = skip_connection = window + skip_connection

                else:
                    # However a ``` `True` value indicates that the corresponding key value will be IGNORED ```
                    # for some reasons irl it's the other way around
                    window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)

            elif isinstance(layer, RNN):
                window, state_dict = layer(window, key_padding_mask=key_padding_mask, seq_len=seq_len)
                key_padding_mask, seq_len = state_dict['key_padding_mask'], state_dict['seq_len']

            else: 
                window = layer(window)

        window: torch.Tensor # |window|: (B, S, F) or (B, F)
        if window.ndim == 3:
            if window.size(-1) > 1:
                if isinstance(layer, RNN):
                    return window[:, -1, :]

                if isinstance(layer, AttentionBase) or (isinstance(layer, RNN) and self.pool_output):
                    # |window| : (batch_size, seq_len, embed_dim)
                    pooled_w = self.__global_avg_pooling(
                        window=window, seq_len=seq_len,
                        key_padding_mask=key_padding_mask) 
                    # |pooled| : (batch_size, embed_dim)
                    return pooled_w

            else:
                # |window| : (batch_size, embed_dim, 1)
                pooled_w = window.squeeze(2)
                # |pooled| : (batch_size, embed_dim)
                return pooled_w
        else:
            return window

    def __global_avg_pooling(self, window: torch.Tensor, seq_len: torch.Tensor, key_padding_mask: torch.Tensor)  -> torch.FloatTensor:
        # multiply window by mask - zeros all padded tokens
        pooled = torch.mul(window, key_padding_mask.unsqueeze(2))
        # |pooled| : (batch_size, seq_len, d_model)

        # sum all elements by seq_len dim
        pooled = pooled.sum(dim=1)
        # |pooled| : (batch_size, d_model)

        # divide samples by by its seq_len, so we will get mean values by each sample
        pooled = pooled / seq_len.unsqueeze(1)
        # |pooled| : (batch_size, d_model)

        return pooled

class OutputHead(ConfigBase, nn.Module):
    def __init__(self, in_dim:int, regression:bool):
        super().__init__()
        self.emb_storage = {}
        self.regression = regression
        self.model_config: dict = self._get_config('models')['prematch']
        self.temperature = self.model_config['compare_encoder']['temperature']

        if self.model_config['compare_encoder_type'] == 'transformer':
            conf = self.model_config['compare_encoder']['transformer']
            embed_dim = conf['embed_dim']

            self.compare_embedding = nn.Embedding(2, embed_dim)

            self.in_fnn = nn.Linear(in_features=in_dim, out_features=embed_dim, bias=False)
            
            self.compare = nn.Sequential(*[
                    SelfAttention(**conf) for _ in range(conf['num_encoder_layers'])
            ])
            self.out_fnn = SparseConnectedLayer(
                in_features=embed_dim, 
                out_features=len(FEATURES) if regression else 1, 
                **conf)

        elif self.model_config['compare_encoder_type'] == 'linear':
            conf = self.model_config['compare_encoder']['linear']
            assert not regression, 'regression only for transformer'
            assert conf['norm'] in ['batch', 'layer', 'none', None]

            # -------------------------------------------------- #
            # in fnn
            modules = []
            norm = get_norm(conf['norm'])
            for _in, _out in zip( [in_dim]+conf['in_fnn_dims'][:-1], conf['in_fnn_dims'] ):
                modules += [SparseConnectedLayer(_in, _out*2 if conf['in_activation'] in GATED_ACT else _out, bias=conf['in_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout'])]

                if conf['predropout']:
                    modules += [nn.Dropout(conf['dropout'])]

                if conf['prenorm']:
                    modules += [
                        get_norm(norm)(_out*2 if conf['in_activation'] in GATED_ACT else _out), 
                        get_activation(conf['in_activation']), 
                        ]
                else:
                    modules += [
                        get_activation(conf['in_activation']), 
                        get_norm(norm)(_out), 
                        ]

                if not conf['predropout']:
                    modules += [nn.Dropout(conf['dropout'])]

            self.in_fnn = nn.Sequential(*modules)

            # -------------------------------------------------- #
            # compare fnn
            modules = []
            for _in, _out in zip( [_out*2]+conf['compare_fnn_dims'][:-1], conf['compare_fnn_dims']):
                modules += [
                    SparseConnectedLayer(_in, _out*2 if conf['compare_activation'] in GATED_ACT else _out, bias=conf['compare_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']),
                    get_activation(conf['compare_activation']),
                    nn.Dropout(conf['dropout']),
                    ]

            modules += [nn.Linear(conf['compare_fnn_dims'][-1], conf['out_dim'], bias=conf['compare_fnn_bias'])]
            self.compare_fnn = nn.Sequential(*modules)

        elif self.model_config['compare_encoder_type'] == 'subtract':
            assert not regression, 'regression only for transformer'
            conf = self.model_config['compare_encoder']['subtract']

            # -------------------------------------------------- #
            # int fnn
            modules = []
            for _in, _out in zip( [in_dim]+conf['in_fnn_dims'][:-1], conf['in_fnn_dims']):
                modules += [
                    SparseConnectedLayer(_in, _out*2 if conf['activation'] in GATED_ACT else _out, bias=conf['in_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']),
                    get_activation(conf['activation']),
                    nn.Dropout(conf['dropout']),
                    ]
            self.in_fnn = nn.Sequential(*modules)

            # -------------------------------------------------- #
            # compare fnn
            modules = []
            for _in, _out in zip( [_out]+conf['compare_fnn_dims'][:-1], conf['compare_fnn_dims']):
                modules += [
                    SparseConnectedLayer(_in, _out, bias=conf['compare_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']), 
                    nn.Dropout(conf['dropout']),
                    nn.Tanh(), 
                    ]
            modules += [nn.Linear(conf['compare_fnn_dims'][-1], 1, bias=conf['compare_fnn_bias'])]
            self.compare_fnn = nn.Sequential(*modules)

            # -------------------------------------------------- #
            #  init
            for w in self.compare_fnn.parameters():
                if w.ndim == 2:
                    nn.init.uniform_(w, -0.1, 0.1)

                if w.ndim == 1:
                    nn.init.zeros_(w)
        
        else: raise Exception

    def forward(self, radiant: torch.Tensor, dire: torch.Tensor):
        # | radiant, dire| : (batch_size, in_dim)
        radiant, dire = self.in_fnn(radiant), self.in_fnn(dire)
        # | radiant, dire | : (batch_size, embed_dim)
        self.emb_storage['radiant'] = radiant
        self.emb_storage['dire'] = dire

        if self.model_config['compare_encoder_type'] == 'transformer':
            radiant = radiant.unsqueeze(1)
            dire = dire.unsqueeze(1)
            # | radiant, dire | : (batch_size, 1, embed_dim)
            cat = torch.cat([dire, radiant], dim=1)
            # |pooled| : (batch_size, 2, embed_dim)

            # Add pos info
            cat = cat + self.compare_embedding(self.__generate_pos_tokens(cat))
            # |pooled| : (batch_size, 2, embed_dim)

            if self.regression:
                compare: torch.Tensor = self.compare(cat)
                self.emb_storage['compared'] = compare

                compare = self.out_fnn(compare)
                # |compare| : (batch_size, 2, len(FEATURES))
                return compare[:, 1], compare[:, 0]
                # |compare| : (batch_size, len(FEATURES))

            else:
                compare: torch.Tensor = self.compare(cat)
                self.emb_storage['compared'] = compare

                compare = self.out_fnn(compare)
                # |compare| : (batch_size, 2, 1)
                # index 0 - dire
                # index 1 - radiant
                compare = compare.squeeze(2)
                # |compare| : (batch_size, 2)
                return compare / self.temperature

        elif self.model_config['compare_encoder_type'] == 'linear':
            cat = torch.cat([radiant, dire], dim=1)
            # | cat | : (batch_size, embed_dim*2)
            compare = self.compare_fnn(cat)
            self.emb_storage['compared'] = compare
            # | compare | : (batch_size, 1/2)
            return compare / self.temperature

        elif self.model_config['compare_encoder_type'] == 'subtract':
            compare = self.compare_fnn(radiant - dire)
            self.emb_storage['compared'] = compare
            # | compare | : (batch_size, 1)
            compare = torch.cat([compare, -compare], dim=1)
            # | compare | : (batch_size, 2)
            return compare / self.temperature

    def __generate_pos_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.arange(inputs.size(1), device=inputs.device, dtype=torch.int64).repeat(inputs.size(0), 1)


