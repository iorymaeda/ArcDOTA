import math
import time

import torch
import torch.nn as nn

from .modules import LayerNorm, TransformerEncoder, Callable, RNN, TransformerEncoder, SparseConnectedLayer, SeqMasking, SeqPermutation
from ..base import ConfigBase
from .._typing.property import FEATURES


# ----------------------------------------------------------------------------------------------- #
# Initialization
def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

def set_init(func):
    nn.Linear.reset_parameters = func
    SparseConnectedLayer.reset_parameters = func

set_init(reset_parameters)

# ----------------------------------------------------------------------------------------------- #
# Prematch blocks
class StatsEncoder(nn.Module):
    def __init__(self, in_dim:int, ff_dim:int, out_dim:int, bias: bool, num_layers:int, dropout:float, wdropoout:float, bdropoout:float, prenorm:bool, norm:str):
        super(StatsEncoder, self).__init__()
        assert num_layers >= 1
        assert norm in ['layer', 'batch']

        norm = nn.LayerNorm if norm == 'layer' else nn.BatchNorm1d
        # -------------------------------- #
        modules = [SparseConnectedLayer(in_dim, ff_dim, bias=bias, pw=wdropoout, pb=bdropoout), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            if prenorm:
                modules += [SparseConnectedLayer(ff_dim, ff_dim, bias=bias, pw=wdropoout, pb=bdropoout), norm(ff_dim), nn.GELU(), nn.Dropout(dropout)]
            else:
                modules += [SparseConnectedLayer(ff_dim, ff_dim, bias=bias, pw=wdropoout, pb=bdropoout), nn.GELU(), norm(ff_dim), nn.Dropout(dropout)]

        self.stats1 = SparseConnectedLayer(ff_dim, out_dim, bias=bias, pw=wdropoout, pb=bdropoout)
        self.stats2 = SparseConnectedLayer(ff_dim, out_dim, bias=bias, pw=wdropoout, pb=bdropoout)
        self.stats_encoder = nn.ModuleList(modules)


    def forward(self, inputs: torch.Tensor, opponent_stats=False) -> torch.Tensor:
        # |inputs| : (batch_size, seq_len, d_model)

        output = inputs
        for layer in self.stats_encoder:
            layer: nn.Linear | LayerNorm | nn.BatchNorm1d | nn.GELU | nn.Dropout | SparseConnectedLayer
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
        if self.models_config['pos_encoding']:
            self.pos_embedding = nn.Embedding(self.features_config['window_size']+1, self.models_config['embed_dim'])

        self.resultEncoder = ResultEncoder(self.models_config['embed_dim'])

    def forward(self, inputs: dict) -> torch.Tensor:
        """Preprocces and encode input data
        inputs is raw dict output from dataset"""

        r_window = self.encode_features(inputs['r_window'])
        d_window = self.encode_features(inputs['d_window'])

        # position encoding
        if self.models_config['pos_encoding']:
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
                    TransformerEncoder(**self.model_config['windows_seq_encoder']['transformer'])
                    for _ in range(self.model_config['windows_seq_encoder']['transformer']['num_encoder_layers'])
                ]
                output_dim = out_dim

            elif t in ['LSTM', 'LSTMN', 'GRU', 'IRNN']:
                config = self.model_config['windows_seq_encoder'][t]
                input_size = output_dim if output_dim != 0 else in_dim
                modules += [RNN(rnn_type=t, input_size=input_size, **config)]

                output_dim = config['embed_dim']
                if 'bidirectional' in config and config['bidirectional']:
                    output_dim *= 2

            else: raise Exception('Unexcpeted windows seq encoder type')

        self.windows_seq_encoder = nn.ModuleList(modules)
        self.output_dim = output_dim

        self.masking = SeqMasking(**self.model_config['windowGamesFeatureEncoder']['seq_masking'])
        self.permutation = SeqPermutation(**self.model_config['windowGamesFeatureEncoder']['seq_permutation'])

    def forward(self, window: torch.Tensor, key_padding_mask:torch.Tensor=None, seq_len:torch.Tensor=None) -> torch.FloatTensor:
        with torch.no_grad():
            window = self.masking(window)
            window = self.permutation(window)

        skip_connection = None
        for idx, layer in enumerate(self.windows_seq_encoder):
            if isinstance(layer, nn.Linear):
                window = layer(window)

            elif isinstance(layer, TransformerEncoder):
                if idx == 0:
                    window = skip_connection = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)

                elif (idx+1)%self.model_config['windows_seq_encoder']['transformer']['skip_connection'] == 0:
                    window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
                    window = skip_connection = window + skip_connection

                else:
                    # However a ``` `True` value indicates that the corresponding key value will be IGNORED ```
                    # for some reasons irl it's the other way around
                    window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
            else: 
                window = layer(window)

        window: torch.Tensor 
        if len(window.shape) == 3:
            if window.shape[-1] > 1:
                # |window| : (batch_size, seq_len, embed_dim)
                pooled_w = self.__global_avg_pooling(
                    window=window, seq_len=seq_len,
                    key_padding_mask=key_padding_mask) 
                # |pooled| : (batch_size, embed_dim)
                return pooled_w
            else:
                # |window| : (batch_size, embed_dim, 1)
                pooled = window.squeeze(2)
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

        if self.model_config['compare_encoder_type'] == 'transformer':
            conf = self.model_config['compare_encoder']['transformer']
            embed_dim = conf['embed_dim']

            self.compare_embedding = nn.Embedding(2, embed_dim)

            self.in_fnn = nn.Linear(in_features=in_dim, out_features=embed_dim, bias=False)
            
            self.compare = nn.Sequential(*[
                    TransformerEncoder(**conf) for _ in range(conf['num_encoder_layers'])
            ])
            self.out_fnn = SparseConnectedLayer(
                in_features=embed_dim, 
                out_features=len(FEATURES) if regression else 1, 
                **conf)

        elif self.model_config['compare_encoder_type'] == 'linear':
            assert not regression, 'regression only for transformer'
            conf = self.model_config['compare_encoder']['linear']

            assert conf['norm'] in ['batch', 'layer', None]
            if conf['norm'] == 'batch':
                norm = nn.BatchNorm1d
            elif conf['norm'] == 'layer':
                norm = nn.LayerNorm
            else:
                norm = Callable

            modules = []
            for _in, _out in zip( [in_dim]+conf['in_fnn_dims'][:-1], conf['in_fnn_dims'] ):
                if conf['prenorm']:
                    modules += [SparseConnectedLayer(_in, _out, bias=conf['in_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']), norm(_out), nn.GELU(), nn.Dropout(conf['dropout'])]
                else:
                    modules += [SparseConnectedLayer(_in, _out, bias=conf['in_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']), nn.GELU(), norm(_out), nn.Dropout(conf['dropout'])]
            self.in_fnn = nn.Sequential(*modules)

            modules = []
            for _in, _out in zip( [_out*2]+conf['compare_fnn_dims'][:-1], conf['compare_fnn_dims']):
                modules += [SparseConnectedLayer(_in, _out, bias=conf['compare_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']), nn.GELU(), nn.Dropout(conf['dropout'])]
            modules += [nn.Linear(conf['compare_fnn_dims'][-1], conf['out_dim'], bias=conf['compare_fnn_bias'])]
            self.compare_fnn = nn.Sequential(*modules)

        elif self.model_config['compare_encoder_type'] == 'subtract':
            assert not regression, 'regression only for transformer'
            conf = self.model_config['compare_encoder']['subtract']

            modules = []
            for _in, _out in zip( [in_dim]+conf['in_fnn_dims'][:-1], conf['in_fnn_dims'] ):
                modules += [SparseConnectedLayer(_in, _out, bias=conf['in_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']), nn.GELU(), nn.Dropout(conf['dropout'])]
            self.in_fnn = nn.Sequential(*modules)

            modules = []
            for _in, _out in zip( [_out]+conf['compare_fnn_dims'][:-1], conf['compare_fnn_dims']):
                modules += [SparseConnectedLayer(_in, _out, bias=conf['compare_fnn_bias'], pw=conf['wdropoout'], pb=conf['bdropoout']), nn.Tanh(), nn.Dropout(conf['dropout'])]
            modules += [nn.Linear(conf['compare_fnn_dims'][-1], 1, bias=conf['compare_fnn_bias'])]
            self.compare_fnn = nn.Sequential(*modules)
        
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
                self.emb_storage['compared'] = compare.detach()

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
                return compare

        elif self.model_config['compare_encoder_type'] == 'linear':
            cat = torch.cat([radiant, dire], dim=1)
            # | cat | : (batch_size, embed_dim*2)
            compare = self.compare_fnn(cat)
            self.emb_storage['compared'] = compare
            # | compare | : (batch_size, 1/2)
            return compare

        elif self.model_config['compare_encoder_type'] == 'subtract':
            compare = self.compare_fnn(radiant - dire)
            self.emb_storage['compared'] = compare
            # | compare | : (batch_size, 1)
            compare = torch.cat([compare, -compare], dim=1)
            # | compare | : (batch_size, 2)
            return compare

    def __generate_pos_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.arange(inputs.size(1), device=inputs.device, dtype=torch.int64).repeat(inputs.size(0), 1)


