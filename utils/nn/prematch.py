import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchsummary

from . import blocks
from ..base import ConfigBase
from .._typing.property import FEATURES


class PrematchModel(ConfigBase, nn.Module):
    def __init__(self, teams_num, regression:bool=True, **kwargs):
        super().__init__()
        # --------------------------------------------------------- #
        self.emb_storage = {}
        self.regression = regression
        self.features_config: dict = self._get_config('features')['league']
        self.model_config: dict = self._get_config('models')['preamtch']

        # --------------------------------------------------------- #
        if self.features_config['features']['tabular']['teams']:
            self.team_embedding = nn.Embedding(
                num_embeddings=teams_num+1, 
                embedding_dim=self.model_config['team_embedding']['embed_dim'],
            )

        # --------------------------------------------------------- #
        self.windowedGamesFeatureEncoder = blocks.WindowedGamesFeatureEncoder()

        # --------------------------------------------------------- #
        # windows_seq_encoder
        if self.model_config['windows_seq_encoder_type'] == 'transformer':
            modules = [
                blocks.TransformerEncoder(**self.model_config['windows_seq_encoder']['transformer'])
                for _ in range(self.model_config['windows_seq_encoder']['transformer']['num_encoder_layers'])
            ]

            if self.model_config['windowedGamesFeatureEncoder']['embed_dim'] != self.model_config['windows_seq_encoder']['transformer']['embed_dim']:
                modules = [nn.Linear(
                    in_features=self.model_config['windowedGamesFeatureEncoder']['embed_dim'], 
                    out_features=self.model_config['windows_seq_encoder']['transformer']['embed_dim'],
                )] + modules
            self.windows_seq_encoder = nn.ModuleList(modules)
            windows_seq_encoder_output_dim = self.model_config['windows_seq_encoder']['transformer']['embed_dim']


        elif self.model_config['windows_seq_encoder_type'] == 'GRU':
            self.windows_seq_encoder = blocks.RNN(
                RNN_type='GRU', input_size=self.model_config['windowedGamesFeatureEncoder']['embed_dim'], 
                **self.model_config['windows_seq_encoder']['GRU'])

            windows_seq_encoder_output_dim = self.model_config['windows_seq_encoder']['GRU']['embed_dim']
            if self.model_config['windows_seq_encoder']['GRU']['bidirectional']:
                windows_seq_encoder_output_dim *= 2

        else: raise Exception
        
        # --------------------------------------------------------- #
        # output head
        if self.features_config['features']['tabular']['teams']:
            windows_seq_encoder_output_dim += self.model_config['team_embedding']['embed_dim']

        self.output_head = blocks.OutputHead(in_dim=windows_seq_encoder_output_dim, regression=regression)



    def forward(self, inputs: dict):
        # --------------------------------------------------------- #
        r_window, d_window = self.windowedGamesFeatureEncoder(inputs)
        # |window| : (batch_size, seq_len, embed_dim)

        # --------------------------------------------------------- #
        r_window = self.__forward_through_seq_encoder(
            window=r_window, 
            seq_len=inputs['r_window']['seq_len'],
            key_padding_mask=inputs['r_window']['padded_mask'])
        # |window| : (batch_size, embed_dim)
        d_window = self.__forward_through_seq_encoder(
            window=d_window, 
            seq_len=inputs['d_window']['seq_len'],
            key_padding_mask=inputs['d_window']['padded_mask'])
        # |window| : (batch_size, embed_dim)
        self.emb_storage['r_window'] = r_window
        self.emb_storage['d_window'] = d_window

        # --------------------------------------------------------- #
        if self.features_config['features']['tabular']['teams']:
            team_r = self.team_embedding(inputs['teams']['radiant'])
            r_window = torch.cat([r_window, team_r], dim=-1)
            # |pooled| : (batch_size, embed_dim+team_emb_dim)

            team_d = self.team_embedding(inputs['teams']['dire'])
            d_window = torch.cat([d_window, team_d], dim=-1)
            # |pooled| : (batch_size, embed_dim+team_emb_dim)

        # --------------------------------------------------------- #
        output = self.output_head(r_window, d_window)
        return output


    def predict(self, inputs: dict):
        self.eval()
        with torch.no_grad():
            output: torch.Tensor = self.forward(inputs)
            output = output.cpu()

        if self.regression: 
            return output

        elif len(output.shape) == 2:
            if output.shape[1] == 2:
                output = output.softmax(dim=1)
                return output[:, 1]
                
            if output.shape[1] == 1:
                return output.sigmoid()

        elif len(output.shape) == 1:
            # In fact, it's impossible
            return output.sigmoid()


    def __forward_through_seq_encoder(self, window: torch.Tensor, key_padding_mask:torch.Tensor=None, seq_len:torch.Tensor=None) -> torch.FloatTensor:
        if self.model_config['windows_seq_encoder_type'] == 'transformer':
            self.windows_seq_encoder: nn.ModuleList

            skip_connection = None
            for idx, layer in enumerate(self.windows_seq_encoder):
                if isinstance(layer, nn.Linear):
                    window = layer(window)

                elif isinstance(layer, blocks.TransformerEncoder):
                    if (idx+1)%self.model_config['transformer']['skip_connection'] == 0:
                        window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
                        window = skip_connection = window + skip_connection

                    elif idx == 0:
                        window = skip_connection = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)

                    else:
                        # However a ``` `True` value indicates that the corresponding key value will be IGNORED ```
                        # for some reasons irl it's the other way around
                        window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
                else: raise Exception

            window: torch.Tensor 
            # |window| : (batch_size, seq_len, embed_dim
            pooled_w = self.__global_avg_pooling(
                window=window,
                seq_len=seq_len,
                key_padding_mask=key_padding_mask)
            # |pooled| : (batch_size, embed_dim)
            return pooled_w

        elif self.model_config['windows_seq_encoder_type'].upper() in ['GRU', 'LSTM']:
            self.windows_seq_encoder: nn.Module

            pooled_w = self.windows_seq_encoder(window)
            # |pooled| : (batch_size, embed_dim)
            return pooled_w


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


    def summary(self):
        torchsummary.summary(self)


