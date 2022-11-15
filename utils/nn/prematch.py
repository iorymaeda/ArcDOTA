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
        super(PrematchModel, self).__init__()
        # --------------------------------------------------------- #
        self.configs = {
            "features": self._get_config("features"),
            "match": self._get_config("match"),
            "models": self._get_config("models"),
            "train": self._get_config("train")
        }
        
        # --------------------------------------------------------- #
        self.emb_storage = {}
        self.regression = regression
        self.features_config: dict = self.configs['features']['league']
        self.model_config: dict = self.configs['models']['prematch']

        # --------------------------------------------------------- #
        # Windows
        self.windowGamesFeatureEncoder = blocks.WindowGamesFeatureEncoder()
        self.windows_seq_encoder = blocks.WindowSeqEncoder()

        output_dim = self.windows_seq_encoder.output_dim

        # --------------------------------------------------------- #
        # Tabular features
        t_config = self.features_config['features']['tabular']
        if t_config['teams']:
            emb_dim = self.model_config['team_embedding']['embedding_dim']
            output_dim += emb_dim
            self.team_embedding = blocks.Embedding(
                num_embeddings=teams_num+1, 
                **self.model_config['team_embedding']
            )
        if t_config['players']:
            raise NotImplementedError

        if t_config['grid']:
            output_dim += 7

        if t_config['prize_pool']:
            emb_dim = self.model_config['prize_pool_embedding']['embedding_dim']
            output_dim += emb_dim
            self.prize_pool_embedding = blocks.Embedding(
                num_embeddings=8, 
                **self.model_config['prize_pool_embedding'],
            )
        # --------------------------------------------------------- #
        # output head
        self.output_head = blocks.OutputHead(in_dim=output_dim, regression=regression)


    def forward(self, inputs: dict):
        # --------------------------------------------------------- #
        r_window, d_window = self.windowGamesFeatureEncoder(inputs)
        # |window| : (batch_size, seq_len, embed_dim)
        self.emb_storage['r_window_featurs'] = r_window
        self.emb_storage['d_window_featurs'] = d_window

        # --------------------------------------------------------- #
        r_emb = self.windows_seq_encoder(
            window=r_window, 
            seq_len=inputs['r_window']['seq_len'],
            key_padding_mask=inputs['r_window']['padded_mask'])
        # |emb| : (batch_size, embed_dim)
        d_emb = self.windows_seq_encoder(
            window=d_window, 
            seq_len=inputs['d_window']['seq_len'],
            key_padding_mask=inputs['d_window']['padded_mask'])
        # |emb| : (batch_size, embed_dim)
        self.emb_storage['r_window'] = r_emb
        self.emb_storage['d_window'] = d_emb

        # --------------------------------------------------------- #
        if self.features_config['features']['tabular']['grid']:
            r_emb = torch.cat([r_emb, inputs['grid']], dim=-1)
            # |r_emb| : (batch_size, embed_dim+G)

            d_emb = torch.cat([d_emb, inputs['grid']], dim=-1)
            # |d_emb| : (batch_size, embed_dim+G)

        if self.features_config['features']['tabular']['teams']:
            team_r = self.team_embedding(inputs['teams']['radiant'])
            r_emb = torch.cat([r_emb, team_r], dim=-1)
            # |r_emb| : (batch_size, embed_dim+T)

            team_d = self.team_embedding(inputs['teams']['dire'])
            d_emb = torch.cat([d_emb, team_d], dim=-1)
            # |d_emb| : (batch_size, embed_dim+T)

        if self.features_config['features']['tabular']['players']:
            raise NotImplementedError

        if self.features_config['features']['tabular']['prize_pool']:
            p_pool = self.prize_pool_embedding(inputs['prize_pool'])

            r_emb = torch.cat([r_emb, p_pool], dim=-1)
            # |r_emb| : (batch_size, embed_dim+P)

            d_emb = torch.cat([d_emb, p_pool], dim=-1)
            # |r_emb| : (batch_size, embed_dim+P)

        # --------------------------------------------------------- #
        output = self.output_head(r_emb, d_emb)
        self.emb_storage['output'] = output
        return output

    @torch.no_grad()
    def predict(self, inputs: dict):
        self.eval()
        output: torch.Tensor = self.forward(inputs)
        output = output.cpu()

        if self.regression: 
            return output

        elif output.ndim == 2:
            if output.size(1) == 2:
                output = output.softmax(dim=1)
                return output[:, 1]
                
            elif output.size(1) == 1:
                return output.sigmoid()

        elif output.ndim == 1:
            # In fact, it's impossible
            return output.sigmoid()

    def summary(self):
        torchsummary.summary(self)


