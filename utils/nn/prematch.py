import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchsummary

from . import blocks
from ..base import ConfigBase


class PrematchModel(ConfigBase, nn.Module):
    def __init__(self, teams_num, team_emb_dim=8, embed_dim=64, num_heads=4, num_encoder_layers=6, dropout=0.25):
        super().__init__()
        # --------------------------------------------------------- #
        self.features_config = self.get_config('features')
        self.models_config = self.get_config('models')

        if self.features_config['league']['features']['tabular']['teams']:
            self.team_embedding = nn.Embedding(teams_num+1, team_emb_dim)

        # --------------------------------------------------------- #
        self.embed_dim = embed_dim
        
        # --------------------------------------------------------- #
        self.windowedGamesFeatureEncoder = blocks.WindowedGamesFeatureEncoder(embed_dim, dropout)
        self.transformer = nn.ModuleList(
            [
                blocks.TransformerEncoder(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    ff_dim=embed_dim, 
                    dropout=dropout,
                ) for _ in range(num_encoder_layers)
            ]
        )
        
        # --------------------------------------------------------- #
        compare_fnn_embed_dim = embed_dim
        if self.features_config['league']['features']['tabular']['teams']:
            compare_fnn_embed_dim += team_emb_dim

        self.compare_fnn = nn.Sequential(
            nn.Linear(compare_fnn_embed_dim,    compare_fnn_embed_dim//2, bias=False), nn.Tanh(),
            nn.Linear(compare_fnn_embed_dim//2, compare_fnn_embed_dim//2, bias=False), nn.Tanh(),
            nn.Linear(compare_fnn_embed_dim//2, 1, bias=False),
        )
        

    def forward(self, inputs: dict):
        # --------------------------------------------------------- #
        r_window, d_window = self.windowedGamesFeatureEncoder(inputs)

        # --------------------------------------------------------- #
        r_window = self.__forward_through_transformer(
            window=r_window, 
            key_padding_mask=inputs['r_window']['padded_mask'])
        d_window = self.__forward_through_transformer(
            window=d_window, 
            key_padding_mask=inputs['d_window']['padded_mask'])

        # --------------------------------------------------------- #
        pooled_r = self.__global_avg_pooling(
            window=r_window,
            seq_len=inputs['r_window']['seq_len'],
            key_padding_mask=inputs['r_window']['padded_mask'])
        pooled_d = self.__global_avg_pooling(
            window=r_window,
            seq_len=inputs['d_window']['seq_len'],
            key_padding_mask=inputs['d_window']['padded_mask'])  

        # --------------------------------------------------------- #
        if self.features_config['league']['features']['tabular']['teams']:
            team_r = self.team_embedding(inputs['teams']['radiant'])
            team_r = team_r * math.sqrt(self.embed_dim)

            team_d = self.team_embedding(inputs['teams']['dire'])
            team_d = team_d * math.sqrt(self.embed_dim)

            pooled_r = torch.cat([pooled_r, team_r], dim=-1)
            pooled_d = torch.cat([pooled_d, team_d], dim=-1)
            
        # --------------------------------------------------------- #
        compare = self.compare_fnn(pooled_r - pooled_d)

        return compare


    def __forward_through_transformer(self, window: torch.Tensor, key_padding_mask: torch.Tensor=None) -> torch.FloatTensor:
        for layer in self.transformer:
            window = layer(window, key_padding_mask=key_padding_mask)

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


    def configure_optimizers(self):
        raise NotImplementedError("This is for torch lightning")
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def training_step(self, train_batch: dict, batch_idx: int):
        raise NotImplementedError("This is for torch lightning")
        out = self.forward(train_batch)    
        loss = F.binary_cross_entropy_with_logits(out, train_batch['y'])
        return loss

    def summary(self):
        torchsummary.summary(self)


