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
        self.models_config: dict = self._get_config('models')['preamtch']

        # --------------------------------------------------------- #
        if self.features_config['features']['tabular']['teams']:
            self.team_embedding = nn.Embedding(
                num_embeddings=teams_num+1, 
                embedding_dim=self.models_config['team_embedding']['embed_dim'],
            )
        self.windowedGamesFeatureEncoder = blocks.WindowedGamesFeatureEncoder()

        # --------------------------------------------------------- #
        modules = [
            blocks.TransformerEncoder(
                embed_dim=self.models_config['transformer']['embed_dim'], 
                num_heads=self.models_config['transformer']['num_heads'], 
                ff_dim=self.models_config['transformer']['ff_dim'], 
                dropout=self.models_config['transformer']['dropout'],
            ) for _ in range(self.models_config['transformer']['num_encoder_layers'])
        ]

        if self.models_config['windowedGamesFeatureEncoder']['embed_dim'] != self.models_config['transformer']['embed_dim']:
            modules = [nn.Linear(
                in_features=self.models_config['windowedGamesFeatureEncoder']['embed_dim'], 
                out_features=self.models_config['transformer']['embed_dim'],
            )] + modules

        self.transformer = nn.ModuleList(modules)
        
        # --------------------------------------------------------- #
        compare_fnn_dim_in = self.models_config['transformer']['embed_dim']
        if self.features_config['features']['tabular']['teams']:
            compare_fnn_dim_in += self.models_config['team_embedding']['embed_dim']

        self.compare_fnn = nn.Linear(
            in_features=compare_fnn_dim_in, 
            out_features=self.models_config['compare_fnn']['embed_dim'],
            bias=False,
        )

        self.compare_embedding = nn.Embedding(2, self.models_config['compare_fnn']['embed_dim'])
        self.compare = nn.Sequential(
            *([
                blocks.TransformerEncoder(
                    embed_dim=self.models_config['compare_fnn']['embed_dim'], 
                    num_heads=self.models_config['compare_fnn']['num_heads'], 
                    ff_dim=self.models_config['compare_fnn']['ff_dim'], 
                    dropout=self.models_config['compare_fnn']['dropout'],
                ) for _ in range(self.models_config['compare_fnn']['num_encoder_layers'])
                ])
        )
        self.linear = nn.Linear(
            in_features=self.models_config['compare_fnn']['embed_dim'], 
            out_features=len(FEATURES) if regression else 1, 
            bias=False
        )


    def forward(self, inputs: dict):
        # --------------------------------------------------------- #
        r_window, d_window = self.windowedGamesFeatureEncoder(inputs)
        # |window| : (batch_size, seq_len, embed_dim)

        # --------------------------------------------------------- #
        r_window = self.__forward_through_transformer(
            window=r_window, 
            key_padding_mask=inputs['r_window']['padded_mask'])
        # |window| : (batch_size, seq_len, embed_dim)
        d_window = self.__forward_through_transformer(
            window=d_window, 
            key_padding_mask=inputs['d_window']['padded_mask'])
        # |window| : (batch_size, seq_len, embed_dim)

        # --------------------------------------------------------- #
        # Pooling
        # pooled_r = torch.mean(r_window, dim=1)
        # pooled_d = torch.mean(d_window, dim=1)
        pooled_r = self.__global_avg_pooling(
            window=r_window,
            seq_len=inputs['r_window']['seq_len'],
            key_padding_mask=inputs['r_window']['padded_mask'])
        # |pooled| : (batch_size, embed_dim)
        pooled_d = self.__global_avg_pooling(
            window=r_window,
            seq_len=inputs['d_window']['seq_len'],
            key_padding_mask=inputs['d_window']['padded_mask'])  
        # |pooled| : (batch_size, embed_dim)

        # --------------------------------------------------------- #
        if self.features_config['features']['tabular']['teams']:
            team_r = self.team_embedding(inputs['teams']['radiant'])
            pooled_r = torch.cat([pooled_r, team_r], dim=-1)
            # |pooled| : (batch_size, embed_dim+team_emb_dim)

            team_d = self.team_embedding(inputs['teams']['dire'])
            pooled_d = torch.cat([pooled_d, team_d], dim=-1)
            # |pooled| : (batch_size, embed_dim+team_emb_dim)

        # --------------------------------------------------------- #
        pooled_r = pooled_r.unsqueeze(1)
        # |pooled| : (batch_size, 1, embed_dim+team_emb_dim)
        pooled_d = pooled_d.unsqueeze(1)
        # |pooled| : (batch_size, 1, embed_dim+team_emb_dim)

        pooled = torch.cat([pooled_d, pooled_r], dim=1)
        # |pooled| : (batch_size, 2, embed_dim+team_emb_dim)

        # Store embeddings
        self.emb_storage['pooled'] = pooled

        pooled = self.compare_fnn(pooled)
        # |pooled| : (batch_size, 2, compare_fnn-embed_dim)

        # Store embeddings
        self.emb_storage['pooled'] = pooled

        # Add pos info
        pooled = pooled + self.compare_embedding(self.__generate_pos_tokens(pooled))
        # |pooled| : (batch_size, 2, compare_fnn-embed_dim)

        if self.regression:
            compare: torch.Tensor = self.compare(pooled)
            # |compare| : (batch_size, 2, compare_fnn-embed_dim)

            # Store embeddings
            self.emb_storage['compared'] = compare

            compare = self.linear(compare)
            # |compare| : (batch_size, 2, len(FEATURES))
            return compare[:, 1], compare[:, 0]
            # |compare| : (batch_size, len(FEATURES))

        else:
            compare: torch.Tensor = self.compare(pooled)
            # |compare| : (batch_size, 2, compare_fnn-embed_dim)

            # Store embeddings
            self.emb_storage['compared'] = compare

            compare = self.linear(compare)
            # |compare| : (batch_size, 2, 1)
            # index 0 - dire
            # index 1 - radiant
            compare = compare.squeeze(2)
            # |compare| : (batch_size, 2)
            return compare


    def __generate_pos_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.arange(inputs.size(1), device=inputs.device, dtype=torch.int64).repeat(inputs.size(0), 1)


    def __forward_through_transformer(self, window: torch.Tensor, key_padding_mask: torch.Tensor=None) -> torch.FloatTensor:
        skip_connection = None
        for idx, layer in enumerate(self.transformer):
            if isinstance(layer, nn.Linear):
                window = layer(window)

            elif isinstance(layer, blocks.TransformerEncoder):
                # if idx%self.models_config['transformer']['skip_connection'] == 0:
                #     if skip_connection is not None:
                #         window = skip_connection + window
                #     window = skip_connection = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
                # else:
                
                # However a ``` `True` value indicates that the corresponding key value will be IGNORED ```
                # for some reasons irl it's the other way around
                window = layer(window, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)

            else: raise Exception
            
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


