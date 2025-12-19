import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os

from .attn import REAttention, AttentionLayer, GaussianLayer, GaussianKernel
from .embed import DataEmbedding, TokenEmbedding

seed_value = 5   # set random seed

np.random.seed(seed_value)

os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.patch_size = patch_size

    def forward(self, x, pre, attn_mask=None):
        batch_size = x.shape[0]
        x = rearrange(x, "b (l1 l2) d -> (b l1) l2 d", l2=self.patch_size)
        new_x, attn = self.attention(
            x, x, x,
            pre,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = rearrange(self.norm2(x + y), "(l1 l2) p d -> l1 (l2 p) d", l1=batch_size)
        return x, attn


class AEEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layers=None):
        super(AEEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norms = norm_layers

    def forward(self, x, layer, pre=None, attn_mask=None):
        x, series = self.attn_layers[layer](x, pre, attn_mask=attn_mask)

        if self.norms is not None:
            x = self.norms[layer](x)

        return x, series


class GaussEncoder(nn.Module):
    def __init__(self, attn_layers, patch_size):
        super(GaussEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.patch_size = patch_size

    def forward(self, x, layer):
        prior, sigma = self.attn_layers[layer](x)
        return prior, sigma


class NRGTrans(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, patch2_size=25):
        super(NRGTrans, self).__init__()
        self.output_attention = output_attention
        self.e_layers = e_layers

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.patch_sizes = [100]
        self.patch_sizes.append(patch2_size)
        self.win_size = win_size
        # global association
        self.encoder_1 = AEEncoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            REAttention(self.patch_sizes[0], False, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        self.patch_sizes[0],
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
            norm_layers=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.e_layers)])
            )
        # neighbor association
        self.encoder_2 = AEEncoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            REAttention(self.patch_sizes[1], False, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        self.patch_sizes[1],
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
            norm_layers=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.e_layers)])
            )

        # prior association
        self.gauss_encoder = GaussEncoder(
                                [
                                    GaussianLayer(
                                        GaussianKernel(win_size, output_attention), d_model, n_heads
                                    ) for _ in range(e_layers)
                                ], self.patch_sizes[0]
                            )

        self.linear_layers = nn.ModuleList([nn.Linear(d_model*2, d_model) for _ in range(self.e_layers)])

        self.norm = nn.LayerNorm(d_model)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.e_layers)])
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

    def cat_series(self, series_1, series_2, patch):
        patch_num = int(self.win_size / self.patch_sizes[patch])
        zero_tensor = torch.zeros_like(series_1)
        s = zero_tensor.shape[0]
        b, h, _, _ = series_2.shape

        split_list = []

        split_tensors_1 = torch.split(zero_tensor, split_size_or_sections=self.patch_sizes[patch], dim=2)
        for row in range(patch_num):
            split_tensors_2 = torch.split(split_tensors_1[row], split_size_or_sections=self.patch_sizes[patch], dim=3)
            split_row = list(split_tensors_2)
            split_list.append(split_row)

        for i1 in range(b):
            for i2 in range(h):
                target_split = split_list[i1 % patch_num][i1 % patch_num]
                target_split[i1 // patch_num][i2] = series_2[i1][i2].detach()
        cat_list = []
        for row in range(patch_num):
            cat_list.append(torch.cat(split_list[row], dim=3))
        zero_tensor = torch.cat(cat_list, dim=2)
        return zero_tensor

    def forward(self, x):
        enc_out = self.embedding(x)

        series_list = []
        series_list_cat_2 = []
        prior_list = []
        sigma_list = []
        series_1 = None
        series_2 = None
        for l in range(self.e_layers):
            prior, sigma = self.gauss_encoder(enc_out, l)
            enc_1_out, series_1 = self.encoder_1(enc_out, l, series_1)
            enc_2_out, series_2 = self.encoder_2(enc_out, l, series_2)

            enc_out = torch.cat((enc_1_out, enc_2_out), dim=2)

            enc_out = self.activation(self.linear_layers[l](enc_out))
            # enc_out = enc_1_out

            enc_out = self.norm_layers[l](enc_out)

            series_2_cat = self.cat_series(series_1, series_2, 1)

            series_list.append(series_1)
            prior_list.append(prior)
            sigma_list.append(sigma)
            series_list_cat_2.append(series_2_cat)

        enc_out = self.norm(enc_out)

        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series_list, prior_list, sigma_list, series_list_cat_2
        else:
            return enc_out  # [B, L, D]