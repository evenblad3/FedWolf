import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

x = torch.randn(size=(1, 3, 224, 224))
'''
patch_size = 16
patchs = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
# rearrange里面的(h s1)表示hxs1,而s1是patch_size=16, 那通过hx16=224可以算出height里面包含了h个patch_size(14个)
# 输出b (h w) (s1 s2 c)，这相当于把每个patch(16x16x3)拉成一个向量，每个batch里面有hxw个这样的向量
print(patchs.size())
# 1 196 768
# 196=14x14
# 768=16x16x3
'''

'''数据嵌入
拆成很多个patch,在每个patch向量前面加上可学习的位置信息
'''


class PatchEmbeddingViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbeddingViT, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 1 1 768，768=16x16x3
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        # (196+1)x768=197x768

    def forward(self, x):
        b = x.size(0)
        x = self.proj(x)  # 1 196 768
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)  # 1 1 768
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions

        return x


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        # n个heads同时计算
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        # 利用queries和values计算attention矩阵
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_ken
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)  # batch head values_len
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)

        return out


# Residuals
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


# MLP
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


# 组合TransformerEncoderBlock
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                ),
                nn.Dropout(drop_p)
            ))
        )
        

# classification
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes=1000):
        super(ClassificationHead, self).__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


# 组合ViT
class VitNet(nn.Sequential):
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_size=768,
                 img_size=224,
                 n_classes=1000,
                 **kwargs):
        super(VitNet, self).__init__(
            PatchEmbeddingViT(in_channels, patch_size, emb_size, img_size),
            TransformerEncoderBlock(emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


print('输入图像:', x.size())
out = VitNet()(x)
print('输出尺寸:', out.size())
