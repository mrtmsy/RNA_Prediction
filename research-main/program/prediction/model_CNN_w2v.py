import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from einops_exts import repeat_many, rearrange_many

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # calculate query / key similarities
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class OuterProduct(nn.Module):
    def __init__(
            self,
            dim
        ):
        super().__init__()
        self.eps = 1e-5
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        #x = self.norm(x)
        left = self.proj_in(x)
        right = self.proj_in(x)

        outer = rearrange(left, 'b i d -> b i () d') * rearrange(right, 'b j d -> b () j d')

        #return outer
        return self.proj_out(outer)


class Net(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        
        #Embedding層
        #111個分のRNA塩基に64次元で埋め込む  
        #6*dimの重みを生成
        # self.token_embedding = nn.Embedding(
        #     cfg.embed.token_size, cfg.dim
        # )
        
        #ポジションの情報を111→dim次元に圧縮
        #self.pe_embedding = nn.Linear(cfg.embed.pe_dim, cfg.dim)
        
        #１次元畳み込み層
        self.one_conv_layer = nn.Sequential(
            nn.Conv1d(cfg.dim, cfg.dim * 2, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(3, stride = 1, padding = 1),
            nn.Conv1d(cfg.dim * 2, cfg.dim, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(3, stride = 1, padding = 1)
            )
        
        self.product = OuterProduct(cfg.dim)
        
        self.two_conv_layer = nn.Sequential(
            nn.Conv2d(cfg.dim, cfg.dim * 2, 3, stride=1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding = 1),
            nn.Conv2d(cfg.dim * 2, cfg.dim, 3, stride=1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding = 1)
            )
            
        
        self.last_layer = nn.Sequential(
            nn.Linear(cfg.dim, int(cfg.dim/2)),
            nn.ReLU(),
            nn.Linear(int(cfg.dim/2), 1),
            nn.Sigmoid()
        )
        
        #self.attn = Attention(dim = cfg.dim)
        #self.fft_1 = FeedForward(dim = cfg.dim, mult = cfg.encoder.ff_mult)
        #self.cross_attn = CrossAttention(dim = cfg.dim)
        #self.fft_2 = FeedForward(dim = cfg.dim, mult = cfg.encoder.ff_mult)
        
        depth = cfg.encoder.depth
        self.attn = nn.ModuleList([])
        for _ in range(depth):
            self.attn.append(nn.ModuleList([
                Attention(dim = cfg.dim),
                FeedForward(dim = cfg.dim, mult = cfg.encoder.ff_mult)
            ]))
        
    def forward(self, token, pe):
        # x = self.token_embedding(token) 
        x = token.squeeze(1) + self.pe_embedding(pe)
        x = rearrange(x, 'b n d -> b d n')
        x = self.one_conv_layer(x)
        x = rearrange(x, 'b d n -> b n d')
        
        x = self.product(x)
        x = rearrange(x, 'b i j d -> b d i j')
        x = self.two_conv_layer(x)
        x = rearrange(x, 'b d i j -> b i j d')
            
        out = self.last_layer(x)
        
        out = rearrange(out, 'b i j n-> b (i j n)')
        out, _ = torch.max(out, dim = 1)
        
        return out
            
        
        
        
        
        
        













