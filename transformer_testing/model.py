import torch
import torch.nn as nn 
import math


class LayerNormalization(nn.Module):
    def __init__(self, features, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    


class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,dropout):
        self.l1=nn.Linear(d_model, 4* d_model )
        self.dropout=dropout
        self.l2= nn.Linear(4*d_model,d_model)
    def forward(self, x):
        return self.l2(self.dropout(torch.relu(self.l1(x))))
    
class InputEmbeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model,h, dropout):
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.d_head= d_model/h
        self.wq= nn.Linear(self.d_model,self.d_model)
        self.wv= nn.Linear(self.d_model,self.d_model)
        self.wk= nn.Linear(self.d_model,self.d_model)
        self.wo= nn.Linear(self.d_model,self.d_model)
        self.dropout = dropout
   
    def attention(query , key , value ,mask, dropout):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(2,3) ) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            scores.masked_fill_(mask == 0, -1e9)
        scores=scores.softmax(-1)
        return (scores @ value), scores
    
    def forward(self, q , k , v,mask):
        self.q=self.wq(q)
        self.k=self.wk(k)
        self.v=self.wv(v)


        q= q.view(q.shape[0],q.shape[1],self.h,self.d_head).Tranpose(1,2)# batch size , h,seq len , ,head embd, 
        k= k.view(k.shape[0],k.shape[1],self.d_head,self.h).Tranpose(1,2)
        v= v.view(v.shape[0],v.shape[1],self.d_head,self.h).Tranpose(1,2)


        x , attention_scores=MultiHeadAttentionBlock.attention(q,k,v,mask,self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_model)
        return self.wo(x)


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))


    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, features ,self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock,feed_forward_block: FeedForwardBlock,dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)



class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    



class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size,tgt_vocab_size, src_seq_len,tgt_seq_len,d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048)->Transformer:
    src_embed  = InputEmbeddings(d_model, src_vocab_size )
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)
    src_pos=PositionalEncoding(d_model,src_seq_len)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    