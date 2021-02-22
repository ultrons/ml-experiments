import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data, datasets, vocab

class MultiheadAttention(nn.Module):
    """
    Implementation of Multihead Attention Module
    Hat tip to Peter Bloem for very well written
    http://peterbloem.nl/blog/transformers
    """
    def __init__(
        self,
        embedding_size,
        num_heads
    ):
        super(MultiheadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Query, Key and Value metrices expanded for all heads
        # ref: wide attention
        self.toquery = nn.Linear(embedding_size, num_heads*embedding_size)
        self.tokey = nn.Linear(embedding_size, num_heads*embedding_size)
        self.tovalue = nn.Linear(embedding_size, num_heads*embedding_size)

        # Unify: final Stage
        self.unifyheads = nn.Linear(num_heads*embedding_size, embedding_size)

    def forward(self, x):
        b, t, k = x.size()
        h = self.num_heads

        # Output of each linear layer is [b, t, h*k]
        # hence the view
        queries = self.toquery(x).view(b, t, h, k)
        keys = self.tokey(x).view(b, t, h, k)
        values = self.tovalue(x).view(b, t, h, k)


        # Fold head into batch dimension
        queries = queries.transpose(1,2).contiguous().view(b*h, t, k)
        keys = keys.transpose(1,2).contiguous().view(b*h, t, k)
        values = values.transpose(1,2).contiguous().view(b*h, t, k)

        # Scale before the dot product
        queries /= k ** (1/4)
        keys /= k ** (1/4)

        # Dot product (query,Key)
        # i.e the raw weights for the weighted sum ahead
        # dimension: [b*h, t, t]
        weights = torch.bmm(queries, keys.transpose(1, 2))
        # softmax along time dimension [Last]
        weights = F.softmax(weights, dim=2)

        # Multiply with Value matrix
        out = torch.bmm(weights, values).view(b, h, t, k)

        # To unify we transpose again such that the head
        # and embedding dimension are next to one another
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)

        # Unify and Out
        return self.unifyheads(out)


class TransformerLayer(nn.Module):
    """
    The following is intended to be reusable/stackable layer

    """
    def __init__(
        self,
        embedding_size,
        num_heads
    ):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.ff = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size)
        )

    def forward(self,x):
        attended = self.attention(x)
        # Layer norm and residual connection
        x = self.norm1(x) + attended
        x = self.ff(x)
        # Layer norm and residual connection
        return self.norm2(x) + x


class sentimentClassifier(nn.Module):
    """  """
    def __init__(
        self,
        embedding_size,
        num_heads,
        num_layers,
        max_seq_length,
        vocab_size,
        num_classes
    ):
        super(sentimentClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, embedding_size)
        self.pos_emb = nn.Embedding(max_seq_length, embedding_size)

        transformer_layers = [
            TransformerLayer(embedding_size, num_heads)
            for _ in range(num_layers)
        ]
        self.transformer_layers = nn.Sequential(*transformer_layers)
        self.tprobs = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        pos = self.pos_emb(torch.arange(t))
        pos = pos[None, :, :].expand(b, t, k)
        x = tokens + pos
        x = self.transformer_layers(x)
        x = self.tprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
def main():
    writer = SummaryWriter('runs/summary')

    train, test = datasets.IMDB.split(TEXT, LABEL)




