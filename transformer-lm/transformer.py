"""
Handwritten version of PyTorch Transformer Tutorial on Official Website
The tutorial walks through the nn.TransformerEncoder Module on the language
modeling task.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLanguageModel(nn.Module):
    """
    Word Language Model
    Args:
    ntoken: Vocab size.
    ninp: Size of input to the transformer, usually embedding size
    nhead: Number of attention heads.
    nhid: Transformer Encoder Parameter
    nlayers: Number of Transformer Layer
    dropout: dropout probability
    """
    def __init__(
        self,
        ntokens,
        ninp,
        nheads,
        nhid,
        nlayers,
        dropout=0.5
    ):
        super(TransformerLanguageModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nheads, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntokens)

        self.init_weights()


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) *  math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """
    d_model: refers to the data input dimension for the model
    or better known as the embedding size
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.tensorboard import SummaryWriter


url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
tokenizer = get_tokenizer('basic_english')
# Reads the file (open returns an iterable)
# Then converts it into an iterator using iter() method
# Then calls tokenizer method when iterated upon
# Nice
vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding='utf8'))))

def data_process(raw_text_iter):
    """ Convert Raw Text Iterator to Tensor Iterator """
    data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_data = data_process(iter(io.open(train_filepath, encoding='utf8')))
val_data = data_process(iter(io.open(valid_filepath, encoding='utf8')))
test_data = data_process(iter(io.open(test_filepath, encoding='utf8')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off the extra elements that wouldn't cleanly fit (remainders)
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # Notice (bsz, -1), instead of (-1, bsz)
    # such that columns have the sequence
    # Hence division into bsz equal chunks
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, batch_size)
test_data = batchify(test_data, batch_size)

# Concept/Note:
# The following function generatest the input and target sequence for the transformer model
# It further subdevides the source data input chunks of length bptt (commonly known as the context size)
# or S dimension of the transformer

bptt = 35
def get_batch(source, i):
    # Grabbing sequence length worth of the rows, bsz columns
    seq_len = min(bptt, len(source) - 1 -i)
    data = source [i:i+seq_len]
    target = source [i+1:i+1+seq_len].reshape(-1)
    return data, target

# Model hyperparameters
ntokens = len(vocab.stoi) # The size of vocabulary
emsize = 200 # embedding size
nhid = 200 # Dimension of the feedforward network model in the nn.TransformerEncoder
nlayers = 2 # Dimension of the nn.TransformerEncoderLayer
nheads = 2 # Number of attention heads
dropout = 0.2 # the dropout probability
model = TransformerLanguageModel(ntokens, emsize, nheads, nhid, nlayers, dropout).to(device)

writer = SummaryWriter("runs/loss_plot")

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer =  torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train()
    total_loss = 0.
    global_step = 0
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    # NOTE: Check the dimensions here
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Init the optimizer ?
        # NOTE: Check what zero_grad means
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss +=loss.item()
        global_step += 1
        writer.add_scalar("Training loss", loss, global_step=global_step)
        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed_time = time.time() - start_time
            print('|epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                elapsed_time * 100 / log_interval, cur_loss, math.exp(cur_loss)
            ))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0), bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float('-inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-'*89)
    print('|end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
    ))
    print('-'*89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    scheduler.step()

