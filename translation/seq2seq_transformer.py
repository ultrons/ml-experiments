import torch
import torch.nn as nn
import torch.optim as optim

import spacy
#from  util import translate_sentense

from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# Note: Spacy for quick tockenizing/vocab building etc
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Key_Concept: Field Abstraction
german = Field(tokenize=tokenize_de, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>')

# Note: Torch Text datasets for quick benchmarking work
train_data, valid_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.tgt_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx


    def make_src_mask(self, src):
        #src shape = [S,N], pytorch transformer requires [N,S]
        src_mask = src.transpose(0,1) == self.src_pad_idx
        # Output is [N,S]
        return src_mask

    def forward(self, src, tgt):
        src_seq_length, N = src.shape
        tgt_seq_length, N = tgt.shape

        src_positions = (
            torch.arange(src_seq_length).unsqueeze(1).expand(src_seq_length, N)
            .to(self.device)
        )

        tgt_positions = (
            torch.arange(tgt_seq_length).unsqueeze(1).expand(tgt_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )

        embed_tgt = self.dropout(
            (self.tgt_word_embedding(tgt) + self.tgt_position_embedding(tgt_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_tgt,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        out = self.fc_out(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = False
save_model = True

# Training Hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

# Model Hyperparameters
src_vocab_size = len(german.vocab)
tgt_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100

forward_expansion = 4
src_pad_idx = english.vocab.stoi['<pad>']

# Tensorboard
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    tgt_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

sentence = "ein pfred gent unter einer brucke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    model.train()
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1




