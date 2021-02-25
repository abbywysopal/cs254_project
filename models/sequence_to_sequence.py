import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

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
import torch
import tokenizer
from torch.utils.data import DataLoader
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

valid_filepath, test_filepath, train_filepath = glob.glob("tests/data/*")
print(valid_filepath)
print(test_filepath)

# tokenizer = get_tokenizer('basic_english')
# vocab = build_vocab_from_iterator(map(tokenizer,
#                                       iter(io.open(train_filepath,
#                                                    encoding="utf8"))))

def data_process(path):
    files = glob.glob(path + "/json/*.txt")
    dataset = []
    index = 0
    for filename in files:
        file = open(filename)
        data = file.readlines()
        dataset.append(data)
        index += 1

    return DataLoader(dataset, batch_size=index, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None)

    # return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_loader = data_process(train_filepath)
print(train_loader)
for i, batch in enumerate(train_loader):
    print(i, batch)
val_loader = data_process(valid_filepath)
test_loader = data_process(test_filepath)

# train_data = data_process(iter(io.open(train_filepath, encoding="utf8")))
# val_data = data_process(iter(io.open(valid_filepath, encoding="utf8")))
# test_data = data_process(iter(io.open(test_filepath, encoding="utf8")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(loader, bsz):
    data = iter(loader)
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
# train_data = batchify(train_data, batch_size)
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)