import re
import pandas as pd
from string import punctuation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import math 
from torch.utils.data import Dataset, DataLoader
import torchtext.vocab as vocab 
import torch
import wandb

# wandb.init(project="Transformers")
# wandb.config.dropout = 0.05
# wandb.config.batch_size = 32
# wandb.config.learning_rate = 0.001
# wandb.config.epochs = 500
# wandb.config.hidden_dim1 = 150
# wandb.config.hidden_dim2 = 150

pad_index = 0
unk_index = 1
start_index = 2
separator_index = 3
import time






# Function to get word index from vocabulary
def get_word(word,glove_vocab):
    if word not in glove_vocab:
        return unk_index
    return glove_vocab[word]

class TextDataset(Dataset):
    def __init__(self, tokens_en, tokens_fr, textsize):
        self.tokens_en = tokens_en
        self.tokens_fr = tokens_fr
        self.textsize = textsize

    def __len__(self):
        return min(len(self.tokens_en), len(self.tokens_fr))

    def __getitem__(self, idx):
        x = self.tokens_en[idx][:30]
        y = self.tokens_fr[idx][:30]
        if len(x)<self.textsize:
            x.extend([pad_index]*(self.textsize-len(x)))
        if len(y)<self.textsize:
            y.extend([pad_index]*(self.textsize-len(y)))
        x = torch.from_numpy(np.array(x, dtype=np.int64))
        y = torch.from_numpy(np.array(y, dtype=np.int64))
        # print("x shape: ", x.shape)
        # print("y shape: ", y.shape)
        return {"x": x, "y": y}

    @staticmethod
    def vocab(data,glove_vocab) :
        w_counts = {}
        for sent in data:
            for word in sent:
                if word in w_counts:
                    w_counts[word] += 1
                else:
                    w_counts[word] = 1
        
        word_2_index = {'<pad>': 0,'<unk>': 1,'<start>': 2,'<separator>': 3}
        index_2_word = ['<pad>', '<unk>', '<start>', '<separator>']
        curr_ix = 4
        wc = 0
        for sent in data:
            for word in sent:
                wc += 1
                if word in word_2_index: 
                    continue
                if w_counts[word] < 3:
                    continue
                if word not in glove_vocab:
                    continue
                word_2_index[word] = curr_ix
                index_2_word.append(word)
                curr_ix += 1
        return word_2_index, index_2_word, wc
    @staticmethod
    def vocab_without_glove(data,glove_vocab) :
        w_counts = {}
        for sent in data:
            for word in sent:
                if word in w_counts:
                    w_counts[word] += 1
                else:
                    w_counts[word] = 1
        
        word_2_index = {'<pad>': 0,'<unk>': 1,'<start>': 2,'<separator>': 3}
        index_2_word = ['<pad>', '<unk>', '<start>', '<separator>']
        curr_ix = 4
        wc = 0
        for sent in data:
            for word in sent:
                wc += 1
                if word in word_2_index: 
                    continue
                if w_counts[word] < 3:
                    continue
                # if word not in glove_vocab:
                #     continue
            
                word_2_index[word] = curr_ix
                index_2_word.append(word)
                curr_ix += 1
        return word_2_index, index_2_word, wc
    @staticmethod
    def extract_embeddings(filepath, word_2_index, embedding_dim):
        vocab_size = len(word_2_index)
        
        embeddings = np.zeros((vocab_size, embedding_dim))
        embeddings[0] = torch.rand(embedding_dim)
        embeddings[1] = torch.rand(embedding_dim)
        embeddings[2] = torch.rand(embedding_dim)
        embeddings[3] = torch.rand(embedding_dim)
    
        with open(filepath, encoding="utf8") as f:
            for line in f:
                word, *vector = line.split()
                if word in word_2_index:
                    idx = word_2_index[word]
                    embeddings[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]

        return torch.from_numpy(embeddings)
    @staticmethod
    def tokens_to_index(tokens,w2i):
        embeds = []
        for token in tokens:
            embeds.append([get_word(word,w2i) for word in token])
        return embeds

    @staticmethod
    def tokenization(file):
        with open(file, 'r') as myfile:
            text = myfile.read()
            sentences = text.split('.')
            tokens = []
            for sentence in sentences:
                text = TextDataset.preprocess_text(sentence)
                if text != "":
                    x = text.split()
                    sen_tokens = ['<start>']+ x
                    tokens.append(sen_tokens)
        return tokens

    @staticmethod
    def preprocess_text(text):
        text = text.casefold()
        text = re.sub(f"[{re.escape(punctuation)}]", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", text)
        text = re.sub(r"\b[0-9]+\b\s*", "", text)
        #remmove mails
        text = re.sub(r"\S+@\S+", "", text)
        text = " ".join(text.split())
        return text
    
    @staticmethod
    def split_dataset(test_len, val_len,file):
        tokens = TextDataset.tokenization(file)
        # split 
        test_tokens = tokens[:test_len]
        val_tokens = tokens[test_len:test_len + val_len]
        train_tokens = tokens[test_len + val_len:]
        return train_tokens, test_tokens, val_tokens
    

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model_dim, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model_dim = d_model_dim
        pe = torch.zeros(max_len, d_model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_dim, 2) * -(math.log(10000.0) / d_model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model_dim)
        #print("x shape: ", x.shape)
        len = x.size(1)
        #constant 
        pe = self.pe[:len, :]
        pe = pe.unsqueeze(0)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
       #print("x shape: ", x.shape)
        return self.dropout(x)

class MUltiHeadAttention(nn.Module):
    def __init__(self, d_model_dim, heads):
        super(MUltiHeadAttention, self).__init__()
        self.d_model_dim = d_model_dim
        self.heads = heads
        self.d_k = d_model_dim // heads
        self.q = nn.Linear(self.d_k, self.d_k, bias=False)
        self.k = nn.Linear(self.d_k, self.d_k, bias=False)
        self.v = nn.Linear(self.d_k, self.d_k, bias=False)
        self.out = nn.Linear(self.heads*self.d_k, self.d_model_dim, bias=False)
    def forward(self, qurey, key, value, mask=None):
       # print("key shape: ", key.shape)
        key = key.view(key.size(0),key.size(1), self.heads, self.d_k)
       # print("key shape: ", key.shape)
        qurey = qurey.view(qurey.size(0),qurey.size(1), self.heads, self.d_k)
        value = value.view(value.size(0),value.size(1),self.heads, self.d_k)
        key = self.k(key)
        #print("key shape: ", key.shape)
        qurey = self.q(qurey)
        value = self.v(value)
        key = key.transpose(1,2)      # (batch_size, n_heads, seq_len, single_head_dim) 
       # print("key shape: ", key.shape)
        qurey = qurey.transpose(1,2)
        value = value.transpose(1,2)
        #attention
        scores = torch.matmul(qurey, key.transpose(-1, -2)) / math.sqrt(self.d_k) # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
             #mask = mask.unsqueeze(1)
           # print("mask shape: ", mask.shape)
            #print("scores shape: ", scores.shape)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        #multiply scores with value
        scores = torch.matmul(scores, value)
        #concatenate heads
        scores = scores.transpose(1,2).contiguous()
        scores = scores.view(scores.size(0), -1, self.heads*self.d_k)
        #print("scores shape: ", scores.shape)
        scores = self.out(scores)
        #print("scores shape: ", scores.shape)
        return scores


class TransformerBlock(nn.Module):
    #d_model_dim: embedding dimension
    #heads: number of heads

    def __init__(self, d_model_dim, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MUltiHeadAttention(d_model_dim, heads)
        self.norm1 = nn.LayerNorm(d_model_dim)
        self.norm2 = nn.LayerNorm(d_model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model_dim, d_model_dim)
    def forward(self, qurey, key, value, mask=None):
        
        x = self.attention(qurey, key, value, mask)
        #residual
        x = self.dropout1(x)
        norm1 = self.norm1(x + qurey)
        #feed forward
        x = self.fc(norm1)
        x = self.dropout2(x)
        x = self.norm2(x + norm1)
    
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix):
        super(TransformerEncoder, self).__init__()
        #self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, d_model_dim)
        self.pe = PositionalEncoding(d_model_dim)
        self.layers = nn.ModuleList([TransformerBlock(d_model_dim, heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        x = self.embedding(x).type(torch.FloatTensor).to(x.device)
        x = self.fc(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model_dim, heads, dropout=0.01):
        super(Decoder, self).__init__()
        self.attention = MUltiHeadAttention(d_model_dim, heads)
        self.norm = nn.LayerNorm(d_model_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer_block = TransformerBlock(d_model_dim, heads)
    def forward(self, x, encoder_out, src_mask, trg_mask):
        at = self.attention(x, x, x, trg_mask)
        x = self.norm(x + self.dropout(at))
        x = self.transformer_block(x, encoder_out, encoder_out, src_mask)
        return x

    
    
class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix,dropout=0.01):
        super(TransformerDecoder, self).__init__()
        #self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, d_model_dim)
        self.pe = PositionalEncoding(d_model_dim)
        self.layers = nn.ModuleList([Decoder(d_model_dim, heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model_dim, vocab_size)
    def forward(self, x, encoder_out, src_mask, trg_mask):
        x =self.embedding(x).type(torch.FloatTensor).to(x.device)
        #print("x11 shape: ", x.shape)
        x = self.fc1(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, trg_mask)
        x = self.fc(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self,src_vocab_size, trg_vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix_src, embedding_matrix_trg, dropout=0.01):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix_src)
        self.decoder = TransformerDecoder(trg_vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix_trg)
    def target_mask(self, trg):
        trg_pad_mask = (trg != pad_index).unsqueeze(1).unsqueeze(2)
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(trg.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = (src != pad_index).unsqueeze(1).unsqueeze(2)
        
        #print(src_mask)
       # print("src mask shape: ", src_mask.shape)
        trg_mask = self.target_mask(trg)
        #print("mask" ,trg_mask)
       # print("trg mask shape: ", trg_mask.shape)
        encoder_out = self.encoder(src, src_mask)
        out = self.decoder(trg, encoder_out, src_mask, trg_mask)
        #generation step
        out = F.log_softmax(out, dim=-1)
        return out
    


