import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import math

from jiwer import wer

from tokenizer import GraphemeDictionary, PhonemeDictionary

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import torch.nn.functional as F
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(13)

def load_dict(filename):
    with open('./dictionaries/' + filename, 'rb') as f:
            return pickle.load(f)

class PhonemeDataset(torch.utils.data.Dataset):
    def __init__(self, path, graphemedict, phonemedict, max_word_length = 40, max_phoneme_length = 32):
        self.graphemedict = graphemedict
        self.phonemedict = phonemedict
        self.X, self.Y = self.get_data(path, max_word_length, max_phoneme_length)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_data(self, path, max_word_length, max_phoneme_length):
        max_phoneme_length = max_phoneme_length + 2 # account for <START> and <END> tokens. SUPERCALIFRAGILISTICEXPIALIDOCIOUS will be 34 tokens (32 + 2)       
        # Tokenize the file content
        X = []
        Y = []
        with open(path, 'r') as file:
            for index, line in enumerate(file):
                word, phonemes = line.split("  ")
                x = [self.graphemedict.grm2idx[char] for char in word]
                x = x + [self.graphemedict.grm2idx["<PAD>"]] * (max_word_length - len(x))

                phonemes = phonemes.strip().split(" ")
                y = [self.phonemedict.phn2idx["<START>"]] + [self.phonemedict.phn2idx[phn] for phn in phonemes] + [self.phonemedict.phn2idx["<END>"]]
                y = y + [self.phonemedict.phn2idx["<PAD>"]] * (max_phoneme_length - len(y)) 
                X.append(x)
                Y.append(y)
                    
        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)       
        return X,Y

class PhonemeDataModule(L.LightningDataModule):
    def __init__(self, graphemedict, phonemedict, mb, train_path, dev_path, pin_memory=False, num_workers=7):
        super().__init__()

        self.mb = mb
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.trainset = PhonemeDataset(train_path, graphemedict, phonemedict)
        self.devset = PhonemeDataset(dev_path, graphemedict, phonemedict)

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.mb,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.devset, batch_size=self.mb,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def phoneme_length(self):
        return len(self.trainset.phonemedict)
    def grapheme_length(self):
        return len(self.trainset.graphemedict)
    
class PhonemeTransformer(L.LightningModule):
    def __init__(self, graphemedict, phonemedict, grapheme_size, phoneme_size, embed_size, n_head, num_layers, lr, hidden_size):
        super().__init__()
        self.save_hyperparameters()
        self.graphemedict = graphemedict
        self.phonemedict = phonemedict
        self.lr = lr
        self.embed_size = embed_size
        self.grapheme_embed = nn.Embedding(grapheme_size, embed_size)
        self.phoneme_embed = nn.Embedding(phoneme_size, embed_size)
        self.pe = self.positional_encoding(embed_size)
        telayer = nn.TransformerEncoderLayer(embed_size, n_head, batch_first=True, dim_feedforward=hidden_size)
        self.encoder = nn.TransformerEncoder(telayer, num_layers)
        tdlayer = nn.TransformerDecoderLayer(embed_size, n_head, batch_first=True, dim_feedforward=hidden_size)
        self.decoder = nn.TransformerDecoder(tdlayer, num_layers)
        self.linear = nn.Linear(embed_size, phoneme_size)

    def positional_encoding(self, embed_size):
        pe = torch.zeros(40, embed_size)
        position = torch.arange(0, 40).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_size, 2, dtype=torch.float) * -(math.log(10000.0) / embed_size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe

    def forward(self, x, y):
        x_padding_mask = (x == 0)
        y_padding_mask = (y == 0)
        x = self.grapheme_embed(x)
        x = x.to(device)
        
        T = x.shape[1]
        pos_emb_x = self.pe[:T, :].unsqueeze(0).to(device) # [1, 40, 128]
        x = x + pos_emb_x

        encoder_output = self.encoder(x, src_key_padding_mask=x_padding_mask)

        y = self.phoneme_embed(y)
        y = y.to(device)

        y_len = y.shape[1]
        pos_emb_y = self.pe[:y_len, :].unsqueeze(0).to(device)
        y = y + pos_emb_y
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(y_len).to(device)
        decoder_output = self.decoder(y, encoder_output, tgt_mask = tgt_mask, memory_key_padding_mask = x_padding_mask, tgt_key_padding_mask = y_padding_mask) # batch size, T, embed_size
        return self.linear(decoder_output) # batch_size, T, phoneme_size

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_input = y[:, :-1] # removing last to make match targets
        y_targets = y[:, 1:]
        y_pred = self(x, y_input)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1]) # From [20, 33, 72] to [660, 72]
        y_targets = y_targets.reshape(-1) # From [20, 33] to [660]
        loss = F.cross_entropy(y_pred, y_targets, ignore_index=0)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predicted_outputs = self.generate(x) # [20, 34]
        loss = self.calculate_per(y, predicted_outputs)
        self.log("per", loss, prog_bar=True)
        return loss

    def calculate_per(self, true, predicted):
        batch_size = true.shape[0]
        # print(predicted)
        true_phonemes = [self.ids_to_phonemes(x) for x in true]
        pred_phonemes = [self.ids_to_phonemes(x) for x in predicted]
        loss = 0
        for i in range(batch_size):
            # print(f"Comparing true {true_phonemes[i]} to pred {pred_phonemes[i]}")
            loss += wer(true_phonemes[i], pred_phonemes[i])
        return loss/batch_size

    def generate(self, x, max_new_tokens=32):
        self.eval()
        pad_token = 0
        start_token = 1
        end_token = 2
        batch_size = x.shape[0]
        outputs = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)

        # Has this word generated <END> yet?
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                
                logits = self(x, outputs)
                logits = logits[:, -1, :] # B,C, [20,72]
                probs = F.softmax(logits, dim = -1)
                # print(probs[0])
                idx_next = torch.argmax(probs, dim = 1).unsqueeze(1) # NOT multinomial, because I want it to be greedy?
                
                # idx_next = torch.multinomial(probs, 1)
                # print(idx_next)
                idx_next[done] = pad_token # for completed sequences, just add padding

                outputs = torch.cat((outputs, idx_next), dim=1)

                done = done | (idx_next == end_token).squeeze(1)
        end_token_tensor = torch.full((batch_size,1), end_token).to(device)
        end_token_tensor[done] = pad_token
        outputs = torch.cat((outputs, end_token_tensor), dim=1)
        self.train()
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def ids_to_graphemes(self, ids):
        return "".join([self.graphemedict.idx2grm[int(idx)] for idx in ids if int(idx) != 0])

    def ids_to_phonemes(self, ids):
        return " ".join([self.phonemedict.idx2phn[int(idx)] for idx in ids if int(idx) not in [0,1,2]])

def parse_all_args():
    # Parses commandline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-train_path",type=str,\
            help="Path to training data", default='/local/202510_csci581_project/project_data/task3/train.txt')
    parser.add_argument("-dev_path",type=str,\
            help="Path to dev data", default='/local/202510_csci581_project/project_data/task3/dev.txt')
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]", default=0.0002)
    parser.add_argument("-mb",type=int,\
        help="The minibatch size (num sequences per batch) rate (int) [default: 20]", default=800)
    parser.add_argument("-embed_sz",type=int,\
        help="The input embedding dimension (int) [default: 128]", default=128)
    parser.add_argument("-L", type=int, default=256,\
                        help='The dimension of the hidden units (int) [default: 1024]')
    parser.add_argument("-K", type=int, default=3,\
                        help='The number of (stacked) layers for both encoders and decoders (int) [default: 4]')
    parser.add_argument("-epochs",type=int, default=20,\
            help="The number of training epochs (int) [default: 5]")
    parser.add_argument("-H",type=int,\
            help="The number of attention heads in each decoder layer", default=8)
    return parser.parse_args()

def main():
    args = parse_all_args()

    max_word_length = 40
    max_phoneme_length = 32
    graphemedict = load_dict("GraphemeDictionary.pkl") # defined in tokenizer.py, has grm2idx
    phonemedict = load_dict("PhonemeDictionary.pkl") # defined in tokenizer.py, has phn2idx and idx2phn

    data = PhonemeDataModule(graphemedict, phonemedict, args.mb, args.train_path, args.dev_path)
    grapheme_dict_size = data.grapheme_length()
    phoneme_dict_size = data.phoneme_length()
    model = PhonemeTransformer(graphemedict, phonemedict, grapheme_dict_size, phoneme_dict_size, args.embed_sz, args.H, args.K, args.lr, args.L)

    checkpoint_callback = ModelCheckpoint(
        monitor="per",        # Monitor validation accuracy
        mode="min",               # Save when validation accuracy is new maximum 
        filename="best-model-{epoch:02d}",
        save_top_k=1,             # Keep only the best model
        verbose=True
    )

    trainer = L.Trainer(max_epochs=250, accelerator="auto", callbacks = [checkpoint_callback], check_val_every_n_epoch=1, limit_test_batches=0)
    model_path = "/home/korontd/cs581/code/task3/lightning_logs/version_97/checkpoints/best-model-epoch=221.ckpt"
    trainer.fit(model, data, ckpt_path=model_path)

    best_model_path = checkpoint_callback.best_model_path

if __name__ == "__main__":
    main()