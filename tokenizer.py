# This file creates dictionaries which map from phonemes to tokens, tokens to phonemes, and graphemes to tokens.
# They can then be loaded and used to tokenize for use in Pytorch Datasets

import pickle

data_dir = '/local/202510_csci581_project/project_data/task3/'

class PhonemeDictionary(object):
    def __init__(self):
        self.phn2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2}
        self.idx2phn = {0: "<PAD>", 1: "<START>", 2: "<END>"}
        self.idx = 3
    
    def add_sound(self, phn):
        if not phn in self.phn2idx:
            self.phn2idx[phn] = self.idx
            self.idx2phn[self.idx] = phn
            self.idx += 1
    
    def __len__(self):
        return len(self.phn2idx)
    
class GraphemeDictionary(object):
    def __init__(self):
        self.grm2idx = {"<PAD>": 0}
        self.idx2grm = {0: "<PAD>"}
        self.idx = 1
    
    def add_char(self, grm):
        if not grm in self.grm2idx:
            self.grm2idx[grm] = self.idx
            self.idx2grm[self.idx] = grm
            self.idx += 1
    
    def __len__(self):
        return len(self.grm2idx)
    

if __name__ == "__main__":
    gd = GraphemeDictionary()
    pd = PhonemeDictionary()
    char_dict = {}

    with open(data_dir + 'train.txt', 'r') as file:
        for index, line in enumerate(file):
            word, phonemes = line.split("  ")
            for char in word:
                if not char in char_dict:
                    char_dict[char] = index

    for key in sorted(char_dict):
        gd.add_char(key)

    with open(data_dir + 'phonemes.txt', 'r') as file:
        for line in file:
            pd.add_sound(line.strip())

    with open('./dictionaries/GraphemeDictionary.pkl', 'wb') as file:
        pickle.dump(gd, file)

    with open('./dictionaries/PhonemeDictionary.pkl', 'wb') as file:
        pickle.dump(pd, file)

    print(pd.phn2idx)
    print(pd.idx2phn)

    print(gd.grm2idx)