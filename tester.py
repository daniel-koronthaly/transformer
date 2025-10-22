from transformer import PhonemeTransformer, load_dict
from tokenizer import GraphemeDictionary
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    ckpt_path = "/home/korontd/cs581/code/task3/lightning_logs/version_101/checkpoints/best-model-epoch=261.ckpt"
    max_word_length = 40
    model = PhonemeTransformer.load_from_checkpoint(ckpt_path)
    graphemedict = load_dict("GraphemeDictionary.pkl")
    X = []
    input_words = []
    with open('/local/202510_csci581_project/project_test_data/task3_test.txt', 'r') as file:
        for index, line in enumerate(file):
            input_words.append(line.strip().upper())
            x = [graphemedict.grm2idx[char] for char in line.strip().upper()]
            x = x + [graphemedict.grm2idx["<PAD>"]] * (max_word_length - len(x))
            X.append(x)
        
        X = torch.tensor(X, dtype=torch.long).to(device)

    outputs = model.generate(X)
    with open('task3_predictions.txt', 'w') as file:
        for i, out in enumerate(outputs):
            file.write(input_words[i] + '  ' + model.ids_to_phonemes(out) + '\n')
    # print(outputs)



if __name__ == "__main__":
    main()