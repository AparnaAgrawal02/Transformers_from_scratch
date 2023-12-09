from transformers_scratch import *
import os 
#import blue score
import numpy as np
import torch
from torchmetrics.text import BLEUScore 



w2i_en = np.load('w2i_en1.npy', allow_pickle=True).item()
i2w_en = np.load('i2w_en1.npy', allow_pickle=True)
w2i_fr = np.load('w2i_fr1.npy', allow_pickle=True).item()
i2w_fr = np.load('i2w_fr1.npy', allow_pickle=True)



embeddding_matrix_en = TextDataset.extract_embeddings('/ssd_scratch/cvit/aparna/glove.42B.300d.txt', w2i_en, 300)
embeddding_matrix_fr =  TextDataset.extract_embeddings('/ssd_scratch/cvit/aparna/glove.42B.300d.txt', w2i_fr, 300)


src_vocab_size = len(w2i_en)
tgt_vocab_size = len(w2i_fr)
print("src_vocab_size = ", src_vocab_size)
print("tgt_vocab_size = ", tgt_vocab_size)
#src_vocab_size, trg_vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix_src, embedding_matrix_trg, dropout=0.1
model = Transformer(src_vocab_size, tgt_vocab_size, 300,512,4,4,embeddding_matrix_en, embeddding_matrix_fr, 0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load("/scratch/aparna/trained_transformer_not_glove1.pt"))
model.eval()
model.to(device)
print("model loaded")

# find blue score

def bleu_index(output, target):
    # convet to word
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(output)
    # print(target)
    output_word = []
    target_word = []
    for i in range(output.shape[0]):
        #ignore padding
        output_word.append([i2w_fr[x] for x in output[i]])
        target_word.append([i2w_fr[x] for x in target[i]])
    # print("output_word = ", output_word)
    # print("target_word = ", target_word)
    #ignore padding
    bleu_score = BLEUScore()
    for i in range(len(output_word)):
        if "<pad>" in target[i]:
            output_word[i] = output_word[i][:output_word[i].index("<pad>")]
            target_word[i] = target_word[i][:target_word[i].index("<pad>")]
    
    output_sentence = [" ".join(x) for x in output_word]
    target_sentence = [" ".join(x) for x in target_word]
    print("output_sentence = ", output_sentence)
    print("target_sentence = ", target_sentence)
    #ignore padding
    output_sentence = [x for x in output_sentence if x != "<pad>"]
    target_sentence = [x for x in target_sentence if x != "<pad>"]
    bleu_score.update(output_sentence,[target_sentence] )
    return bleu_score.compute()


if __name__ == "__main__":
    input = input("Enter a sentence:")
    print("input = ", input)
    input = input.split()
    input = [x.lower() for x in input]
    src = [w2i_en[x] for x in input]
    src = torch.tensor(src).unsqueeze(0)
    src = src.to(device)
    trg = torch.zeros((1,30)).long()
    trg = trg.to(device)

    output = model(src, trg[:,:-1])
    output = output.argmax(dim=-1)
    output = output.cpu().numpy()
    output_word = []
    for i in range(output.shape[0]):
        output_word.append([i2w_fr[x] for x in output[i]])
    output_sentence = [" ".join(x) for x in output_word]
    print("output_sentence = ", output_sentence)
    #print("bleu_score = ", bleu_index(output, tgt[:,1:]))
   
