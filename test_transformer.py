from transformers_scratch import *
import os 
#import blue score
import numpy as np
import torch
import evaluate
hf_scb = evaluate.load("sacrebleu")

test_path_En = "ted-talks-corpus/test.en"
test_path_Fr = "ted-talks-corpus/test.fr"
train_path_En = "ted-talks-corpus/train.en"
train_path_Fr = "ted-talks-corpus/train.fr"

w2i_en = np.load('w2i_en1.npy', allow_pickle=True).item()
i2w_en = np.load('i2w_en1.npy', allow_pickle=True)
w2i_fr = np.load('w2i_fr1.npy', allow_pickle=True).item()
i2w_fr = np.load('i2w_fr1.npy', allow_pickle=True)
train_tokens_en = TextDataset.tokenization(train_path_En)
train_tokens_fr = TextDataset.tokenization(train_path_Fr)
test_tokens_en = TextDataset.tokenization(test_path_En)
test_tokens_fr = TextDataset.tokenization(test_path_Fr)

print("test EN: ", len(test_tokens_en))
print("test FR: ", len(test_tokens_fr))

embeddding_matrix_en = TextDataset.extract_embeddings('/ssd_scratch/cvit/aparna/glove.42B.300d.txt', w2i_en, 300)
embeddding_matrix_fr =  TextDataset.extract_embeddings('/ssd_scratch/cvit/aparna/glove.42B.300d.txt', w2i_fr, 300)


train_indexes_en = TextDataset.tokens_to_index(train_tokens_en,w2i_en)
test_indexes_en = TextDataset.tokens_to_index(test_tokens_en,w2i_en)
train_indexes_fr = TextDataset.tokens_to_index(train_tokens_fr,w2i_fr)
test_indexes_fr = TextDataset.tokens_to_index(test_tokens_fr,w2i_fr)

test_dataset = TextDataset(test_indexes_en, test_indexes_fr, 30)
train_dataset = TextDataset(train_indexes_en, train_indexes_fr, 30)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
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
 
    new_output_word = []
    new_target_word = []
    for i in range(len(output_word)):
        if "<pad>" in target_word[i]:
            print(target_word[i].index("<pad>"))
            new_output_word.append(output_word[i][:target_word[i].index("<pad>")+1])
            new_target_word.append(target_word[i][:target_word[i].index("<pad>")])
            #breakpoint()
    
    new_output_word = [" ".join(x) for x in new_output_word]
    new_target_word = [" ".join(x) for x in new_target_word]
    print("output_sentence = ",new_output_word)
    print("target_sentence = ", new_target_word)
    if len(new_output_word) == 0:
        return 1,new_output_word,new_target_word
    ans = hf_scb.compute(predictions=new_output_word, references=[new_target_word])["score"]
    return ans,new_output_word,new_target_word


def test():
    with torch.no_grad():
        bleu_score = 0
        for i, data in enumerate(test_loader):
            src = data["x"].view(1,-1)
            tgt = data["y"].view(1,-1)
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt[:,:-1])
            output = output.argmax(dim=-1)
            blue,new_output_word,new_target_word =bleu_index(output, tgt[:,1:])
            with open("output_test.txt", "a+") as f:
                #remove padding
                if len(new_output_word) == 0:
                    continue
                output = new_output_word[0]
                f.write("tgt = " + output + "  "+"\n")
            
            with open("output_test_blue.txt", "a+") as f:
                f.write( str(blue) + "\n")
                
            bleu_score += blue
           

            print("batch = ", i)
            print("bleu_score = ", bleu_score/(i+1))

        print("bleu_score = ", bleu_score/len(test_loader))

test()

def train():
    with torch.no_grad():
        bleu_score = 0
        for i, data in enumerate(train_loader):
            src = data["x"].view(1,-1)
            tgt = data["y"].view(1,-1)
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt[:,:-1])
            output = output.argmax(dim=-1)
            blue,new_output_word,new_target_word =bleu_index(output, tgt[:,1:])
            bleu_score += blue
            with open("output_train.txt", "a+") as f:
                #remove padding
                if len(new_output_word) == 0:
                    continue
                output = new_output_word[0]
                f.write("tgt = " + output + "  "+"\n")

            with open("output_train_blue.txt", "a+") as f:
                f.write( str(blue) + "\n")
            print("batch = ", i)
            print("bleu_score = ", bleu_score/(i+1))

        print("bleu_score = ", bleu_score/len(test_loader))

train()