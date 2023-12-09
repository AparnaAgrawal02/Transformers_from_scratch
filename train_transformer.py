from transformers_scratch import *
import os 
import numpy as np
import torch
from torchmetrics.text import BLEUScore 

wandb.init(project="Transformers_without_glove")
# save test and validation set to file 
train_path_En = "ted-talks-corpus/train.en"
train_path_Fr = "ted-talks-corpus/train.fr"
val_path_En = "ted-talks-corpus/dev.en"
val_path_Fr = "ted-talks-corpus/dev.fr"

train_tokens_en = TextDataset.tokenization(train_path_En)
train_tokens_fr = TextDataset.tokenization(train_path_Fr)
val_tokens_en = TextDataset.tokenization(val_path_En)
val_tokens_fr = TextDataset.tokenization(val_path_Fr)

print(train_tokens_en)
print(train_tokens_fr)
file = open("/ssd_scratch/cvit/aparna/glove.42B.300d.txt", "r")
glove_vocab = set()
for line in file:
    line = line.strip()
    line = line.split()
    glove_vocab.add(line[0])
file.close()
if "w2i_en.npy" not in os.listdir():

    w2i_en, i2w_en, wc_en = TextDataset.vocab(train_tokens_en, glove_vocab)
    w2i_fr, i2w_fr, wc_fr = TextDataset.vocab(train_tokens_fr, glove_vocab)
    print("wc_en = ", wc_en)
    print("wc_fr = ", wc_fr)
    np.save('w2i_en.npy', w2i_en)
    np.save('i2w_en.npy', i2w_en)
    np.save('w2i_fr.npy', w2i_fr)
    np.save('i2w_fr.npy', i2w_fr)

else:
    w2i_en = np.load('w2i_en.npy', allow_pickle=True).item()
    i2w_en = np.load('i2w_en.npy', allow_pickle=True)
    w2i_fr = np.load('w2i_fr.npy', allow_pickle=True).item()
    i2w_fr = np.load('i2w_fr.npy', allow_pickle=True)
    print(w2i_en)
    print(w2i_fr)

if "w2i_en1.npy" not in os.listdir():

    w2i_en1, i2w_en1, wc_en1 = TextDataset.vocab_without_glove(train_tokens_en, glove_vocab)
    w2i_fr1, i2w_fr1, wc_fr1 = TextDataset.vocab_without_glove(train_tokens_fr, glove_vocab)
    print("wc_en1 = ", wc_en1)
    print("wc_fr1 = ", wc_fr1)
    np.save('w2i_en1.npy', w2i_en1)
    np.save('i2w_en1.npy', i2w_en1)
    np.save('w2i_fr1.npy', w2i_fr1)
    np.save('i2w_fr1.npy', i2w_fr1)

else:
    w2i_en1 = np.load('w2i_en1.npy', allow_pickle=True).item()
    i2w_en1 = np.load('i2w_en1.npy', allow_pickle=True)
    w2i_fr1 = np.load('w2i_fr1.npy', allow_pickle=True).item()
    i2w_fr1 = np.load('i2w_fr1.npy', allow_pickle=True)
    print(w2i_en1)
    print(w2i_fr1)
embeddding_matrix_en = TextDataset.extract_embeddings('/ssd_scratch/cvit/aparna/glove.42B.300d.txt', w2i_en1, 300)
embeddding_matrix_fr =  TextDataset.extract_embeddings('/ssd_scratch/cvit/aparna/glove.42B.300d.txt', w2i_fr1, 300)

print("train EN: ", len(train_tokens_en))
print("val EN: ", len(val_tokens_en))

print("train FR: ", len(train_tokens_fr))
print("val FR: ", len(val_tokens_fr))

# train_indexes_en = TextDataset.tokens_to_index(train_tokens_en,w2i_en)
# val_indexes_en = TextDataset.tokens_to_index(val_tokens_en,w2i_en)

# train_indexes_fr = TextDataset.tokens_to_index(train_tokens_fr,w2i_fr)
# val_indexes_fr = TextDataset.tokens_to_index(val_tokens_fr,w2i_fr)

# train_dataset = TextDataset(train_indexes_en, train_indexes_fr, 30)
# val_dataset = TextDataset(val_indexes_en, val_indexes_fr, 30)

train_indexes_en = TextDataset.tokens_to_index(train_tokens_en,w2i_en1)
val_indexes_en = TextDataset.tokens_to_index(val_tokens_en,w2i_en1)

train_indexes_fr = TextDataset.tokens_to_index(train_tokens_fr,w2i_fr1)
val_indexes_fr = TextDataset.tokens_to_index(val_tokens_fr,w2i_fr1)

train_dataset = TextDataset(train_indexes_en, train_indexes_fr, 30)
val_dataset = TextDataset(val_indexes_en, val_indexes_fr, 30)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

# src_vocab_size = len(w2i_en)
# tgt_vocab_size = len(w2i_fr)
src_vocab_size = len(w2i_en1)
tgt_vocab_size = len(w2i_fr1)

print("src_vocab_size = ", src_vocab_size)
print("tgt_vocab_size = ", tgt_vocab_size)
#src_vocab_size, trg_vocab_size, embedding_dim, d_model_dim, heads, num_layers, embedding_matrix_src, embedding_matrix_trg, dropout=0.1
model = Transformer(src_vocab_size, tgt_vocab_size, 300,512,4,4,embeddding_matrix_en, embeddding_matrix_fr, 0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)
wandb.watch(model)
wandb.config.update({"src_vocab_size": src_vocab_size, "tgt_vocab_size": tgt_vocab_size, "embedding_dim": 300, "d_model_dim": 256, "heads": 4, "num_layers": 4, "dropout": 0.1})

criterion = nn.CrossEntropyLoss(ignore_index=w2i_fr1["<pad>"])
wandb.config.model = model
wandb.config.criterion = criterion
def bleu_index(output, target):
    # convet to word
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    output_word = []
    target_word = []
    for i in range(output.shape[0]):
        output_word.append([i2w_fr1[x] for x in output])
        target_word.append([i2w_fr1[x] for x in target])
    # print("output_word = ", output_word)
    # print("target_word = ", target_word)
    output_sentence = [" ".join(x) for x in output_word]
    target_sentence = [" ".join(x) for x in target_word]
    bleu_score = BLEUScore()
    bleu_score.update(output_sentence,[target_sentence])
    return bleu_score.compute()

def train(model, iterator, optimizer, criterion):
    model = model.to(device)
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_perplexity = 0
    epoch_blue = 0
    model.train()
    for batch in iterator:
        if batch["x"].shape[0] != 32:
            continue
        optimizer.zero_grad()
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        #train transformer to convert from en to fr
        #use lr scheduler
        #x is en, y is fr
        output = model(x, y[:, :-1])
        #print(output.shape)
        #decode x ,y and output to words
        x_decoded_0 = [i2w_en1[x] for x in x[0]]
        y_decoded_0 = [i2w_fr1[x] for x in y[0]]
        print(output.shape)
        print(output.argmax(dim=-1).shape)
        #breakpoint()
        output_decoded_0 = [i2w_fr1[x] for x in output.argmax(dim=-1)[0]]
        print("x_decoded_0 = ", x_decoded_0)
        print("y_decoded_0 = ", y_decoded_0)
        print("output_decoded_0 = ", output_decoded_0)
        #breakpoint()
        output = output.view(-1, output.shape[-1])
        y = y[:, 1:]
        y = y.contiguous().view(-1)
        # print(output.shape)
        # print(y.shape)
        #print(output.argmax(dim=-1).shape)
        #blue = bleu_index(output.argmax(dim=-1), y)
        blue =0 
        loss = criterion(output, y)
        perplexity = torch.exp(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_perplexity += perplexity.item()
        epoch_blue += blue
        #epoch_acc += acc.item()
        #update on wandb
        wandb.log({"Train Loss": loss.item(), "Train Perplexity": perplexity.item(), "Train Bleu": blue})
    return epoch_loss / len(iterator), epoch_perplexity / len(iterator), epoch_blue / len(iterator)


def evaluate(model, iterator, criterion):
    model = model.to(device)
    epoch_loss = 0
    epoch_perplexity = 0
    epoch_blue = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            if batch["x"].shape[0] != 32:
                continue
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            output = model(x, y[:, :-1])
            output = output.view(-1, output.shape[-1])
            y = y[:, 1:]
            y = y.contiguous().view(-1)
            # convert tot word and compute bleu score
            #blue = bleu_index(output.argmax(dim=-1), y)
            blue =0
            loss = criterion(output, y)
            perplexity = torch.exp(loss)
            epoch_loss += loss.item()
            epoch_perplexity += perplexity.item()
            epoch_blue += blue
            #epoch_acc += acc.item()
            #update on wandb
            wandb.log({"Val Loss": loss.item(), "Val Perplexity": perplexity.item(), "Val Bleu": blue})

        
    return epoch_loss / len(iterator),  epoch_perplexity / len(iterator), epoch_blue / len(iterator)


for lr in [0.00005, 0.0001, 0.0005, 0.001]:
    model = Transformer(src_vocab_size, tgt_vocab_size, 300,512,4,4,embeddding_matrix_en, embeddding_matrix_fr, 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    N_EPOCHS = 50
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start = time.time()
        print(epoch)
        train_loss,  train_perplexity, train_blue = train(model, train_loader, optimizer, criterion)
        valid_loss,  valid_perplexity, val_blue = evaluate(model, val_loader, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'/scratch/aparna/trained_transformer_not_glove1{lr}.pt')
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        print(f'Epoch: {epoch+1:02} | Train Perplexity: {train_perplexity:.3f} | Val. Perplexity: {valid_perplexity:.3f}')
        print(f'Epoch: {epoch+1:02} | Train Bleu: {train_blue:.3f} | Val. Bleu: {val_blue:.3f}')
        wandb.log({"Train Loss_epoch": train_loss, "Train Perplexity_epoch": train_perplexity, "Val Loss_epoch": valid_loss, "Val Perplexity_epoch": valid_perplexity})
        wandb.log({"Train Bleu_epoch": train_blue, "Val Bleu_epoch": val_blue})
        wandb.log({"epoch": epoch})
        end = time.time()
        print("time: ", end-start)
    wandb.log({"lr": lr})
    wandb.log({"best_valid_loss": best_valid_loss})
