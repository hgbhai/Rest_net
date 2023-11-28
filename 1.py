# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5)
        self.pool5 = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
    

# %%
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx, start_idx, end_idx, teacher_forcing_ratio=0.5):
        super(DecoderLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_vector, target_sequence, hidden=None):
        # Initialize the input with the start token for the first time step
        input_seq = torch.zeros((128,1),dtype=int)
        input_seq = input_seq.to(device)
        outputs = []
        predicted = torch.zeros((128,1),dtype=int)
        for t in range(target_sequence.size(1)):
            
            teacher_force = True if torch.rand(1).item() < self.teacher_forcing_ratio else False

            if teacher_force and t > 0:
                input_seq = self.embedding(target_sequence[:, t]).unsqueeze(1)
            else:
                input_seq = self.embedding(input_seq)

            # Concatenate context vector with the input sequence
            input_seq = torch.cat((context_vector.unsqueeze(1), input_seq),dim=2)

            # LSTM forward pass
            output, hidden = self.lstm(input_seq, hidden)
            
            output = self.out(output)
            
            # criterion= nn.CrossEntropyLoss()
            # loss = criterion(output.view(-1, self.vocab_size), target_sequence)
            
            # output = F.softmax(output, dim=2)
            
            outputs.append(output)

            _, predicted = output.max(2)
            input_seq = predicted

        return torch.cat(outputs, dim=1)
    

# %%
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image

class LatexDataset(Dataset):
    def __init__(self, csv_file, root_dir, pad, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.pad = pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        latex_formula = self.pad[idx] #self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, latex_formula


# %%
def build_vocab(formulas):
    
    token2idx={'<start>':0, '<end>':1, '<PAD>': 2 }
    idx2token={'<start>':0, '<end>':1, '<PAD>': 2 }
    idx= 3
    for line in formulas:
        tokens= line.rstrip('\n').strip(' ').split()
        for token in tokens:
            if not token in token2idx:
                token2idx[token]=idx
                idx2token[idx]=token
                idx+=1
        
    return token2idx, idx2token

# %%
from torch.utils.data import DataLoader
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
syn_img= r'/home/harsh_s/scratch/exp/converting-handwritten-equations-to-latex-code/col_774_A4_2023/SyntheticData/images'
syn_csv= r'/home/harsh_s/scratch/exp/converting-handwritten-equations-to-latex-code/col_774_A4_2023/SyntheticData/train.csv'


# %%
data_form= pd.read_csv(syn_csv)
formulas= data_form['formula'].tolist()
vocab1, vocab2 = build_vocab(formulas)
# print(vocab1)

# %%
from torch.nn.utils.rnn import pad_sequence

indexed_sequences = [[vocab1[token] for token in sequence.rstrip('\n').strip(' ').split()] for sequence in formulas]
padded_sequences = pad_sequence([torch.tensor(sequence) for sequence in indexed_sequences], batch_first=True, padding_value=vocab1['<PAD>'])
# print(padded_sequences.shape)

dataset_formed= LatexDataset(csv_file=syn_csv, root_dir=syn_img, pad=padded_sequences, transform=transform)


# %%
batch_size = 128  # Adjust as needed
train_loader = DataLoader(dataset_formed, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# %%
import torch.optim as optim
embedding_dim = 512
hidden_dim = 512
learning_rate = 0.001
epochs = 10
teacher_forcing_ratio = 0.5
vocab_size= len(vocab1)
pad_idx=2
start_idx=0
end_idx=1
criterion = nn.CrossEntropyLoss()
encoder = Encoder()
decoder = DecoderLSTM(vocab_size, embedding_dim, hidden_dim, pad_idx, start_idx, end_idx, teacher_forcing_ratio)

optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
encoder = encoder.to(device)
decoder = decoder.to(device)
# %%

def train(model1, model2, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model1.to(device)
    model2.to(device)
    model1.train()
    model2.to(device)

    print(device)
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for input_sequence, target_sequence in tqdm(train_loader):
            input_sequence, target_sequence = input_sequence.to(device), target_sequence.to(device)
            optimizer.zero_grad()
            output = model1(input_sequence)
            output = model2(output, target_sequence)    
            # output = output[1:].view(-1, len(word_vocab))
            # target_sequence = target_sequence[1:].view(-1)
            loss = criterion(output.flatten(0,1), target_sequence.flatten(0,1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}')

        # Validation
        # model1.eval()
        # model2.eval()
        # with torch.no_grad():
        #     total_val_loss = 0
        #     for val_input_sequence, val_target_sequence in val_loader:
        #         val_input_sequence, val_target_sequence = val_input_sequence.to(device), val_target_sequence.to(device)
        #         output = model1(val_input_sequence)
        #         val_output = model2(output, target_sequence)
        #         # val_output = val_output[1:].view(-1, len(word_vocab))
        #         # val_target_sequence = val_target_sequence[1:].view(-1)
        #         val_loss = criterion(val_output, val_target_sequence)
        #         total_val_loss += val_loss.item()

        #     average_val_loss = total_val_loss / len(val_loader)
        #     print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss}')

    print('Training finished!')
    return model1, model2

a1, b1= train(encoder,decoder, train_loader, train_loader, criterion, optimizer)



