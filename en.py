# %%
import torch
import torch.nn as nn

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
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTMCell(embedding_dim + 512, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h_c):
        embed = self.embedding(x)
        input_lstm = torch.cat((embed, h_c[0]), dim=1)
        h_c = self.lstm(input_lstm, h_c)
        output = self.output_layer(h_c[0])
        return output, h_c

# %%
import math
from nltk import word_tokenize
from collections import Counter
from nltk.util import ngrams

class BLEU(object):
    @staticmethod
    def compute(candidate, references, weights):
        candidate = [c.lower() for c in candidate]
        references = [[r.lower() for r in reference] for reference in references]

        p_ns = (BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)

        bp = BLEU.brevity_penalty(candidate, references)
        return bp * math.exp(s)
    @staticmethod
    def modified_precision(candidate, references, n):
        counts = Counter(ngrams(candidate, n))

        if not counts:
            return 0

        max_counts = {}
        for reference in references:
            reference_counts = Counter(ngrams(reference, n))
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

        clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

        return sum(clipped_counts.values()) / sum(counts.values())
    
    @staticmethod
    def brevity_penalty(candidate, references):
        c = len(candidate)

        r = min(len(r) for r in references)

        if c > r:
            return 1
        else:
            return math.exp(1 - r / c)
#jhbfrvjkndgbjk
print("hello")


# %%
batch_size= 32

# %%
scorer = BLEU()
grount_truths = ["$ \sin ^ { 2 } \theta + \cos ^ { 2 } \theta = 1 $",
                "$ \sum _ { { T \geq g } } { 8 . 2 } $",
                "$ r = r ( \theta ) $"]


# the predictions must be in the same format where each symbol is followed by a space
predictions = ["$ \cos ^ { 2 } \theta + \cos ^ { 2 } \theta = 1 } } } $  ",
                "$ \sum _ { { T \leq g } } { 0 . 2 } $",
                "$ x = R ( \theta ) $"]


overall = 0
for gt, pred in zip(grount_truths, predictions):
    gt = gt.split()
    pred = pred.split()
    overall += BLEU.compute(pred,[gt], weights=[1/4, 1/4, 1/4, 1/4])

print("Macro Bleu : ", overall/len(predictions))

# %%
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Assuming you have a custom dataset class
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Assuming you have a custom dataset class
class MathExpressionDataset(Dataset):
    def __init__(self, data, start_token, end_token, transform=None):
        self.data = data  # Your dataset should contain pairs of images and corresponding LaTeX formulas
        self.start_token = start_token
        self.end_token = end_token
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, latex_formula = self.data[index]
        
        # Load the image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transformations if needed
        if self.transform:
            img = self.transform(img)
        
        # Add start and end tokens to the LaTeX formula
        latex_formula = f"{self.start_token} {latex_formula} {self.end_token}"
        
        # Return image and corresponding LaTeX formula
        return img, latex_formula

# Example transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create datasets with start and end tokens
handwritten_data=[]
synthetic_data=[]
start_token = "<start>"
end_token = "<end>"
handwritten_dataset = MathExpressionDataset(handwritten_data, start_token, end_token, transform=transform)
synthetic_dataset = MathExpressionDataset(synthetic_data, start_token, end_token, transform=transform)
batch_size= 32
# Create dataloaders
handwritten_dataloader = DataLoader(handwritten_dataset, batch_size=batch_size, shuffle=True)
synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

# Assuming you have defined Encoder and Decoder classes as mentioned before

# Define the dataset and dataloaders
# ...

# Hyperparameters

embedding_dim = 512
hidden_dim = 512
learning_rate = 0.001
epochs = 10
teacher_forcing_ratio = 0.5

# Instantiate the encoder and decoder
encoder = Encoder()
decoder = Decoder(output_vocab_size, embedding_dim, hidden_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for input_data, target_data in dataloader:
        # Forward pass
        encoder_output = encoder(input_data)
        
        # Initialize decoder hidden state and cell state
        h_c = (torch.zeros(batch_size, hidden_dim), torch.zeros(batch_size, hidden_dim))
        
        # Initialize input for the first timestep
        input_seq = torch.tensor([START_TOKEN] * batch_size)  # START_TOKEN is your token for the start of a sequence
        
        # Use teacher forcing with a certain probability
        use_teacher_forcing = np.random.random() < teacher_forcing_ratio
        
        # Training the decoder
        for t in range(target_data.size(1)):
            # Choose whether to use teacher forcing or not
            if use_teacher_forcing and t > 0:
                input_seq = target_data[:, t-1]
            
            # Embedding of the previous output
            prev_output_embedding = decoder.embedding(input_seq)
            
            # Concatenate context vector with the embedding
            lstm_input = torch.cat((prev_output_embedding, encoder_output), dim=1)
            
            # Forward pass through the decoder
            output, h_c = decoder(lstm_input, h_c)
            
            # Compute the loss
            loss = criterion(output, target_data[:, t])
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Evaluation - BLEU score calculation
# ...


