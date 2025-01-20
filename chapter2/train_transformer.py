import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer


# ----------------------------
# Data Preparation
# ----------------------------
# Synthetic dataset: English to French translation
src_sentences = [
    ['hello', 'world'],
    ['good', 'morning'],
    ['how', 'are', 'you'],
    ['good', 'night']
]

tgt_sentences = [
    ['bonjour', 'le', 'monde'],
    ['bonjour'],
    ['comment', 'ça', 'va'],
    ['bonne', 'nuit']
]

# Vocabulary creation
def build_vocab(sentences, special_tokens=['<pad>', '<unk>', '<cls>', '<sep>']):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    idx = len(vocab)
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

src_vocab = build_vocab(src_sentences, special_tokens=['<pad>', '<unk>'])
tgt_vocab = build_vocab(tgt_sentences, special_tokens=['<pad>', '<unk>', '<cls>', '<sep>'])

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# ----------------------------
# Dataset and DataLoader
# ----------------------------

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=10):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Convert tokens to indices
        # example: ['hello', 'world','sdsdfs','dfdf] -> [4,5,1,1]
        src_indices = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in src_sentence]
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in tgt_sentence]
        
        # Add <cls> and <sep> tokens to target indices
        tgt_input = [self.tgt_vocab['<cls>']] + tgt_indices
        tgt_output = tgt_indices + [self.tgt_vocab['<sep>']]
        
        # Pad sequences to max_len
        src_indices = src_indices + [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
        tgt_input = tgt_input + [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_input))
        tgt_output = tgt_output + [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_output))
        
        return torch.tensor(src_indices), torch.tensor(tgt_input), torch.tensor(tgt_output)

# Create dataset and dataloader
max_len = 5
dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# ----------------------------
# Helper Functions
# ----------------------------

def create_src_mask(src):
    # src: [batch_size, src_len]
    src_mask = (src != src_vocab['<pad>']).unsqueeze(1)
    # src_mask: [batch_size, 1, src_len]
    return src_mask

# ----------------------------
# Model Instantiation
# ----------------------------

d_model = 16
num_heads = 2
num_encoder_layers = 6
num_decoder_layers = 8
d_ff = 12
dropout = 0.1

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    d_ff=d_ff,
    dropout=dropout
)

# ----------------------------
# Loss Function and Optimizer
# ----------------------------

criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training Loop
# ----------------------------

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for src_batch, tgt_input_batch, tgt_output_batch in dataloader:
        # Create masks
        src_mask = create_src_mask(src_batch)
        # tgt_mask is created inside the model
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, enc_attn_weights, dec_attn_weights, dec_enc_attn_weights = model(
            src_batch, tgt_input_batch, src_mask)
        
        # Reshape output and target for loss computation
        output = output.view(-1, tgt_vocab_size)
        tgt_output_batch = tgt_output_batch.view(-1)
        
        # Compute loss
        loss = criterion(output, tgt_output_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ----------------------------
# Evaluation Loop
# ----------------------------

model.eval()
with torch.no_grad():
    for src_batch, tgt_input_batch, tgt_output_batch in dataloader:
        src_mask = create_src_mask(src_batch)
        output, _, _, _ = model(src_batch, tgt_input_batch, src_mask)
        # Get the predicted tokens
        pred_tokens = output.argmax(dim=-1)
        for i in range(src_batch.size(0)):
            src_tokens = [list(src_vocab.keys())[list(src_vocab.values()).index(idx.item())] for idx in src_batch[i] if idx.item() not in [src_vocab['<pad>'], src_vocab['<unk>']]]
            tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(idx.item())] for idx in tgt_output_batch[i] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<unk>'], tgt_vocab['<cls>'], tgt_vocab['<sep>']]]
            pred_sentence = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(idx.item())] for idx in pred_tokens[i] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<unk>'], tgt_vocab['<cls>'], tgt_vocab['<sep>']]]
            print("-" * 30)
            print(f"Source: {' '.join(src_tokens)}")
            print(f"Target: {' '.join(tgt_tokens)}")
            print(f"Predicted: {' '.join(pred_sentence)}")
