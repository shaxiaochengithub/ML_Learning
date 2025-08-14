import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = list(sorted(set(text)))


stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
stoi["["] = len(chars)+1
itos[len(chars)+1] = "["


encode = lambda s: [stoi[ch] for ch in s]
decode = lambda c: "".join([itos[i] for i in c])



split_ratio = 0.9
batch_size = 64
emb_size = 384
seq_len = 128
trans_layers = 6
num_head = 6
max_iters = 1000
eval_interval = 100
eval_iters = 200
vocab_size = len(chars)+2
learning_rate = 3e-4
device = "cuda"
mask_ratio = 0.2

train_data = text[:int(split_ratio*len(text))]
test_data = text[int(split_ratio*len(text)):]



def get_batch(batch_size, seq_len, split):
    X = []
    Y = []
    init_pos = torch.randint(0, len(train_data)-seq_len, size=[batch_size, ]) if split== "train" else torch.randint(0, len(test_data)-seq_len, size=[batch_size, ])
    for pos in init_pos:
        X.append(encode(train_data[pos:pos+seq_len])) if split == "train" else X.append(encode(test_data[pos:pos+seq_len])) 
        Y.append(encode(train_data[pos:pos+seq_len])) if split == "train" else Y.append(encode(test_data[pos:pos+seq_len])) 

    mask_num = int(seq_len * mask_ratio)
    return_pos = torch.zeros(batch_size, seq_len, dtype=bool)
    mask_positions = torch.randint(0, seq_len, size = [batch_size, mask_num])

    for i in range(batch_size):
        for pos in mask_positions[i]:
            X[i][pos] = len(chars)+1
            return_pos[i][pos] = True

    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    
    return X, return_pos, Y

X, mask_positions, Y = get_batch(batch_size, seq_len, "train")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, mask_positions, Y = get_batch(batch_size, seq_len, split)
            logits, loss = model(X, mask_positions, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(emb_size, int(emb_size/num_head), bias=False)
        self.key = nn.Linear(emb_size, int(emb_size/num_head), bias=False)
        self.value = nn.Linear(emb_size, int(emb_size/num_head), bias=False)

    def forward(self, input):
        B, S, C= input.shape # batch size; sequence length; feature number (embedding dimentionality)
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)

        out = q @ k.transpose(-2, -1) * k.shape[-1]**(-1/2)  # q: batch size x seq length x head feature size @ k: batch size x head feature size
        #tril_rec = torch.tril(torch.ones(S, S)).to(device)
        #out = out.masked_fill(tril_rec == 0, float('-inf'))   # not needed anymore for BERT
        out = F.softmax(out, dim=1) @ v

        return out


class MultiHead(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.num_head = num_head
        self.heads = nn.ModuleList([Head() for _ in range(num_head)])   
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, input):
        out = torch.cat([h(input) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head = MultiHead(num_head)
        self.ffn = nn.Sequential(nn.Linear(emb_size, 4*emb_size),
                                          nn.ReLU(),
                                          nn.Linear(4*emb_size, emb_size),
                                          )
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input):
        out = input + self.multi_head(self.ln1(input))
        out = out + self.ffn(self.ln2(out))

        return out      

class BERT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(seq_len, emb_size).to(device)
        self.transformer_encoder_seq = nn.Sequential(*[TransformerBlock() for _ in range(trans_layers)])
        self.final_proj = nn.Linear(emb_size, vocab_size)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, input, mask_positions, target):
    
        B, S = input.shape
        
        # Add bounds checking
        assert S <= seq_len, f"Sequence length {S} exceeds maximum {seq_len}"
        positions = torch.arange(S, device=input.device)  # Use same device as input
        p_emb = self.pos_emb(positions)
        t_emb = self.token_emb(input)
        
        emb = t_emb + p_emb
        
        out = self.transformer_encoder_seq(emb)
        out = self.layer_norm(out)
        logits = self.final_proj(out)  # batch_size x seq length x vocab_size
        
        loss = F.cross_entropy(logits[mask_positions], target[mask_positions])

        return logits, loss
    


model = BERT()
model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, mask_positions, yb = get_batch(batch_size, seq_len, 'train')

    # evaluate the loss
    logits, loss = model(xb, mask_positions, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
