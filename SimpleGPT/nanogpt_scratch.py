import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = list(sorted(set(text)))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda c: "".join([itos[i] for i in c])

split_ratio = 0.9
batch_size = 64
emb_size = 384
seq_len = 256
trans_layers = 6
num_head = 6
max_iters = 1000
eval_interval = 100
eval_iters = 200
vocab_size = len(chars)
learning_rate = 3e-4
device = "cuda"

train_data = text[:int(split_ratio*len(text))]
test_data = text[int(split_ratio*len(text)):]



def get_batch(batch_size, seq_len, split):
    X = []
    Y = []
    init_pos = torch.randint(0, len(train_data)-seq_len, size=[batch_size, ]) if split== "train" else torch.randint(0, len(test_data)-seq_len, size=[batch_size, ])
    for pos in init_pos:
        X.append(encode(train_data[pos:pos+seq_len])) if split == "train" else X.append(encode(test_data[pos:pos+seq_len])) 
        Y.append(encode(train_data[pos+1:pos+seq_len+1])) if split == "train" else Y.append(encode(test_data[pos+1:pos+seq_len+1])) 
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    return X, Y

X, Y = get_batch(batch_size, seq_len, "train")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(batch_size, seq_len, split)
            logits, loss = model(X, Y)
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
        tril_rec = torch.tril(torch.ones(S, S)).to(device)
        out = out.masked_fill(tril_rec == 0, float('-inf'))
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

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(seq_len, emb_size).to(device)
        self.transformer_encoder_seq = nn.Sequential(*[TransformerBlock() for _ in range(trans_layers)])
        self.final_proj = nn.Linear(emb_size, vocab_size)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, input, target):
    
        B, S = input.shape
        t_emb = self.token_emb(input)
        p_emb = self.pos_emb(torch.arange(S, device=device)) 
        emb = t_emb + p_emb
        
        out = self.transformer_encoder_seq(emb)
        out = self.layer_norm(out)
        logits = self.final_proj(out)  # batch_size x seq length x vocab_size
        
        loss = F.cross_entropy(logits.view(B*S, vocab_size), target.view(B*S))

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -batch_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = GPT()
model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(batch_size, seq_len, 'train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))