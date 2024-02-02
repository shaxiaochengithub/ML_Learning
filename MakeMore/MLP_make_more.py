import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

block_size = 4
emb_size = 15
hiddenlayer_size = 300
batch_size = 64

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]     

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = torch.randn((27, emb_size)) 

g = torch.Generator().manual_seed(2147483647)
W1 = torch.randn((block_size * emb_size, hiddenlayer_size), generator=g)
b1 = torch.randn(hiddenlayer_size, generator=g)
W2 = torch.randn((hiddenlayer_size, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

for i in range(200000):

    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = C[Xtr[ix]]
    h = torch.relu(emb.view(emb.shape[0], emb.shape[1] * emb.shape[2]) @ W1 + b1)

    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    #print(loss.item())
    for p in parameters:
        p.grad = None

    loss.backward()
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -0.1 * p.grad

ix = torch.randint(0, Xtr.shape[0], (batch_size,))
emb = C[Xtr[ix]]
h = torch.tanh(emb.view(emb.shape[0], emb.shape[1] * emb.shape[2]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr[ix])
print(loss.item())

ix = torch.randint(0, Xdev.shape[0], (batch_size,))
emb = C[Xdev[ix]]
h = torch.tanh(emb.view(emb.shape[0], emb.shape[1] * emb.shape[2]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev[ix])
print(loss.item())


for _ in range(20):
    out = []
    context = [0]*block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))