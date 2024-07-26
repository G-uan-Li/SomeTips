import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import matplotlib.pyplot as plt

batch_size = 4
context_length = 16
d_model = 64
num_blocks = 8
num_heads = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 500
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_seed = 1337
torch.manual_seed(torch_seed)

with open("sales_textbook.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokenizer
encoding = tiktoken.get_encoding("cl100k_base")
tokenizer_text = encoding.encode(text)
max_token_value = max(tokenizer_text) +1
tokenizer_text = torch.tensor(tokenizer_text, dtype=torch.long, device=device)   #    100069
# print("max token value:", max_token_value)


# 分割数据集
train_size = int(0.9 * len(tokenizer_text))
train_data = tokenizer_text[:train_size]
val_data = tokenizer_text[train_size:]


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        return self.net(x)


class Attention(nn.Module):   # 单头注意力
    def __init__(self, head_size: str):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        self.wq = nn.Linear(d_model, self.head_size, bias=False)
        self.wk = nn.Linear(d_model, self.head_size, bias=False)
        self.wv = nn.Linear(d_model, self.head_size, bias=False)
        # apply mask
        self.register_buffer("tril", torch.tril(
            torch.ones((self.context_length, self.context_length))))# 16, 16
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
         B, T, C = x.shape

         q =   self.wq(x)
         k =   self.wk(x)
         v =   self.wv(x)

         weight = (q @ k.transpose(-2, -1) ) / (math.sqrt(k.size(-1))) # 单头
         weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
         weight = F.softmax(weight, dim=-1)
         weight = self.dropout(weight)
         out = weight @ v

         return out



class  MultiHeadAttention(nn.Module):
    def __init__(self, head_size: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads =  nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # head size should be divisible by d_model
        self.num_heads = num_heads
        self.dropout = dropout


        self.multiHeadAttention = MultiHeadAttention(self.head_size)
        self.feedForward = FeedForward()
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):

        x = x + self.multiHeadAttention(self.ln1(x))  # Residual connection
        x = x + self.feedForward(self.ln2(x))  # Residual connection

        return x



class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value

        self.token_embedding_lookup_table = nn.Embedding(self.max_token_value, self.d_model)

        self.blocks = nn.Sequential(*([TransformerBlock(self.num_heads) for _ in range(num_blocks)] +
                                      [nn.LayerNorm(self.d_model)]))
        self.model_out_linear_layer= nn.Linear(self.d_model, self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        """
        # Set up position embedding look-up table
        # following the same approach as the original Transformer paper (Sine and Cosine functions)
        """
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.blocks(x)
        # The "logits" are the output values of our model before applying softmax
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            return idx




model = TransformerLanguageModel()
model = model.to(device)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    indx = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in indx])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in indx])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_loss = list()
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        tracked_loss.append(losses['val'])

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



plt.figure(figsize=(10, 5))
plt.plot(tracked_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "model.pt")

model.eval()
start = 'the man is '
strat_idx = encoding.encode(start)
x = torch.tensor(strat_idx, dtype=torch.long, device=device)[None, ...]
y = model.generate(x, max_new_tokens=100)
print('===========================================================')
print(encoding.decode(y[0].tolist()))
print('===========================================================')

