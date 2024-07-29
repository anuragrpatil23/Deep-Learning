from dataclasses import dataclass
import math
import pathlib
import tiktoken
import time
import inspect
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024 #context window length
    vocab_size: int = 50257 #number of tokens in the vocabulary
    n_layer: int = 12 #number of transformer layers
    n_head: int = 12 #number of heads in the multi-head attention
    n_embd: int = 768 #embedding dimension.


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all the heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection (what is this for?)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query key and value for all heads in batch and move head forward to be the batch dim. 
        #nh is the number of heads, hs is "head size" and C (number of channels) = nh * hs
        # eg in GPT-2 (124M), n_head=12, hs = 64, so C = 768 channels in the transformer. 

        qkv  = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # att = (q@k.transpose(-2, -1)) * (1.0/ math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T, :T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # just ressemble everything again. this actually performs the concatination operation.

        #output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, target=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        #forward the token and position embeddings /
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape T
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        # those two lines are equivalent to this - pos_emb = self.transformer.wpe.weight[None, :T, :].view(T, -1) #shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        #forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)

        #forward the final layernorm and the classifier head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #logits of shape (B, T, vocab_size)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained model weights from huggingface.
        steps:
        1. Initialize the model state dict for our model
        2. initialize the model state dict for the same model from huggingface
        3. copy the dict values from huggingface model state dict to our model state dict. 
        4. Take care of the the idosyncracies of the model state dict. Eg - biases are not needed. some vectors from the hugging face have to be transposed to work with pytorch. 
        5. Return the model. 

        Parameters
        ----------
        model_type : str
            the model of interest.
        """

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt:", model_type)

        # n_layer, n_head and n_embd are determined by the model type. 
        config_args={
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), #124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), #345M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600) #1558M params
        }[model_type]

        config_args["vocab_size"] = 50257 #always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024 #always 1024 for GPT model checkpoints

        #create a initialization for our GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()

        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] #discard this mask/buffer key as it doesnt need to be learned. its a mask for the attention mechanism and we have already defined it as trill one tensor.

        # init a huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all of the parameters are aligned and match in names and shape. 
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] #ignore these. 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] #ignore these.
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        #basically the openai checkpoints use a "Conv1D" module but we only want to use a vanialla linear layer. 
        #this means that we have to transpose the weights when we import them. 
        #sanity check
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any (k.endswith(w) for w in transposed):
                #special treatment for the conv1d weights we need to transpose. 
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanilla copy over the other parameters. 
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):

        # start with all of the candidate parameters (that required grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        #create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. 
        #i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params':nodecay_params, 'weigh_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)  
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        #create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused=use_fused)
        return optimizer
#---------------------------------------------------------------------------------------------------------------------#

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        dir_path = pathlib.Path(__file__).parent.absolute()
        #Get the dataset
        with open(dir_path.joinpath('input.txt')) as file:
            data = file.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        #state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) #inputs
        y = buf[1:].view(B, T) #targets

        # advance the position in the tensor
        self.current_position += B*T*self.num_processes

        #if is the end of the data, loop back around
        if self.current_position + (B*T*self.num_processes + 1)> len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y


#run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#set up DDP (distributed data parallel)
#torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 #is this a ddp run?
if ddp:
    print("Using DDP")
    #use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend = 'nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    print(f"ddp_rank :{ddp_rank}")
    print(f"ddp_local_rank :{ddp_local_rank}")
    print(f"ddp_world_size :{ddp_world_size}")
    print(f"device :{device}")
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  
    print("Using device: ", device)


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 # 2^19, ~0.5M, in number of tokens
B = 16 #micro batch size
T = 1024 #sequence length
assert total_batch_size % (B*T*ddp_world_size)==0, "make sure total batch size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size//(B*T* ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"==> calculated gradient accumulation steps:{grad_accum_steps}")


# Create dataloader to get x and y
train_loader = DataLoaderLite(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size)


torch.set_float32_matmul_precision('high')


# Create the Model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model #always contains the raw unwrapped model

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 

def get_lr(it):
    #1) linear warmup for warmup_iters steps
    if it<warmup_steps:
        return max_lr * (it+1)/warmup_steps
    #2) if it>lr_decay_iters, return min learning rate
    if it>max_steps:
        return min_lr
    #3) in between, use cosine decay down to min learning rate
    decay_ratio = (it-warmup_steps)/ (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to 0
    return min_lr + coeff*(max_lr-min_lr)

#optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model.forward(x, target=y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize() #wait for cuda instructions to finish. cpu is just sending instructions to cuda at this point so python interpretator gotta wait for those tasks to finish before timing them. 

    t1 = time.time()
    dt = (t1-t0)*1000 # time diff in milliseconds. 
    token_processed = train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size
    tokens_per_sec = (token_processed)/(t1-t0)

    if master_process:
        print(f"step {step}, loss: {loss_accum.item():.6f}, norm:{norm:.4f}, lr:{lr:.4e} dt: {dt:.2f}ms, tok/sec:{tokens_per_sec:.2f}")


if ddp:
    destroy_process_group()

import sys; sys.exit(0)
#load the model
num_return_sequences = 5
max_length = 30


model.eval()

#prefix tokens


# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)
# x = tokens.to(device)

num_return_sequences = B
max_length = T
x_og = x
x = x[:,:5] #take the firs 5 tokens from all the batches. 

    
#generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)

while x.size(1) < max_length:

    #forward the model to get the logits
    with torch.no_grad():
        logits, _ = model(x) #(B, T, vocab_size)
        #take the logits at the last position
        logits = logits[:,-1, :] #(B, vocab_size)

        #get the probabilities
        probs = F.softmax(logits, dim=-1)

        #do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim =-1)

        #select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)

        #gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #(B, 1)

        #append to the sequence
        x = torch.cat((x, xcol), dim=1)


#print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
    decoded = enc.decode(x_og[i, :max_length].tolist())
    print("ORIGINAL:", decoded )

