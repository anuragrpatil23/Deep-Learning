"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
from datasets import load_dataset
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm


# -----------------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) #100M tokens per shard

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__),local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# -----------------------------------------------------

#download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split='train', cache_dir=DATA_CACHE_DIR)

#init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # token number for |end of text| - 50256

def tokenize(doc):
    #tokenizes a single document and returns a numpy array of uint16 tokens. 
    tokens = [eot] # even though its called endoftext, it also marks the beginning of text. this special token delimits all documents. 
    tokens.extent(enc.encode(doc))
    tokens_np = np.array(tokens)
    assert (0<=tokens_np).all() and (tokens_np<2**16).all(), "token dictionary too large for uint16" #16 bit unsigned integers
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def get_filename(shard_index):
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    return filename



# tokenize all documents and write output shards, each of shard_size tokens(last shard has remainder)
nprocs = max(1, os.cpu_count()//2)

with mp.Pool(nprocs) as pool:
    shard_index = 0

    #preallocate buffer to hold current shard
    all_tokens_np = np.empty(shard_size, dtype=np.uint16)
    token_count = 0 # number of tokens in the current shard
    
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        #is there enough space in the current shard
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            filename = get_filename(shard_index)
            # split the document into whatever fits in this shard; the remainder goes into the next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:shard_size] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            #Get new shard
            shard_index += 1
            token_count = 0
            progress_bar = None

            # Process the remaining tokens if they need multiple shards
            remaining_tokens = tokens[remainder:]
            num_full_shards = len(remaining_tokens) // shard_size
            for i in range(num_full_shards):
                all_tokens_np[:shard_size] = remaining_tokens[i*shard_size:(i+1)*shard_size]
                write_datafile(filename, all_tokens_np)
                shard_index += 1

            # populate the next shard with the leftovers of the current document. 
            leftover_tokens = remaining_tokens[num_full_shards*shard_size:]
            if len(leftover_tokens) > 0:
                all_tokens_np[:len(leftover_tokens)] = leftover_tokens
                token_count = len(leftover_tokens)


    # write any remaining tokens as the last shard
    if token_count != 0:
        filename = get_filename(shard_index)
        write_datafile(filename, all_tokens_np[:token_count])
