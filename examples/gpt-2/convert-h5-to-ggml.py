# Convert GPT-2 h5 transformer model to ggml format
#
# Load the model using GPT2Model.
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import sys
import struct
import json
import numpy as np
import re
from pathlib import Path

from transformers import GPT2LMHeadModel, AutoTokenizer

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

if len(sys.argv) < 2:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model-f16.bin"

encoder = {}
encoder_added = {}

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 2 and int(sys.argv[2]) != 1:
    use_f16 = False
    fname_out = sys.argv[1] + "/ggml-model-f32.bin"

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["n_positions"]))
fout.write(struct.pack("i", hparams["n_embd"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
#fout.write(struct.pack("i", hparams["rotary_dim"]))
fout.write(struct.pack("i", use_f16))

# TODO: temporary hack to not deal with implementing the tokenizer

if Path(dir_model + "/vocab.json").is_file():

    with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
        encoder = json.load(f)

    if Path(dir_model + "/added_tokens.json").is_file():
        with open(dir_model + "/added_tokens.json", "r", encoding="utf-8") as f:
            encoder_added = json.load(f)

    fout.write(struct.pack("i", len(encoder) + len(encoder_added)))
    for key in encoder:
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    for key in encoder_added:
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

else:
    
    tokenizer = AutoTokenizer.from_pretrained(dir_model,use_fast=False )
    tk_vocab = tokenizer.get_vocab()
    vocab_size = len(tk_vocab)
    reverse_vocab = {tk_vocab[key]: key for key in tk_vocab.keys()}

    fout.write(struct.pack("i", vocab_size))
    for i in range(vocab_size):
        text: bytes
        if tokenizer.sp_model.is_unknown(i):
            text = " \u2047 ".encode("utf-8")
        elif tokenizer.sp_model.is_control(i):
            text = tokenizer.decode([i]).encode("utf-8")
        elif tokenizer.sp_model.is_byte(i):
            piece = tokenizer.sp_model.id_to_piece(i)
            if len(piece) != 6:
                raise Exception(f"Invalid token: {piece}")
            byte_value = int(piece[3:-1], 16)
            text = struct.pack("B", byte_value)
        else:
            text = tokenizer.sp_model.id_to_piece(i).replace("\u2581", " ").encode("utf-8")

        fout.write(struct.pack("i", len(text)))
        fout.write(text)


# MODEL
model = GPT2LMHeadModel.from_pretrained(dir_model, low_cpu_mem_usage=True)
#print (model)

list_vars = model.state_dict()
#print (list_vars)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith("attn.masked_bias") \
        or name.endswith(".attn.bias") :
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if use_f16:
        if name[-7:] == ".weight" and n_dims == 2 and not name.endswith("wpe.weight"):
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # for efficiency - transpose these matrices:
    if name.endswith(".mlp.c_proj.weight") \
    or re.match(r".*h\.\d+\.attn\.c_attn\.weight", name) \
    or re.match(r".*h.\d+.mlp.c_fc.weight", name) \
    or re.match(r".*h\.\d+\.attn\.c_proj\.weight", name):
        print("  Transposing")
        data = data.transpose()

    # rename headers to keep compatibility
    if name.endswith("ln_f.weight"):
        name = "model/ln_f/g"
    elif name.endswith("ln_f.bias"):
        name = "model/ln_f/b"
    elif name.endswith("wte.weight"):
        name = "model/wte"
    elif name.endswith("wpe.weight"):
        name = "model/wpe"
    elif name.endswith("lm_head.weight"):
        name = "model/lm_head"
    elif re.match(r".*h\.\d+\.ln_1\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/g"
    elif re.match(r".*h\.\d+\.ln_1\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/b"
    elif re.match(r".*h\.\d+\.attn\.c_attn\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/w"
    elif re.match(r".*h\.\d+\.attn\.c_attn\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/b"
    elif re.match(r".*h\.\d+\.attn\.c_proj\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/w"
    elif re.match(r".*h.\d+.attn.c_proj.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/b"
    elif re.match(r".*h.\d+.ln_2.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/g"
    elif re.match(r".*h.\d+.ln_2.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/b"
    elif re.match(r".*h.\d+.mlp.c_fc.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/w"
    elif re.match(r".*h.\d+.mlp.c_fc.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/b"
    elif re.match(r".*h.\d+.mlp.c_proj.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/w"
    elif re.match(r".*h.\d+.mlp.c_proj.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/b"
    else:
        print("Unrecognized variable name. %s", name)

    str = name.encode('utf-8')

    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
