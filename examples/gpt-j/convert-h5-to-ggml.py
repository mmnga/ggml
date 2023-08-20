# Convert GPT-J-6B h5 transformer model to ggml format
#
# Load the model using GPTJForCausalLM.
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
import torch
import numpy as np
from pathlib import Path
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
from sentencepiece import SentencePieceProcessor

vocab_bpe_json_name = "vocab.json"
vocab_added_tokens_json_name = "added_tokens_.json"
vocab_model_name = "spiece.model"

vocabtype = "no bpe" # read spiece.model
# vocabtype = "bpe" # read vocab.json


class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path], vocabtype: Optional[str]) -> None:
        self.vocabtype = vocabtype
        if self.vocabtype == "bpe":
          self.sentencepiece_tokenizer = json.loads(open(str(fname_tokenizer)).read())
        else:
          self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens))
        else:
            added_tokens = {}
        if self.vocabtype == "bpe":
          vocab_size: int = len(self.sentencepiece_tokenizer)
        else:
          vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            raise Exception(f"Expected added token IDs to be sequential and start at {len(added_tokens)}; got {actual_ids}")
        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base: int = vocab_size
        self.vocab_size: int = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def vocab_size(self) -> int:
        return self.sentencepiece_tokenizer.vocab_size()

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float]]:
        tokenizer = self.sentencepiece_tokenizer
        if self.vocabtype == "bpe":
          from transformers.models.gpt2 import tokenization_gpt2
          byte_encoder = tokenization_gpt2.bytes_to_unicode()
          byte_decoder = {v: k for k, v in byte_encoder.items()}
          for i, item in enumerate(tokenizer):
            text: bytes
            text = b''.join([x.to_bytes(1, byteorder='big') for x in [byte_decoder[y] for y in item]])
            score: float = -i
            yield text, score
        else:
          for i in range(tokenizer.vocab_size()):
              text: bytes
              if tokenizer.is_unknown(i):
                  text = " \u2047 ".encode("utf-8")
              elif tokenizer.is_control(i):
                  text = b""
              elif tokenizer.is_byte(i):
                  piece = tokenizer.id_to_piece(i)
                  if len(piece) != 6:
                      raise Exception(f"Invalid token: {piece}")
                  byte_value = int(piece[3:-1], 16)
                  text = struct.pack("B", byte_value)
              else:
                  text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
              score: float = tokenizer.get_score(i)
              yield text, score

    def added_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

def load_vocab(path: Path, vocabtype: Optional[str]) -> SentencePieceVocab:
    print(f"vocabtype: {vocabtype}")
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    if path.is_dir():
        vocab_file = vocab_model_name
        if vocabtype == 'bpe':
          vocab_file = vocab_bpe_json_name
        path2 = path / vocab_file
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / vocab_file
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            raise FileNotFoundError(
                f"Could not find tokenizer.model in {path} or its parent; "
                "if it's in another directory, pass the directory as --vocab-dir")
    added_tokens_path = path.parent / vocab_added_tokens_json_name
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None,
                              vocabtype)


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"


vocab = load_vocab(Path(dir_model), vocabtype)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["n_positions"]))
fout.write(struct.pack("i", hparams["n_embd"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", hparams["rotary_dim"]))
fout.write(struct.pack("i", ftype))

# TODO: temporary hack to not deal with implementing the tokenizer
fout.write(struct.pack("i", hparams["vocab_size"]))

i = 0
for text, score in vocab.all_tokens():
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    # fout.write(struct.pack("f", score))
    i+=1

for _ in range(hparams["vocab_size"]-i):
    fout.write(struct.pack("i", 0))


from transformers import GPTJForCausalLM
model = GPTJForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
#print (model)

list_vars = model.state_dict()
#print (list_vars)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0;
    if ftype != 0:
        if name[-7:] == ".weight" and n_dims == 2:
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
    # (note - with latest ggml this is no longer more efficient, so disabling it)
    #  "transformer.h.*.mlp.fc_in.weight"
    #  "transformer.h.*.attn.out_proj.weight"
    #  "transformer.h.*.attn.q_proj.weight"
    #  "transformer.h.*.attn.k_proj.weight"
    #  "transformer.h.*.attn.v_proj.weight"
    #if name.endswith(".mlp.fc_in.weight")     or \
    #   name.endswith(".attn.out_proj.weight") or \
    #   name.endswith(".attn.q_proj.weight")   or \
    #   name.endswith(".attn.k_proj.weight")   or \
    #   name.endswith(".attn.v_proj.weight"):
    #    print("  Transposing")
    #    data = data.transpose()

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
