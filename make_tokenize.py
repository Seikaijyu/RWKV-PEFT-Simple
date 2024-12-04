# encoding: utf-8
import json, random, sys, os
import numpy as np
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
from src.binidx import MMapIndexedDataset
import math

"""
如何使用：

python make_data.py demo.jsonl 3 4096

这将会：
==> 洗牌并复制 demo.jsonl (为了3个周期，适合微调) 
==> 加载 jsonl 并进行标记化
==> 保存为 demo.bin 和 demo.idx
==> 计算 ctxlen 4096 的 "magic_prime"

示例：

假设你的源 jsonl 是：
{"text":"aa"}
{"text":"bb"}
{"text":"cc"}
{"text":"dd"}

最终的 binidx 将会像这样 (这里的 "/" 表示 end_of_doc，实际上是 token [0])：
bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/

其中数据重复了3次 (每次都有不同的洗牌)
"""
# 获取2的次幂
def next_power_of_two(n):
    if n < 1:
        return 1
    power = 1
    while power < n:
        power <<= 1
    # 检查是否已经是2的次幂
    if power == n:
        return n
    return power

def find_factors_range(n, range_):
    def find_factors(n):
        factors = [i for i in range(2, math.isqrt(n) + 1) if n % i == 0]
        factors += [n // i for i in factors if n // i != i]
        return sorted(factors)[:20] if factors else [n]

    return {i: find_factors(i) for i in range(n - range_, n + range_ + 1)}


# 输出字典
def pretty_print_dict_factors(d):
    for key, value in d.items():
        value_array = np.array(value)
        max_len = max(len("MICRO_BSZ"), len("EPOCH_STEPS"))
        print(f"\n{key}")
        print(f"{'MICRO_BSZ'.ljust(max_len)} = {value}")
        print(f"{'EPOCH_STEPS'.ljust(max_len)} = {list(key // value_array)}")
if len(sys.argv) <= 2:
    N_EPOCH = 1
else:
    N_EPOCH = int(sys.argv[2].strip())
IN_FILE = sys.argv[1].strip()
OUT_PATH = os.path.dirname(IN_FILE)
OUT_NAME = os.path.splitext(os.path.basename(IN_FILE))[0]
CTX_LEN = 0
EOF = True
try:
    CTX_LEN = int(sys.argv[3].strip())
except:
    if len(sys.argv) > 3 and sys.argv[3].strip() == "--no-eof":
        EOF = False
else:
    if len(sys.argv) > 4 and sys.argv[4].strip() == "--no-eof":
        EOF = False
with open(IN_FILE, "r", encoding="utf-8") as file:
    non_empty_lines = []
    count = 0
    for line in file:
        count += 1
        stripped_line = line.strip()
        if stripped_line:
            try:
                data_text = json.loads(stripped_line)["text"]
                non_empty_lines.append(data_text)
            except:
                print(f"Error in line {count}: {stripped_line}")
                print(f"Please check your jsonl file. (Failed at line {count})")
                sys.exit(0)
            

# 在内存中重复并洗牌，避免重复写入和读取文件
shuffled_lines = []
for _ in range(N_EPOCH):
    random.shuffle(non_empty_lines)
    shuffled_lines.extend(non_empty_lines)
# def is_prime(n):
#         if n <= 1:
#             return False
#         if n <= 3:
#             return True
#         if n % 2 == 0 or n % 3 == 0:
#             return False
#         i = 5
#         while i * i <= n:
#             if n % i == 0 or n % (i + 2) == 0:
#                 return False
#             i += 6
#         return True
class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    def add_item(self, np_array):
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
    def end_document(self):
        self._doc_idx.append(len(self._sizes))
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
    

builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")
cnt = 0
max_size = 0
data_length = 0
for raw in shuffled_lines:
    out = tokenizer.encode(raw)
    if tokenizer.decode(out) != raw:
        print("ERROR" * 100)
        sys.exit(0)
    if EOF:
        out.append(0)  # [0] = end_of_doc for rwkv tokenizer
    if CTX_LEN > 0 and len(out) > CTX_LEN:
        continue
    data_length += 1
    if len(out) > max_size:
        max_size = len(out)
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if cnt % 500 == 0:
        print(cnt, end=" ", flush=True)
    cnt += 1

if data_length <= 0:
    if CTX_LEN > 0:
        print("Error: The data has been filtered to 0 entries. Please adjust the filtered context length.")
    else:
        print("Error: No data has been provided. Please add data.")
    sys.exit(0)

builder.finalize(f"{OUT_NAME}.idx")
print("done")

print("### Verifying result...")
data = MMapIndexedDataset(OUT_NAME)
data_len = len(data)
data_size = len(data._bin_buffer) // data._index._dtype_size
TODO = [0, data_len - 1]
PREVIEW_LIMIT = 100
for idx in TODO :
    ptr, size = data._index[idx]
    dix = data.get(idx=idx, offset=0, length=size).astype(int)
    print("-" * 70 + f"[{OUT_NAME} idx {idx} sz {size}]")
    if EOF:
        assert dix[-1] == 0
        dix = dix[:-1]
    if len(dix) > PREVIEW_LIMIT:
        try:
            print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
        except:
            try:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 1]))
            except:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 2]))
        print("· " * 30)
        try:  # avoid utf-8 bug
            print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
        except:
            try:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1 :]))
            except:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 2 :]))
    else:
        print(tokenizer.decode(dix))

print(f"{'-'*80}\n### Final {OUT_NAME}.bin/idx has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")

# if CTX_LEN > 0 and data_size >= CTX_LEN * 3:
#     n_chunk = int(data_size // CTX_LEN) - 1
#     for i in range(n_chunk, 0, -1):
#         if i % 3 == 2:
#             if is_prime(i):
#                 print(f"\n### magic_prime = {i} (for ctxlen {CTX_LEN})\n")
#                 break
print(f"### append_eof = {EOF}")
print(f"### max_length = {max_size}")
print(f"### max_length_power_of_two = {next_power_of_two(max_size)}")
print(f"### data_line_count = {data_length}")
# 附近5个数字的前十个个因子
print(f"### The first ten factors of the five numbers nearby (±10):")
try:
    pretty_print_dict_factors(find_factors_range(data_length, 10))
except:
    pass