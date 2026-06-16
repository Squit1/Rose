# ============================================================
# MATH & CODE FINE-TUNE DATA PIPELINE
# ============================================================

import json, os, random, torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

HOME         = "/home/Eren"
OUTPUT_PT    = f"{HOME}/processed_data/math_code_packed.pt"
TOKENIZER_DIR= f"{HOME}/custom_tokenizer_info"
SEQ_LENGTH   = 4096

os.makedirs(f"{HOME}/processed_data", exist_ok=True)

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
EOS = tokenizer.eos_token_id

sequences = []
buffer = []

def add_text(text):
    global buffer
    if not text or len(text.strip()) < 30:
        return
    ids = tokenizer.encode(text.strip())
    ids.append(EOS)
    buffer.extend(ids)
    while len(buffer) >= SEQ_LENGTH:
        sequences.append(buffer[:SEQ_LENGTH])
        buffer = buffer[SEQ_LENGTH:]

print("=" * 60)
print("📥 MATH & CODE DATA")
print("=" * 60)

# 1. MetaMathQA (200K)
print("\n⏳ MetaMathQA...")
ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
count = 0
for ex in tqdm(ds, total=200_000, desc="MetaMath"):
    if count >= 200_000: break
    q = ex.get("query", "").strip()
    a = ex.get("response", "").strip()
    if not q or not a: continue
    lines = a.split("\n")
    steps, final = [], a
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ["the answer is", "therefore", "thus", "= "]):
            steps = lines[:i]
            final = "\n".join(lines[i:])
            break
    if steps:
        think = "\n".join(steps)
        text = f"User: {q}\n\nAssistant: <think>\n{think}\n</think>\n{final}"
    else:
        text = f"User: {q}\n\nAssistant: {a}"
    add_text(text)
    count += 1
print(f"  ✅ MetaMathQA: {count:,}")

# 2. OpenR1-Math (50K)
print("\n⏳ OpenR1-Math...")
try:
    ds2 = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds2, total=50_000, desc="OpenR1"):
        if count >= 50_000: break
        q = ex.get("problem", "").strip()
        a = ex.get("solution", "").strip()
        ans = ex.get("answer", "").strip()
        if not q or not a: continue
        text = f"User: {q}\n\nAssistant: <think>\n{a}\n</think>\nThe answer is: {ans}"
        add_text(text)
        count += 1
    print(f"  ✅ OpenR1-Math: {count:,}")
except Exception as e:
    print(f"  ⚠️  OpenR1-Math atlandı: {e}")

# 3. CodeAlpaca (111K)
print("\n⏳ CodeAlpaca...")
ds3 = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
count = 0
for ex in tqdm(ds3, desc="CodeAlpaca"):
    instruction = ex.get("instruction", "").strip()
    output      = ex.get("output", "").strip()
    if not instruction or not output: continue
    text = f"User: {instruction}\n\nAssistant: <think>\nLet me think about this step by step.\n</think>\n{output}"
    add_text(text)
    count += 1
print(f"  ✅ CodeAlpaca: {count:,}")

# 4. CodeFeedback (50K)
print("\n⏳ CodeFeedback...")
ds4 = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
count = 0
for ex in tqdm(ds4, total=50_000, desc="CodeFeedback"):
    if count >= 50_000: break
    q = ex.get("query", "").strip()
    a = ex.get("answer", "").strip()
    if not q or not a: continue
    text = f"User: {q}\n\nAssistant: <think>\nLet me analyze this carefully.\n</think>\n{a}"
    add_text(text)
    count += 1
print(f"  ✅ CodeFeedback: {count:,}")

# Sequence shuffle
print(f"\n📊 Toplam sequence: {len(sequences):,}")
print("🔀 Shuffle...")
random.shuffle(sequences)

tensor = torch.tensor(sequences, dtype=torch.int64)
torch.save(tensor, OUTPUT_PT)
print(f"✅ Kaydedildi: {OUTPUT_PT}")
print(f"   Shape: {tensor.shape}")
