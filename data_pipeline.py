# ============================================================
# NOESIS - TEMİZ DATA PIPELINE v2
# Genel + Reasoning + Matematik + Kodlama
# FORMAT: User: ... \n\nAssistant: ...
# REASONING: <think>...</think> bloğu ile
# ============================================================

import json, os, random, torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

HOME         = "/home/Eren"
OUTPUT_JSONL = f"{HOME}/clean_train.jsonl"
OUTPUT_PT    = f"{HOME}/processed_data/clean_packed.pt"
TOKENIZER_DIR= f"{HOME}/custom_tokenizer_info"
SEQ_LENGTH   = 4096

os.makedirs(f"{HOME}/processed_data", exist_ok=True)

def write_jsonl(outfile, text):
    if text and len(text.strip()) > 30:
        outfile.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")

def format_conversations(convs):
    turns = []
    for t in convs:
        role = t.get("from", "")
        val  = t.get("value", "").strip()
        if not val: continue
        if role == "system":
            turns.append(f"System: {val}")
        elif role in ("human", "user"):
            turns.append(f"User: {val}")
        elif role in ("gpt", "assistant"):
            turns.append(f"Assistant: {val}")
    return "\n\n".join(turns)

print("=" * 60)
print("📥 AŞAMA 1: VERİ TOPLAMA")
print("=" * 60)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:

    # 1. OpenHermes-2.5 (~1M genel instruct)
    print("\n⏳ OpenHermes-2.5...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    count = 0
    for ex in tqdm(ds, desc="OpenHermes"):
        text = format_conversations(ex.get("conversations", []))
        if text:
            write_jsonl(f, text)
            count += 1
    print(f"  ✅ OpenHermes: {count:,}")

    # 2. SlimOrca - reasoning (200K)
    print("\n⏳ SlimOrca (200K reasoning)...")
    ds2 = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds2, total=200_000, desc="SlimOrca"):
        if count >= 200_000: break
        text = format_conversations(ex.get("conversations", []))
        if text:
            write_jsonl(f, text)
            count += 1
    print(f"  ✅ SlimOrca: {count:,}")

    # 3. MetaMathQA - matematik reasoning (200K)
    print("\n⏳ MetaMathQA (200K matematik)...")
    ds3 = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds3, total=200_000, desc="MetaMath"):
        if count >= 200_000: break
        q = ex.get("query", "").strip()
        a = ex.get("response", "").strip()
        if not q or not a: continue
        # Adım adım çözümü <think> bloğuna al
        lines = a.split("\n")
        steps = []
        final = a
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ["the answer is", "therefore", "thus", "so the answer", "= "]):
                steps = lines[:i]
                final = "\n".join(lines[i:])
                break
        if steps:
            think_block = "\n".join(steps)
            text = f"User: {q}\n\nAssistant: <think>\n{think_block}\n</think>\n{final}"
        else:
            text = f"User: {q}\n\nAssistant: {a}"
        write_jsonl(f, text)
        count += 1
    print(f"  ✅ MetaMathQA: {count:,}")

    # 4. CodeAlpaca - kodlama (20K)
    print("\n⏳ CodeAlpaca (kodlama)...")
    ds4 = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    count = 0
    for ex in tqdm(ds4, desc="CodeAlpaca"):
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if not instruction or not output: continue
        text = f"User: {instruction}\n\nAssistant: <think>\nLet me think about this step by step.\n</think>\n{output}"
        write_jsonl(f, text)
        count += 1
    print(f"  ✅ CodeAlpaca: {count:,}")

    # 5. CodeFeedback - gerçek kod soruları (50K)
    print("\n⏳ CodeFeedback (50K)...")
    ds5 = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds5, total=50_000, desc="CodeFeedback"):
        if count >= 50_000: break
        q = ex.get("query", "").strip()
        a = ex.get("answer", "").strip()
        if not q or not a: continue
        text = f"User: {q}\n\nAssistant: <think>\nLet me analyze this carefully.\n</think>\n{a}"
        write_jsonl(f, text)
        count += 1
    print(f"  ✅ CodeFeedback: {count:,}")

    # 6. Alpaca Cleaned (52K genel)
    print("\n⏳ Alpaca-cleaned...")
    ds6 = load_dataset("yahma/alpaca-cleaned", split="train")
    count = 0
    for ex in tqdm(ds6, desc="Alpaca"):
        instruction = ex.get("instruction", "").strip()
        inp         = ex.get("input", "").strip()
        output      = ex.get("output", "").strip()
        if not instruction or not output: continue
        user = f"{instruction}\n{inp}" if inp else instruction
        write_jsonl(f, f"User: {user}\n\nAssistant: {output}")
        count += 1
    print(f"  ✅ Alpaca: {count:,}")

    # 7. Dolly-15k
    print("\n⏳ Dolly-15k...")
    ds7 = load_dataset("databricks/databricks-dolly-15k", split="train")
    count = 0
    for ex in tqdm(ds7, desc="Dolly"):
        instruction = ex.get("instruction", "").strip()
        context     = ex.get("context", "").strip()
        response    = ex.get("response", "").strip()
        if not instruction or not response: continue
        user = f"{instruction}\n\nContext: {context}" if context else instruction
        write_jsonl(f, f"User: {user}\n\nAssistant: {response}")
        count += 1
    print(f"  ✅ Dolly: {count:,}")

    # 8. OASST1 - gerçek insan diyaloğu
    print("\n⏳ OASST1...")
    ds8 = load_dataset("OpenAssistant/oasst1", split="train")
    messages = {m["message_id"]: m for m in ds8}
    count = 0
    for msg in tqdm(ds8, desc="OASST1"):
        if msg["role"] != "assistant": continue
        parent = messages.get(msg.get("parent_id", ""))
        if not parent or parent["role"] != "prompter": continue
        user_text = parent["text"].strip()
        asst_text = msg["text"].strip()
        if not user_text or not asst_text: continue
        write_jsonl(f, f"User: {user_text}\n\nAssistant: {asst_text}")
        count += 1
    print(f"  ✅ OASST1: {count:,}")

    # 9. Wikipedia QA (100K)
    print("\n⏳ Wikipedia QA (100K)...")
    ds9 = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    count = 0
    for item in tqdm(ds9, total=100_000, desc="Wikipedia"):
        if count >= 100_000: break
        title = item.get("title", "").strip()
        text  = item.get("text", "").strip()
        if len(text) < 150: continue
        summary = " ".join(text.split()[:200])
        write_jsonl(f, f"User: Tell me about {title}.\n\nAssistant: {summary}")
        count += 1
    print(f"  ✅ Wikipedia: {count:,}")

    # 10. OpenR1-Math - ileri matematik reasoning (50K)
    print("\n⏳ OpenR1-Math (50K)...")
    try:
        ds10 = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
        count = 0
        for ex in tqdm(ds10, total=50_000, desc="OpenR1-Math"):
            if count >= 50_000: break
            q = ex.get("problem", "").strip()
            a = ex.get("solution", "").strip()
            if not q or not a: continue
            text = f"User: {q}\n\nAssistant: <think>\n{a}\n</think>\nThe answer is: {ex.get('answer', '').strip()}"
            write_jsonl(f, text)
            count += 1
        print(f"  ✅ OpenR1-Math: {count:,}")
    except Exception as e:
        print(f"  ⚠️  OpenR1-Math atlandı: {e}")

print(f"\n✅ JSONL tamamlandı: {OUTPUT_JSONL}")

# ============================================================
# AŞAMA 2: TOKENİZASYON & PACKING
# ============================================================

print("\n" + "=" * 60)
print("⚙️  AŞAMA 2: TOKENİZASYON & PACKING")
print("=" * 60)

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
print(f"Tokenizer vocab: {len(tokenizer)} | EOS: {EOS} | PAD: {PAD}")

sequences = []
buffer = []

print("⏳ Tokenizing ve packing...")
with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        try:
            text = json.loads(line)["text"]
            ids  = tokenizer.encode(text)
            ids.append(EOS)
            buffer.extend(ids)

            # Buffer dolunca sequence çıkar
            while len(buffer) >= SEQ_LENGTH:
                sequences.append(buffer[:SEQ_LENGTH])
                buffer = buffer[SEQ_LENGTH:]
        except:
            continue

print(f"📊 Toplam sequence: {len(sequences):,}")

# Sequence shuffle (token sırası BOZULMAZ)
print("🔀 Sequence shuffle...")
random.shuffle(sequences)

tensor = torch.tensor(sequences, dtype=torch.int64)
print(f"💾 Tensor shape: {tensor.shape}")
torch.save(tensor, OUTPUT_PT)

print(f"\n✅ Pipeline tamamlandı!")
print(f"   JSONL : {OUTPUT_JSONL}")
print(f"   Tensor: {OUTPUT_PT}")
print(f"   Shape : {tensor.shape[0]:,} x {tensor.shape[1]}")
