# ============================================================
# NOESIS - TEMİZ DATA PIPELINE
# ============================================================
# FORMAT: Her yerde "User: ...\nAssistant: ..." kullanılıyor
# SHUFFLE: Token değil, sequence shuffle
# DR. STRANGE: YOK
# ============================================================

import json, os, random, torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

HOME = "/home/Eren"
OUTPUT_JSONL = f"{HOME}/clean_train.jsonl"
OUTPUT_PT    = f"{HOME}/processed_data/clean_packed.pt"
TOKENIZER_DIR = f"{HOME}/custom_tokenizer_info"
SEQ_LENGTH   = 4096

os.makedirs(f"{HOME}/processed_data", exist_ok=True)

# ============================================================
# BÖLÜM 1: DATASET TOPLAMA
# ============================================================

def write_jsonl(outfile, text):
    if text and len(text.strip()) > 20:
        outfile.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")

print("=" * 60)
print("📥 AŞAMA 1: VERİ TOPLAMA")
print("=" * 60)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:

    # --- 1. OpenHermes-2.5 (~1M kaliteli instruct) ---
    print("\n⏳ OpenHermes-2.5...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    count = 0
    for ex in tqdm(ds, desc="OpenHermes"):
        convs = ex.get("conversations", [])
        if not convs: continue
        turns = []
        for t in convs:
            role = t.get("from", "")
            val  = t.get("value", "").strip()
            if not val: continue
            if role == "system":
                turns.append(f"System: {val}")
            elif role == "human":
                turns.append(f"User: {val}")
            elif role == "gpt":
                turns.append(f"Assistant: {val}")
        text = "\n\n".join(turns)
        write_jsonl(f, text)
        count += 1
    print(f"  ✅ OpenHermes: {count:,}")

    # --- 2. Alpaca Cleaned (~52K) ---
    print("\n⏳ Alpaca-cleaned...")
    ds2 = load_dataset("yahma/alpaca-cleaned", split="train")
    count = 0
    for ex in tqdm(ds2, desc="Alpaca"):
        instruction = ex.get("instruction", "").strip()
        inp         = ex.get("input", "").strip()
        output      = ex.get("output", "").strip()
        if not instruction or not output: continue
        user = f"{instruction}\n{inp}" if inp else instruction
        write_jsonl(f, f"User: {user}\n\nAssistant: {output}")
        count += 1
    print(f"  ✅ Alpaca: {count:,}")

    # --- 3. Dolly-15k ---
    print("\n⏳ Dolly-15k...")
    ds3 = load_dataset("databricks/databricks-dolly-15k", split="train")
    count = 0
    for ex in tqdm(ds3, desc="Dolly"):
        instruction = ex.get("instruction", "").strip()
        context     = ex.get("context", "").strip()
        response    = ex.get("response", "").strip()
        if not instruction or not response: continue
        user = f"{instruction}\n\nContext: {context}" if context else instruction
        write_jsonl(f, f"User: {user}\n\nAssistant: {response}")
        count += 1
    print(f"  ✅ Dolly: {count:,}")

    # --- 4. OASST1 (gerçek insan konuşmaları) ---
    print("\n⏳ OASST1...")
    ds4 = load_dataset("OpenAssistant/oasst1", split="train")
    messages = {m["message_id"]: m for m in ds4}
    count = 0
    for msg in tqdm(ds4, desc="OASST1"):
        if msg["role"] != "assistant": continue
        parent = messages.get(msg.get("parent_id", ""))
        if not parent or parent["role"] != "prompter": continue
        user_text = parent["text"].strip()
        assistant_text = msg["text"].strip()
        if not user_text or not assistant_text: continue
        write_jsonl(f, f"User: {user_text}\n\nAssistant: {assistant_text}")
        count += 1
    print(f"  ✅ OASST1: {count:,}")

    # --- 5. Wikipedia QA (100K özet) ---
    print("\n⏳ Wikipedia QA (100K)...")
    ds5 = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    count = 0
    for item in tqdm(ds5, total=100_000, desc="Wikipedia"):
        if count >= 100_000: break
        title = item.get("title", "").strip()
        text  = item.get("text", "").strip()
        if len(text) < 150: continue
        summary = " ".join(text.split()[:200])
        write_jsonl(f, f"User: Tell me about {title}.\n\nAssistant: {summary}")
        count += 1
    print(f"  ✅ Wikipedia: {count:,}")

    # --- 6. SlimOrca (yüksek kalite reasoning) ---
    print("\n⏳ SlimOrca (200K)...")
    ds6 = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds6, total=200_000, desc="SlimOrca"):
        if count >= 200_000: break
        convs = ex.get("conversations", [])
        if not convs: continue
        turns = []
        for t in convs:
            role = t.get("from", "")
            val  = t.get("value", "").strip()
            if not val: continue
            if role == "system":
                turns.append(f"System: {val}")
            elif role == "human":
                turns.append(f"User: {val}")
            elif role == "gpt":
                turns.append(f"Assistant: {val}")
        text = "\n\n".join(turns)
        if text:
            write_jsonl(f, text)
            count += 1
    print(f"  ✅ SlimOrca: {count:,}")

print(f"\n✅ JSONL kaydedildi: {OUTPUT_JSONL}")

# ============================================================
# BÖLÜM 2: TOKENİZE ET VE PACK ET
# ============================================================

print("\n" + "=" * 60)
print("⚙️  AŞAMA 2: TOKENİZASYON & PACKING")
print("=" * 60)

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
EOS = tokenizer.eos_token_id
print(f"Tokenizer vocab: {len(tokenizer)} | EOS: {EOS}")

# Her satırı ayrı sequence olarak tokenize et
sequences = []
print("⏳ Tokenizing...")
with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        try:
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            ids.append(EOS)

            # 2048'den uzunsa böl, kısaysa bir sonrakiyle birleştir (packing)
            buffer = []
            for tok_id in ids:
                buffer.append(tok_id)
                if len(buffer) == SEQ_LENGTH:
                    sequences.append(buffer[:])
                    buffer = []
            # Kalan varsa padding ile doldur
            if len(buffer) > 64:  # çok kısa olanları atla
                pad_id = tokenizer.pad_token_id
                buffer += [pad_id] * (SEQ_LENGTH - len(buffer))
                sequences.append(buffer)
        except:
            continue

print(f"📊 Toplam sequence: {len(sequences):,}")

# Sequence shuffle (token sırası bozulmaz!)
print("🔀 Sequence shuffle yapılıyor...")
random.shuffle(sequences)

# Tensor'a çevir ve kaydet
tensor = torch.tensor(sequences, dtype=torch.int64)
print(f"💾 Tensor shape: {tensor.shape}")
torch.save(tensor, OUTPUT_PT)
print(f"✅ Kaydedildi: {OUTPUT_PT}")
print(f"   Boyut: {tensor.shape[0]:,} sequence x {tensor.shape[1]} token")

