# ============================================================
# NOESIS - KAPSAMLI MATEMATİK DATA PIPELINE v2
# Kolay'dan olimpiyat seviyesine kadar tüm matematik
# FORMAT: <think>adım adım</think> + final cevap
# ============================================================

import json, os, random, torch, re
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

HOME          = "/home/Eren"
OUTPUT_PT     = f"{HOME}/processed_data/math_v2_packed.pt"
TOKENIZER_DIR = f"{HOME}/custom_tokenizer_info"
SEQ_LENGTH    = 2048

os.makedirs(f"{HOME}/processed_data", exist_ok=True)

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
EOS = tokenizer.eos_token_id

sequences = []
buffer    = []
total_added = 0

def add_text(text):
    global buffer, total_added
    if not text or len(text.strip()) < 30:
        return
    ids = tokenizer.encode(text.strip())
    ids.append(EOS)
    buffer.extend(ids)
    while len(buffer) >= SEQ_LENGTH:
        sequences.append(buffer[:SEQ_LENGTH])
        buffer = buffer[SEQ_LENGTH:]
    total_added += 1

def extract_think_and_answer(solution, answer=""):
    """Çözümü <think> bloğu + final cevap olarak formatla"""
    solution = solution.strip()
    # Final cevabı bul
    final = answer.strip() if answer else ""
    if not final:
        # Çözümden son satırı al
        lines = [l.strip() for l in solution.split("\n") if l.strip()]
        final = lines[-1] if lines else ""
    return f"<think>\n{solution}\n</think>\n{final}"

print("=" * 60)
print("📥 KAPSAMlI MATEMATİK DATA PIPELINE v2")
print("=" * 60)

# ── 1. GSM8K ──────────────────────────────────────────────
print("\n⏳ GSM8K (ilkokul-ortaokul matematik)...")
ds = load_dataset("openai/gsm8k", "main", split="train")
count = 0
for ex in tqdm(ds, desc="GSM8K"):
    q   = ex.get("question", "").strip()
    ans = ex.get("answer", "").strip()
    if not q or not ans: continue
    # GSM8K cevabı #### ile bitiyor
    parts = ans.split("####")
    steps = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else ans
    text = f"User: {q}\n\nAssistant: {extract_think_and_answer(steps, final)}"
    add_text(text)
    count += 1
print(f"  ✅ GSM8K: {count:,}")

# ── 2. MetaMathQA ─────────────────────────────────────────
print("\n⏳ MetaMathQA (200K)...")
ds2 = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
count = 0
for ex in tqdm(ds2, total=200_000, desc="MetaMath"):
    if count >= 200_000: break
    q = ex.get("query", "").strip()
    a = ex.get("response", "").strip()
    if not q or not a: continue
    # Final cevabı bul
    final = ""
    for kw in ["The answer is", "the answer is", "= ", "\\boxed{"]:
        if kw in a:
            idx = a.rfind(kw)
            final = a[idx:].split("\n")[0].strip()
            break
    think = a if not final else a[:a.rfind(final)].strip()
    text = f"User: {q}\n\nAssistant: {extract_think_and_answer(think, final)}"
    add_text(text)
    count += 1
print(f"  ✅ MetaMathQA: {count:,}")

# ── 3. Orca-Math ──────────────────────────────────────────
print("\n⏳ Orca-Math (200K sentetik)...")
try:
    ds3 = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    count = 0
    for ex in tqdm(ds3, desc="OrcaMath"):
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        if not q or not a: continue
        text = f"User: {q}\n\nAssistant: {extract_think_and_answer(a)}"
        add_text(text)
        count += 1
    print(f"  ✅ Orca-Math: {count:,}")
except Exception as e:
    print(f"  ⚠️  Orca-Math atlandı: {e}")

# ── 4. MATH dataset ───────────────────────────────────────
print("\n⏳ MATH dataset (zor problemler)...")
try:
    ds4 = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    count = 0
    for ex in tqdm(ds4, desc="MATH"):
        q        = ex.get("problem", "").strip()
        solution = ex.get("solution", "").strip()
        if not q or not solution: continue
        # \boxed{} içindeki cevabı al
        box = re.search(r'\\boxed\{([^}]+)\}', solution)
        final = box.group(1) if box else ""
        text = f"User: {q}\n\nAssistant: {extract_think_and_answer(solution, final)}"
        add_text(text)
        count += 1
    print(f"  ✅ MATH: {count:,}")
except Exception as e:
    print(f"  ⚠️  MATH atlandı: {e}")

# ── 5. NuminaMath ─────────────────────────────────────────
print("\n⏳ NuminaMath (300K yarışma problemleri)...")
try:
    ds5 = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds5, total=300_000, desc="NuminaMath"):
        if count >= 300_000: break
        q        = ex.get("problem", "").strip()
        solution = ex.get("solution", "").strip()
        if not q or not solution: continue
        box = re.search(r'\\boxed\{([^}]+)\}', solution)
        final = box.group(1) if box else ""
        text = f"User: {q}\n\nAssistant: {extract_think_and_answer(solution, final)}"
        add_text(text)
        count += 1
    print(f"  ✅ NuminaMath: {count:,}")
except Exception as e:
    print(f"  ⚠️  NuminaMath atlandı: {e}")

# ── 6. OpenR1-Math ────────────────────────────────────────
print("\n⏳ OpenR1-Math (50K derin reasoning)...")
try:
    ds6 = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds6, total=50_000, desc="OpenR1"):
        if count >= 50_000: break
        q   = ex.get("problem", "").strip()
        a   = ex.get("solution", "").strip()
        ans = ex.get("answer", "").strip()
        if not q or not a: continue
        text = f"User: {q}\n\nAssistant: {extract_think_and_answer(a, ans)}"
        add_text(text)
        count += 1
    print(f"  ✅ OpenR1-Math: {count:,}")
except Exception as e:
    print(f"  ⚠️  OpenR1-Math atlandı: {e}")

# ── 7. DeepMath-103K ──────────────────────────────────────
print("\n⏳ DeepMath-103K (olimpiyat seviyesi)...")
try:
    ds7 = load_dataset("zwhe99/DeepMath-103K", split="train")
    count = 0
    for ex in tqdm(ds7, desc="DeepMath"):
        q   = ex.get("question", "").strip()
        a   = ex.get("r1_solution_1", "").strip()
        ans = ex.get("final_answer", "").strip()
        if not q or not a: continue
        text = f"User: {q}\n\nAssistant: {extract_think_and_answer(a, ans)}"
        add_text(text)
        count += 1
    print(f"  ✅ DeepMath-103K: {count:,}")
except Exception as e:
    print(f"  ⚠️  DeepMath-103K atlandı: {e}")

# ── 8. AM-DeepSeek-R1-Distilled (100K subset) ─────────────
print("\n⏳ AM-DeepSeek-R1-Distilled (100K)...")
try:
    ds8 = load_dataset("a-m-team/AM-DeepSeek-R1-Distilled-1.4M", split="train", streaming=True)
    count = 0
    for ex in tqdm(ds8, total=100_000, desc="AM-Distill"):
        if count >= 100_000: break
        q = ex.get("problem", ex.get("question", "")).strip()
        a = ex.get("solution", ex.get("response", "")).strip()
        if not q or not a: continue
        # Zaten <think> bloğu içeriyorsa direkt kullan
        if "<think>" in a:
            text = f"User: {q}\n\nAssistant: {a}"
        else:
            text = f"User: {q}\n\nAssistant: {extract_think_and_answer(a)}"
        add_text(text)
        count += 1
    print(f"  ✅ AM-Distilled: {count:,}")
except Exception as e:
    print(f"  ⚠️  AM-Distilled atlandı: {e}")

# ── PACKING ───────────────────────────────────────────────
print(f"\n📊 Toplam örnek eklendi: {total_added:,}")
print(f"📊 Toplam sequence: {len(sequences):,}")
print("🔀 Shuffle...")
random.shuffle(sequences)

tensor = torch.tensor(sequences, dtype=torch.int64)
torch.save(tensor, OUTPUT_PT)
print(f"\n✅ Kaydedildi: {OUTPUT_PT}")
print(f"   Shape: {tensor.shape}")
