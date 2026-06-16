import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

HOME = "/home/Eren"
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{HOME}/custom_tokenizer_info")
print(f"✅ Tokenizer yüklendi | Vocab: {len(tokenizer)}")

SEQ_LENGTH = 4096
OUTPUT_PATH = f"{HOME}/processed_data/instruct_data_packed.pt"

print("📂 OpenHermes-2.5 yükleniyor...")
ds = load_dataset("teknium/OpenHermes-2.5", split="train")
print(f"✅ {len(ds)} örnek yüklendi")

def format_and_mask(example):
    conversations = example.get("conversations", [])
    if not conversations:
        return None, None
    
    input_ids = []
    labels = []
    
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "").strip()
        
        if role == "system":
            text = f"System: {value}\n"
            tokens = tokenizer.encode(text)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))  # maskelendi
            
        elif role == "human":
            text = f"User: {value}\n"
            tokens = tokenizer.encode(text)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))  # maskelendi
            
        elif role == "gpt":
            text = f"Assistant: {value}\n"
            tokens = tokenizer.encode(text)
            input_ids.extend(tokens)
            labels.extend(tokens)  # sadece assistant loss'a girer
    
    # EOS ekle
    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    
    return input_ids, labels

print("🔄 Tokenize ediliyor...")
all_input_ids = []
all_labels = []
skipped = 0

for i, example in enumerate(ds):
    if i % 50000 == 0:
        print(f"  {i}/{len(ds)} işlendi...")
    
    input_ids, labels = format_and_mask(example)
    if input_ids is None:
        skipped += 1
        continue
    
    all_input_ids.extend(input_ids)
    all_labels.extend(labels)

print(f"✅ Toplam token: {len(all_input_ids):,} | Atlanan: {skipped}")

# Pack et
num_sequences = len(all_input_ids) // SEQ_LENGTH
all_input_ids = all_input_ids[:num_sequences * SEQ_LENGTH]
all_labels = all_labels[:num_sequences * SEQ_LENGTH]

input_tensor = torch.tensor(all_input_ids, dtype=torch.int64).reshape(num_sequences, SEQ_LENGTH)
label_tensor = torch.tensor(all_labels, dtype=torch.int64).reshape(num_sequences, SEQ_LENGTH)

print(f"✅ Shape: {input_tensor.shape}")

# Dict olarak kaydet
torch.save({"input_ids": input_tensor, "labels": label_tensor}, OUTPUT_PATH)
print(f"✅ Kaydedildi: {OUTPUT_PATH}")

# İstatistik
masked = (label_tensor == -100).sum().item()
total = label_tensor.numel()
print(f"📊 Maskelenen token oranı: {masked/total*100:.1f}% (User/System)")
print(f"📊 Loss'a giren token oranı: {(total-masked)/total*100:.1f}% (Assistant)")
