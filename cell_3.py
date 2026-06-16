HOME = "/home/Eren"
# ============================================================================
# HÜCRE 4: TURBO TOKENIZER TRAINING (ENGLISH TAGS - FIXED)
# ============================================================================
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

INPUT_FILE = f"/home/Eren/unified_train.jsonl"
OUTPUT_DIR = f"/home/Eren/custom_tokenizer_info"

def train_english_tokenizer():
    print(f"🚀 Training English Tokenizer...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = [
        "<pad>",
        "<|endoftext|>",
        "<unk>",
        "<think>", "</think>",
        "User:", "Model:", "Question:",
        "<code>", "</code>",
        "[SHADOW ANALYSIS]",
        "[SIMULATIONS]",
        "[SYNTHESIS]",
        "[DECISION]"
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )
    
    def data_iterator():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    import json
                    data = json.loads(line)
                    if data.get("text"): yield data["text"]
                except: continue

    tokenizer.train_from_iterator(data_iterator(), trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(OUTPUT_DIR, "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    
    fast_tokenizer.pad_token_id = 0
    fast_tokenizer.eos_token_id = 1
    fast_tokenizer.bos_token_id = 1
    
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ English Tokenizer Ready! Vocab Size: {len(fast_tokenizer)}")
    print(f"   Pad ID: {fast_tokenizer.pad_token_id}")
    print(f"   EOS ID: {fast_tokenizer.eos_token_id}")

train_english_tokenizer()