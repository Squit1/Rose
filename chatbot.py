import torch
import sys
from transformers import PreTrainedTokenizerFast

HOME = "/home/Eren"

print("📂 Tokenizer yükleniyor...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{HOME}/custom_tokenizer_info")
print(f"✅ Tokenizer yüklendi | Vocab: {len(tokenizer)}")

print("📂 Model yükleniyor...")
checkpoint = torch.load(f"{HOME}/Noesis_Model_TPU_v5e_instruct/latest.pt", map_location='cpu', weights_only=False)
model_state = checkpoint['model_state_dict']
seq_len = model_state['pos_encoder.pe'].shape[0]
print(f"✅ Checkpoint yüklendi | Epoch: {checkpoint.get('epoch', '?')} | seq_len: {seq_len}")

sys.path.insert(0, f"{HOME}/Rose")
from train_only import UltimateTransformerModel, ModelConfig

cfg = ModelConfig()
cfg.gradient_checkpointing = False
cfg.xla_checkpoint_enabled = False
cfg.is_tpu = False
cfg.is_tpu_v5e = False
cfg.max_seq_length = seq_len
cfg.seq_length = seq_len
cfg.max_position_embeddings = seq_len

model = UltimateTransformerModel(cfg)
model.load_state_dict(model_state, strict=False)
model.eval()
print("✅ Model hazır!")

@torch.no_grad()
def generate(prompt, max_new_tokens=300, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.5):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = []

    for _ in range(max_new_tokens):
        context = input_ids[:, -seq_len:]
        out = model(context)
        logits = out.logits[:, -1, :].float() if hasattr(out, 'logits') else out[0][:, -1, :].float()

        for token_id in set(input_ids[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty

        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < values[:, -1:]] = float('-inf')

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')

        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        generated.append(next_tok.item())

        if next_tok.item() == tokenizer.eos_token_id:
            break

        decoded_so_far = tokenizer.decode(generated, skip_special_tokens=False)
        if "\nUser:" in decoded_so_far or "\n\nUser:" in decoded_so_far:
            result = decoded_so_far.split("\nUser:")[0].split("\n\nUser:")[0]
            return result.strip()

    return tokenizer.decode(generated, skip_special_tokens=True).strip()

print("\n" + "="*50)
print("🤖 Noesis Chatbot - Çıkmak için 'quit' yaz")
print("="*50)

while True:
    user_input = input("\n🗣️  Sen: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    prompt = f"User: {user_input}\n\nAssistant:"
    response = generate(prompt)
    print(f"🤖 Noesis: {response}")
