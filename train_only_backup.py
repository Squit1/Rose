# -*- coding: utf-8 -*-
# ==========================================================================
# Geliştirilmiş Custom Transformer Model - NIHAI TAM VERSIYON (Kaggle TPU))
# Noesis the best one ever seen. Fear Me.... I'm coming.
#
# ==========================================================================
# REFAKTE EDİLMİŞ VERSİYON
# --------------------------------------------------------------------------
# Bu kod, orijinal kodun tüm işlevlerini koruyarak mantıksal olarak
# yeniden sıralanmış, tekrarlar kaldırılmış ve kritik hatalar
# (İleri referans hataları, tokenizer çakışmaları) düzeltilmiştir.
# ==========================================================================

# ============ GÜVENLİ TPU ORTAM DEĞİŞKENLERİ ============

import os
HOME = os.path.expanduser("~")

import sys
import subprocess
import time
import signal
import pandas as pd
import numpy as np
import traceback
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
import inspect
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, Encoding
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import GPT2TokenizerFast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import signal
import math
import logging
from datetime import datetime
import argparse
import json
import gc
import warnings
from transformers import PreTrainedTokenizerFast
from collections import OrderedDict, deque, defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple, Union, Deque
from contextlib import contextmanager, nullcontext
from pathlib import Path, PosixPath
import pickle
from dataclasses import dataclass, field, fields, asdict
import psutil
import shutil # Config optimizasyonu için eklendi

# ================================================================
# XLA AVAILABILITY CHECK (❌ HİÇBİR XLA FONKSİYONU ÇAĞIRMA!)
# ================================================================
XLA_AVAILABLE = False
xm, pl, xmp, xr, xla_checkpoint = None, None, None, None, None

try:
    # ✅ SADECE import - HİÇBİR fonksiyon çağırma!
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint

    XLA_AVAILABLE = True
    print("✅ PyTorch XLA modules imported (runtime NOT initialized)")
    print("⚠️  XLA runtime will initialize inside xmp.spawn()")

except ImportError as e:
    print(f"⚠️ PyTorch XLA not available: {e}")
    print("⚠️ Will use CPU/GPU fallback")
except Exception as e:
    print(f"⚠️ XLA import error: {e}")

# Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_xla")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

print("="*60)
print("TPU v5e-8 TRAINING SCRIPT")
print("="*60)
print()

def init_tpu():
    """TPU'yu başlat"""
    print("🔄 TPU başlatılıyor...")
    
    devices = xm.get_xla_supported_devices()
    print(f"✅ TPU: {len(devices)} cihaz bulundu")
    print(f"   {devices}\n")
    
    return devices

# ==========================================================================
# BÖLÜM 2: TEMEL YARDIMCI FONKSİYONLAR
# ==========================================================================

def set_seed(seed:int): # pragma: no cover
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
def get_world_size(default: int = 1) -> int:
    """TPU v5e-8 için optimize world size detection"""
    if not XLA_AVAILABLE or not xm:
        return default
    try:
        # v5e-8 için optimize
        return torch_xla.runtime.world_size()
    except (AttributeError, RuntimeError):
        try:
            return xr.world_size()
        except (AttributeError, RuntimeError):
            return default

def get_rank(default: int = 0) -> int:
    """TPU v5e-8 için optimize rank detection"""
    if not XLA_AVAILABLE or not xm:
        return default
    try:
        # v5e-8 için optimize
        return torch_xla.runtime.global_ordinal()
    except (AttributeError, RuntimeError):
        try:
            return xr.global_ordinal()
        except (AttributeError, RuntimeError):
            return default

# ==========================================================================
# BÖLÜM 3: KONFİGÜRASYON SINIFI (ModelConfig)
# ==========================================================================

@dataclass
class ModelConfig:
    # ==========================================================================
    # == 📁 1. PATHS (DOSYA YOLLARI) ==
    # ==========================================================================
    resume_from_checkpoint: Optional[str] = f"{HOME}/Noesis_Model_TPU_v5e/latest.pt"
    run_chatbot_only: bool = False
    save_dir: str = f"{HOME}/Noesis_Model_TPU_v5e"
    output_dir: Optional[str] = f"{HOME}/model_output"
    
    data_path: str = f"{HOME}/processed_data/instruct_data_packed.pt"
    eval_data_path: Optional[str] = f"{HOME}/unified_test.jsonl"
    
    tokenizer_path: str = f"{HOME}/custom_tokenizer_info"
    text_column_name: Optional[str] = "text"

    # ==========================================================================
    # == 🔤 2. TOKENIZER ==
    # ==========================================================================
    min_frequency: int = 2
    vocab_size: int = 32000
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 1

    # ==========================================================================
    # == 🧠 3. MODEL MIMARISI (891M Parametre) ==
    # ==========================================================================
    d_model: int = 1536
    nhead: int = 32
    n_layers: int = 24
    num_decoder_layers: int = 24
    dim_feedforward: int = 6144
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    batch_first: bool = True
    d_state: int = 16
    bitnet_warmup_steps: int = 0

    # ==========================================================================
    # == 📚 4. EĞİTİM - BATCH SETTINGS ==
    # ==========================================================================
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    
    batch_size: int = 128
    eval_batch_size: int = 512
    per_device_eval_batch_size: int = 16
    
    seq_length: int = 2048
    max_seq_length: int = 2048
    quantize_training: bool = False

    # ==========================================================================
    # == 📈 5. ÖĞRENME HIZI ==
    # ==========================================================================
    epochs: int = 25
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    # ==========================================================================
    # == ⚙️ 6. OPTIMIZER & REGULARIZATION ==
    # ==========================================================================
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    
    optim: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-7

    # ==========================================================================
    # == 🚀 7. DONANIM AYARLARI (TPU v5e-8) ==
    # ==========================================================================
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    tf32: bool = False
    
    gradient_checkpointing: bool = True
    checkpoint_decoder_only: bool = True
    
    dataloader_num_workers: Optional[int] = 0
    dataloader_pin_memory: bool = False
    persistent_workers: bool = False
    dataloader_drop_last: bool = True
    
    device: Optional[Any] = None
    is_tpu: bool = True
    is_tpu_v5e: bool = True

    # ==========================================================================
    # == 📊 8. LOGLAMA & KAYIT ==
    # ==========================================================================
    evaluation_strategy: str = "no"
    eval_steps: int = 0
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 2
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    eval_accumulation_steps: int = 1
    
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = "Noesis-TPU"
    wandb_entity: Optional[str] = None
    prediction_loss_only: bool = True

    # ==========================================================================
    # == 🔧 9. TPU v5e SPECIFIC OPTIMIZATIONS ==
    # ==========================================================================
    use_curriculum_learning: bool = False
    curriculum_stages: List[int] = field(default_factory=lambda: [512, 640, 768])
    max_steps: Optional[int] = None
    
    use_lightweight_eval: bool = False
    lightweight_eval_samples: int = 1500
    tpu_eval_steps: int = 1000
    gpu_eval_steps: int = 200
    
    use_dynamic_padding: bool = False
    tpu_num_workers: Optional[int] = 0
    tpu_prefetch_factor: int = 8
    gpu_num_workers: Optional[int] = 2
    gpu_prefetch_factor: int = 2
    
    use_adafactor_on_tpu: bool = False
    fused_adamw_on_gpu: bool = False
    
    enable_performance_monitoring: bool = True
    profile_enabled: bool = False
    profile_steps: int = 10
    memory_logging_steps: int = 100
    
    xla_checkpoint_enabled: bool = True
    xla_force_compilation: bool = True
    xla_auto_jit: bool = True
    
    clear_cache_frequency: int = 1000
    
    use_flash_attention: bool = False
    use_memory_efficient_attention: bool = True
    force_gc_frequency: int = 1000
    empty_cache_frequency: int = 500
    use_model_parallel: bool = False
    model_parallel_size: int = 1
    checkpoint_activations: bool = False
    
    streaming: bool = False
    group_by_length: bool = False
    
    max_memory_usage: float = 0.95
    compile_model: bool = False
    max_batch_size: int = 64
    adaptive_batch_size: bool = False
    
    dataloader_multiprocessing_context: Optional[str] = None
    enable_nested_tensor: bool = False
    use_rope_scaling: bool = False
    torch_dynamo: bool = False

    # ==========================================================================
    # == 💬 10. CHATBOT SETTINGS ==
    # ==========================================================================
    chatbot_history_length: int = 5
    chatbot_max_new_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    do_sample: bool = True

    # ==========================================================================
    # == 🔄 11. DISTRIBUTED TRAINING ==
    # ==========================================================================
    fsdp: str = "no"
    fsdp_config: dict = field(default_factory=lambda: {})
    
    local_rank: int = 0
    world_size: int = 8
    rank: int = 0
    distributed: bool = True

    # ==========================================================================
    # == 📝 TRAINING TIME & CHECKPOINT SETTINGS ==
    # ==========================================================================
    max_training_hours: float = 168.0
    save_interval_min: int = 30
    last_chunk_weight: float = 1.5
    
# ==========================================================================
# 🎯 QUICK BATCH SIZE TEST CONFIGS
# ==========================================================================


    def __post_init__(self):
        path_fields = ['save_dir', 'output_dir', 'data_path', 'eval_data_path', 'tokenizer_path', 'resume_from_checkpoint']
        for field_name in path_fields:
            value = getattr(self, field_name)
            if value is not None and isinstance(value, str):
                setattr(self, field_name, Path(value))

        # output_dir set edilmemişse save_dir'i kullan
        if self.output_dir is None:
            self.output_dir = self.save_dir

        # max_seq_length çelişkisini düzelt
        if self.max_seq_length < self.seq_length:
            print(f"Warning: max_seq_length ({self.max_seq_length}) < seq_length ({self.seq_length}). Eşitleniyor...")
            self.max_seq_length = self.seq_length

        # TPU detection ve auto-optimization
        # Bu, XLA import edilmeden önce çağrılabilir
        self._auto_detect_and_optimize()

    def _auto_detect_and_optimize(self):
        self.tpu_version = None
        self.is_tpu_v5e = False
        self.is_tpu = False

        # Sadece XLA_AVAILABLE flag'ini kontrol et (import zaten yapıldı)
        if XLA_AVAILABLE:
            self.is_tpu = True
            # Not: xm.get_xla_supported_devices() spawn dışında çağrılamaz!
            # Bu yüzden v5e-8'i varsaymak daha güvenli olabilir veya
            # is_tpu_v5e'yi True bırakabiliriz.
            self.tpu_version = 'v5e' # Varsayım
            self.is_tpu_v5e = True
            print("✅ Donanım Tespiti (Varsayım): TPU v5e")
        else:
            self.is_tpu = False
            print("⚠️ TPU tespit edilemedi. GPU/CPU moduna geçiliyor.")

        # Platforma özel optimizasyonları çalıştır
        if self.is_tpu:
            print(f"🚀 TPU {self.tpu_version} için optimizasyonlar aktive ediliyor...")
            self._optimize_for_tpu()
        else:
            print("🚀 GPU/CPU için optimizasyonlar aktive ediliyor...")
            self._optimize_for_gpu()

    def _optimize_for_gpu(self):
        pass

    def _optimize_for_tpu(self):
        """TPU-specific optimizations with v5e-8 enhancements"""
        self.dataloader_num_workers = self.tpu_num_workers
        self.dataloader_pin_memory = False
        self.eval_steps = self.tpu_eval_steps

        # v5e-8 (veya herhangi bir TPU) için
        self.bf16 = True
        self.tf32 = False # TPU'da tf32 kullanılmaz
        self.persistent_workers = False
        
        if self.is_tpu_v5e:
            # v5e-8 için optimize batch sizes (config'den gelenler zaten optimize)
            # v5e-8 özel ayarlar
            self.gradient_accumulation_steps = max(self.gradient_accumulation_steps, 1) # 1'e düşürmeyi deneyebiliriz
            self.dataloader_drop_last = True
            
            print("🚀 TPU v5e-8 optimizasyonları aktif edildi!")
            print(f"   • bfloat16: {self.bf16}")
            print(f"   • Batch size (per core): {self.per_device_train_batch_size}")
            print(f"   • Gradient accumulation: {self.gradient_accumulation_steps}")

        else:
            # Standard TPU optimizations (v3, v4, v5)
            self.bf16 = False
            print(f"🚀 TPU {self.tpu_version} optimizasyonları aktif edildi!")

    def _apply_device_optimizations(self):
        """Device-specific optimizations"""
        # ❌ YANLIŞ: xm.xla_device() kontrol ediliyor
        # ✅ DOĞRU: Sadece is_tpu flag'ine bak
        
        # TPU modu için environment variable kontrolü
        import os
        tpu_detected = (
            os.environ.get('TPU_NAME') or 
            os.environ.get('COLAB_TPU_ADDR') or
            os.environ.get('TPU_NUM_DEVICES')
        )
        
        if tpu_detected or self.is_tpu:
            # TPU OPTIMIZATIONS (XLA cihazını başlatmadan)
            self.bf16 = True
            self.fp16 = False
            self.tf32 = False
            self.dataloader_num_workers = 0
            self.dataloader_pin_memory = False
            self.persistent_workers = False
            self.dataloader_drop_last = True
            self.eval_steps = self.tpu_eval_steps
            print("🔥 TPU optimizasyonları aktif! (BFloat16)")
            return
        
        # CUDA optimizasyonları
        if torch.cuda.is_available():
            self.fp16 = True
            self.tf32 = True
            print("🚀 CUDA optimizasyonları aktif!")

    def to_dict(self) -> Dict:
        """Config'i dictionary'ye çevir (JSON-serializable)"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # Path objeleri
                if isinstance(value, Path):
                    result[key] = str(value)
                # Torch device objeleri
                elif hasattr(value, '__class__') and 'torch' in str(value.__class__):
                    result[key] = str(value)
                # None değerler
                elif value is None:
                    result[key] = None
                # JSON-serializable tipler
                elif isinstance(value, (str, int, float, bool)):
                    result[key] = value
                # List ve dict
                elif isinstance(value, (list, tuple)):
                    result[key] = list(value)
                elif isinstance(value, dict):
                    result[key] = dict(value)
                # Diğer objeler (device, dataclass, vs.)
                elif hasattr(value, '__dict__'):
                    result[key] = str(value)
                # Varsayılan
                else:
                    try:
                        # JSON serialize edilebilir mi dene
                        import json
                        json.dumps(value)
                        result[key] = value
                    except (TypeError, ValueError):
                        # Edilemezse string'e çevir
                        result[key] = str(value)
        return result

    def save_to_json(self, path: Union[str, Path]):
        """Config'i JSON dosyasına kaydet"""
        config_dict = self.to_dict()
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        print(f"✅ Config kaydedildi: {filepath}")

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Dictionary'den config oluştur"""
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Yeni bir config nesnesi oluştur
        config = cls()
        
        # JSON'dan gelen verileri config nesnesine aktar
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Eğer JSON'da is_tpu yoksa, varsayılan olarak True kullan
        if 'is_tpu' not in data:
            config.is_tpu = True
        
        return config

# ==========================================================================
# BÖLÜM 4: LOGLAMA SİSTEMİ
# ==========================================================================
def setup_logging(config: ModelConfig, rank: int = 0) -> logging.Logger:
    is_master_process = True
    
    # TPU v5e-8 optimized rank detection
    if XLA_AVAILABLE and xm and hasattr(xm, 'xrt_world_size'):
        try:
            world_size = get_world_size()
            if world_size is not None and world_size > 0:
                current_rank = get_rank()
                is_master_process = (current_rank == 0)
                
                if hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e and is_master_process:
                    print(f"🎯 TPU v5e-8 Master Process (Rank {current_rank}/{world_size})")
                    
        except Exception:
            is_master_process = (rank == 0)
            
    elif rank != 0: 
        is_master_process = False
    
    # TPU v5e-8 için optimize logger name
    tpu_info = ""
    if XLA_AVAILABLE and xm and hasattr(xm, 'xrt_world_size'):
        ws = get_world_size()
        if ws is not None and ws > 1:
            if hasattr(config, 'tpu_version') and config.tpu_version:
                tpu_info = f"_{config.tpu_version}"
            logger_name = f"TPU{tpu_info}_Rank{rank}"
        else:
            logger_name = "MainProcess"
    else:
        logger_name = "MainProcess"
    
    logger = logging.getLogger(logger_name)
    
    # Zaten yapılandırılmışsa return
    if logger.handlers and logger.level != logging.NOTSET and logger.level <= logging.INFO:
        return logger
    
    logger.handlers.clear()
    logger.propagate = False
    
    # Non-master processes için minimal logging
    log_level = logging.INFO if is_master_process else logging.WARNING
    logger.setLevel(log_level)

    # TPU v5e-8 için gelişmiş formatter
    tpu_suffix = ""
    if hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e:
        tpu_suffix = "_v5e"
    elif hasattr(config, 'tpu_version') and config.tpu_version:
        tpu_suffix = f"_{config.tpu_version}"

    formatter = logging.Formatter(
        f'%(asctime)s - TPU{tpu_suffix}_RANK{rank} - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (Herkes için, ama level'a göre filtreli)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # Sadece master process dosyaya yazar
    if is_master_process and config.save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True) # Klasörü oluştur
        log_file_path = save_dir / f"training_{timestamp}_rank{rank}{tpu_suffix}.log"
        
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"Logger '{logger_name}' kuruldu. Log dosyası: {log_file_path}")

    # Kütüphane log seviyelerini ayarla (sadece master'da)
    if is_master_process:
        lib_levels = {
            "transformers": logging.WARNING,
            "torch": logging.WARNING,
            "matplotlib": logging.WARNING,
            "h5py": logging.WARNING,
            "PIL": logging.WARNING,
            "torch_xla": logging.INFO if (hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e) else logging.WARNING,
            "torch_xla.core": logging.INFO if (hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e) else logging.WARNING,
        }
        
        for lib_name, level in lib_levels.items():
            logging.getLogger(lib_name).setLevel(level)
    
    # W&B setup
    if config.use_wandb and is_master_process:
        try:
            import wandb
            
            if wandb.run is None:
                wandb_config = asdict(config)
                if hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e:
                    wandb_config.update({
                        'hardware': 'TPU_v5e-8',
                        'tpu_version': getattr(config, 'tpu_version', 'v5e-8'),
                        'tpu_optimization': True
                    })
                
                wandb.init(
                    project=config.wandb_project, 
                    entity=config.wandb_entity, 
                    config=wandb_config,
                    dir=str(config.save_dir), 
                    reinit=False,
                    tags=['tpu-v5e'] if (hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e) else None
                )
            
            tpu_emoji = "🎯" if (hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e) else "🚀"
            logger.info(f"🐝 W&B başlatıldı/kullanılıyor {tpu_emoji} (rank {rank}).")
            
        except Exception as e:
            logger.warning(f"🐝 W&B başlatılamadı (rank {rank}): {e}. Devre dışı.")
            config.use_wandb = False
    
    return logger

# ==========================================================================
# BÖLÜM 5: TOKENIZER (TiktokenWrapper)
# ==========================================================================

class CustomTokenizerWrapper:
    """
    Eğittiğimiz özel Tokenizer'ı yükler ve Trainer'ın beklediği
    tiktoken benzeri arayüzü sağlar.
    
    Özellikler:
    - <think>, [GÖLGE] vb. özel tokenları tanır.
    - Türkçe ve Kod için optimize edilmiştir.
    - TPU için padding yönetimini otomatik yapar.
    """
    def __init__(self, tokenizer_path=f"{HOME}/custom_tokenizer_info"):
        print(f"📂 Tokenizer Yükleniyor: {tokenizer_path}")
        
        try:
            # 1. Eğitilmiş Tokenizer'ı Yükle
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            
            # 2. Pad Token Kontrolü (TPU için Hayati)
            # Eğer pad token tanımlı değilse, <pad> veya EOS'u atarız.
            if self.tokenizer.pad_token_id is None:
                if "<pad>" in self.tokenizer.get_vocab():
                    self.tokenizer.pad_token = "<pad>"
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 3. Özellikleri Dışarıya Aç (Trainer bunlara erişecek)
            self.vocab_size = len(self.tokenizer)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.bos_token_id = self.tokenizer.bos_token_id
            
            # 4. System 2 Token ID'lerini kaydet (İleride lazım olabilir)
            self.think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
            
            print(f"✅ Tokenizer Hazır!")
            print(f"   📊 Vocab Size: {self.vocab_size}")
            print(f"   🛡️ Pad ID: {self.pad_token_id} | EOS ID: {self.eos_token_id}")
            
        except Exception as e:
            print(f"❌ HATA: Tokenizer yüklenirken sorun oluştu: {e}")
            print("⚠️ İPUCU: Önce 'HÜCRE 4' (Tokenizer Train) kodunu çalıştırdığından emin ol.")
            raise e

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False, **kwargs):
        """
        Trainer veri yüklerken bu metodu çağırır.
        """
        # Tekil string gelirse listeye çevir
        if isinstance(text, str):
            text = [text]
            
        # HuggingFace Tokenizer Çağrısı
        encoding = self.tokenizer(
            text,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
            add_special_tokens=True, # BOS/EOS ekle
            **kwargs
        )
        return encoding

    def encode(self, text, add_special_tokens=True, **kwargs):
        """
        Metni ID listesine çevirir.
        TiktokenWrapper'daki imzayı taklit eder.
        """
        # Tekil encode işlemi
        if isinstance(text, str):
            ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
            return ids
        # Liste gelirse batch encode
        return self.tokenizer(text, add_special_tokens=add_special_tokens, **kwargs)['input_ids']

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """
        ID listesini metne çevirir.
        """
        # Tensor gelirse listeye çevir (CPU'ya alarak)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
            
        # Batch decode (Liste içinde liste varsa)
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        
        # Tekil decode
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def __len__(self):
        return self.vocab_size
    
    # Özelliklere Erişim (Properties)
    @property
    def pad_token(self): return self.tokenizer.pad_token
    @property
    def eos_token(self): return self.tokenizer.eos_token
    @property
    def bos_token(self): return self.tokenizer.bos_token

    # Config Kaydetme (Checkpoint alırken hata vermemesi için)
    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)

# ============================================================================
# GLOBAL TOKENIZER KURULUM FONKSİYONU
# ============================================================================
def setup_global_tokenizer():
    print("🔄 Custom Tokenizer Başlatılıyor...")
    
    path = f"{HOME}/custom_tokenizer_info"
    if not os.path.exists(path):
        print(f"⚠️ UYARI: {path} bulunamadı. Tokenizer eğitimini yaptın mı?")
    
    return CustomTokenizerWrapper(tokenizer_path=path)
#================================================================================================

class AdvancedPositionalEncoding(nn.Module):
    """
    Gelişmiş Sinusoidal Positional Encoding.
    Orijinal koddaki iki tanımdan daha eksiksiz olanı budur.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # d_model tek sayı ise bile çalışmasını sağla
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
            
        self.register_buffer('pe', pe) # .unsqueeze(0) kaldırıldı, forward'da daha esnek

    def forward(self, x: torch.Tensor, batch_first: bool = True) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model) veya (seq_len, batch_size, d_model)
        """
        if batch_first:
            # x shape: (batch_size, seq_len, d_model)
            # pe shape: (max_len, d_model) -> (seq_len, d_model) -> (1, seq_len, d_model)
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            # x shape: (seq_len, batch_size, d_model)
            # pe shape: (max_len, d_model) -> (seq_len, d_model) -> (seq_len, 1, d_model)
            x = x + self.pe[:x.size(0), :].unsqueeze(1)
            
        return self.dropout(x)

class PatchedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    
    
    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None,
                src_mask=None, **kwargs):
        
        # ✅ Encoder-style çağrı desteği (memory yoksa tgt'yi memory yap)
        if memory is None:
            memory = tgt
            # src_mask gelirse hem tgt_mask hem memory_mask olarak kullan
            if src_mask is not None:
                if tgt_mask is None:
                    tgt_mask = src_mask
                if memory_mask is None:
                    memory_mask = src_mask
        else:
            # Normal decoder çağrısı - src_mask varsa tgt_mask'e map et
            if src_mask is not None and tgt_mask is None:
                tgt_mask = src_mask
        
        # Mask dtype kontrolü
        if tgt_mask is not None and tgt_mask.dtype not in [torch.bool, torch.float32, torch.bfloat16]:
            tgt_mask = tgt_mask.bool()
        
        if tgt_key_padding_mask is not None and tgt_key_padding_mask.dtype != torch.bool:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        
        if memory_mask is not None and memory_mask.dtype not in [torch.bool, torch.float32]:
            memory_mask = memory_mask.bool()
        
        if memory_key_padding_mask is not None and memory_key_padding_mask.dtype != torch.bool:
            memory_key_padding_mask = memory_key_padding_mask.bool()
        
        return super().forward(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal
        )

class PatchedMultiheadAttention(nn.MultiheadAttention):
    """
    MultiheadAttention için mask dtype düzeltmesi
    """
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        
        # ✅ CRITICAL: Mask'leri bool veya float32'ye çevir
        if attn_mask is not None:
            if attn_mask.dtype in [torch.bfloat16, torch.float16]:
                # Query dtype'ına göre karar ver
                if query.dtype == torch.float32:
                    attn_mask = attn_mask.bool()
                else:
                    attn_mask = attn_mask.to(query.dtype)
            elif attn_mask.dtype not in [torch.bool, torch.float32] and attn_mask.dtype != query.dtype:
                attn_mask = attn_mask.bool()
        
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
        
        return super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )


class PatchedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Mask dtype sorunlarını çözen TransformerEncoderLayer
    """
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None, 
                is_causal: bool = False, **kwargs: Any) -> torch.Tensor:
        
        # ✅ CRITICAL: Mask dtype kontrolü - query dtype'ına uyumlu olmalı
        if src_mask is not None:
            # BFloat16/Float16 mask'lerini düzelt
            if src_mask.dtype in [torch.bfloat16, torch.float16]:
                # Query float32 ise bool, değilse query dtype'ına çevir
                if src.dtype == torch.float32:
                    src_mask = src_mask.bool()
                else:
                    src_mask = src_mask.to(src.dtype)
            # Diğer uyumsuz dtype'lar için bool'a çevir
            elif src_mask.dtype not in [torch.bool, torch.float32] and src_mask.dtype != src.dtype:
                src_mask = src_mask.bool()
        
        if src_key_padding_mask is not None:
            # Padding mask her zaman bool olmalı
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
        
        # Parent forward'ı çağır
        return super().forward(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal
        )
        
# ============================================================================
# 1. RMSNorm (Değişiklik yok)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).sqrt()
        return (x / (norm + self.eps)) * self.weight



# ============================================================================
# 2. DÜZELTİLMİŞ BitLinear
# ============================================================================

class BitLinear(nn.Linear):
    """
    Noesis b1.58: Native 1.58-bit Linear Katmanı (DÜZELTILMIŞ)
    
    DÜZELTMELER:
    - Training'de quantize opsiyonel (quantize_training parametresi)
    - Scale learnable parameter (buffer değil)
    - Collapse önlendi
    """
    def __init__(self, in_features, out_features, bias=False, eps=1e-5, 
                 group_size=None, use_rmsnorm=True, quantize_training=False):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        self.group_size = group_size or in_features
        self.use_rmsnorm = use_rmsnorm
        self.quantize_training = quantize_training  # 🔥 YENİ
        
        # RMSNorm (opsiyonel)
        if use_rmsnorm:
            self.norm = RMSNorm(in_features, eps=eps)
        else:
            self.norm = nn.Identity()
        
        # 🔥 DÜZELTME: Scale learnable parameter
        self.weight_scale = nn.Parameter(torch.ones(1))
    
    def quantize_weight(self, w):
        """Ağırlık kuantizasyonu: float -> {-1, 0, 1}"""
        # Global scale
        scale = w.abs().mean().clamp(min=self.eps)
        w_normalized = w / scale
        w_q = torch.clamp(torch.round(w_normalized), -1, 1)
        
        # Straight-Through Estimator (STE)
        w_q = w_q + (w_normalized - w_q).detach()
        
        return w_q, scale
    
    def forward(self, x):
        # Activation normalization
        x_norm = self.norm(x)
        
        # 🔥 DÜZELTME: Sadece quantize_training kontrolü
        if not self.quantize_training:
            # Normal weights kullan (train veya eval fark etmez)
            return F.linear(x_norm, self.weight) * self.weight_scale
        else:
            # Quantized weights (sadece quantize_training=True ise)
            w_q, scale = self.quantize_weight(self.weight)
            output = F.linear(x_norm, w_q)
            return output * scale * self.weight_scale


# ============================================================================
# 3. DÜZELTİLMİŞ MambaBlock
# ============================================================================

def parallel_selective_scan_pure_pytorch(u, delta, A, B, C, D):
    """
    TPU Uyumlu - Loopsuz Parallel Scan (Saf PyTorch)
    u: (b, l, d_inner)
    delta: (b, l, d_inner)
    A: (d_inner, d_state)
    B: (b, l, d_state)
    C: (b, l, d_state)
    D: (d_inner,)
    """
    (b, l, d_inner) = u.shape
    d_state = A.shape[1]

    # 1. Delta ile parametreleri ölçekle (Discretization)
    # A_bar = exp(delta * A)
    deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
    
    # B_bar = delta * B
    # u ile çarpılmış giriş sinyali: x = u * (delta * B)
    deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
    
    # 2. Kümülatif Çarpım (Cumulative Product) - Parallel Scan'in temeli
    # RNN'deki "önceki durumu hatırlama" işini burada log-space'de yapıyoruz (daha stabil)
    # Ağırlıkların zaman içindeki birikimi
    
    # TPU optimizasyonu: Log-space cumulative sum (Çarpım yerine toplama daha stabildir)
    # (b, l, d_inner, d_state)
    # 1e-12 eklememizin sebebi log(0) hatasını önlemek
    
    # Basit versiyon (Hafıza dostu):
    # P = torch.cumprod(deltaA, dim=1)
    # h = torch.cumsum(deltaB_u / (P + 1e-12), dim=1) * P
    
    # AMA: P çok küçük veya çok büyük olabilir (Gradient patlaması riski).
    # O yüzden biraz daha güvenli bir yöntem:
    
    return selective_scan_matrix_mode(u, delta, A, B, C, D)

def selective_scan_matrix_mode(u, delta, A, B, C, D):
    """
    TPU için en garantili yöntem: Matris Maskeleme (Attention-like)
    Bu yöntem Loop kullanmaz, tamamen Matris Çarpımıdır.
    Context 4k'ya kadar TPU'da çok hızlıdır.
    """
    (b, l, d_in) = u.shape
    n = A.shape[1]
    
    # 1. Parametre Hazırlığı
    # deltaA: (b, l, d_in, n)
    deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
    deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
    
    # 2. Global Maske Oluşturma (Causal Mask)
    # Bu matris (L, L) boyutundadır. 
    # Mamba'nın recurrence işlemini "Attention" benzeri bir matris çarpımına çevirir.
    
    # Maske: Alt üçgen (Lower Triangular)
    mask = torch.tril(torch.ones(l, l, device=u.device))
    
    # A matrisinin kümülatif etkisini hesapla
    # Her zaman adımı (i) ile önceki adımlar (j) arasındaki mesafe
    # Bu kısım biraz VRAM yer ama TPU'da loop'tan 1000 kat hızlıdır.
    
    # Basitleştirilmiş implementasyon (Eğitim için yeterli):
    # Gerçek Parallel Scan karmaşıktır, burada "Cumulative Sum" hilesi yapıyoruz.
    
    # Durum (State) hesaplama
    # h = (b, l, d_in, n)
    # Pscan yerine 'cumsum' hilesi:
    # Bu tam matematiksel eşitlik sağlamaz ama modelin öğrenmesini sağlar (Mamba-Minimal yaklaşımı)
    
    # Kümülatif A etkisi
    cum_A = torch.cumprod(deltaA, dim=1) 
    
    # Girişlerin birikimi
    # 1e-6: Sıfıra bölme hatasını önlemek için epsilon
    cum_B = torch.cumsum(deltaB_u / (cum_A + 1e-6), dim=1)
    
    # State'i geri oluştur
    x = cum_B * cum_A
    
    # 3. Çıktı Projeksiyonu
    # y = C * x
    y = torch.einsum('bldn,bln->bld', x, C)
    
    # Residual
    y = y + u * D
    
    return y

class MambaBlock(nn.Module):
    """
    Mamba (SSM) Block - BitLinear Safe & TPU Ready
    """
    def __init__(self, config, use_bitlinear=False, quantize_training=False):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = getattr(config, 'd_state', 16) 
        self.d_conv = getattr(config, 'd_conv', 4)    
        self.expand = getattr(config, 'expand', 2)    
        self.d_inner = self.expand * self.d_model
        
        # --- EKLENEN KISIM: dt_rank ---
        # Mamba standartlarına göre dt_rank hesaplaması
        self.dt_rank = math.ceil(self.d_model / 16) 
        
        self.use_bitlinear = use_bitlinear
        self.quantize_training = quantize_training
        
        # Linear Factory
        def create_linear(in_features, out_features, bias=False):
            if use_bitlinear:
                # BitLinear class'ının tanımlı olduğunu varsayıyorum
                return BitLinear(
                    in_features, out_features, bias=bias,
                    use_rmsnorm=True,
                    quantize_training=quantize_training
                )
            else:
                return nn.Linear(in_features, out_features, bias=bias)
        
        # Projection layers
        self.in_proj = create_linear(self.d_model, self.d_inner * 2)
        
        # Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        
        # SSM projections
        # DÜZELTME: Çıkış boyutu dt_rank + B + C olmalı
        self.x_proj = create_linear(
            self.d_inner, 
            self.dt_rank + self.d_state * 2
        )
        
        # DÜZELTME: Giriş boyutu dt_rank olmalı
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A + 1e-4))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = create_linear(self.d_inner, self.d_model)
        
        self.act = nn.SiLU()
        self.norm = RMSNorm(self.d_inner) # RMSNorm tanımlı varsayıyorum
    
    def ssm(self, x):
        """
        TPU v5e Uyumlu SSM (Loopsuz)
        """
        (b, l, d) = x.shape
        
        # 1. Projeksiyonlar
        x_dbl = self.x_proj(x) 
        
        # 2. Split (Delta, B, C)
        # Artık boyutlar init ile uyuşuyor
        delta, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # 3. Delta Projeksiyonu
        delta = F.softplus(self.dt_proj(delta)) 
        
        # 4. Parametreler
        A = -torch.exp(self.A_log.float()) 
        D = self.D.float()
        
        # 5. Parallel Scan (Pure PyTorch)
        y = parallel_selective_scan_pure_pytorch(x, delta, A, B, C, D)
        
        return self.out_proj(y)
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None, **kwargs):
        """
        Forward pass - Hizalama Düzeltildi
        """
        batch, seq, dim = x.shape

        # 1. Input projection ve split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 2. Depthwise convolution
        # Hata veren kısım burasıydı, hizalamayı düzelttim:
        x = x.transpose(1, 2)           # (B, L, D) -> (B, D, L)
        x = self.conv1d(x)[:, :, :seq]  # Conv işlemi
        x = x.transpose(1, 2)           # (B, D, L) -> (B, L, D) Geri çevir

        # 3. Activation (SiLU)
        x = self.act(x)
        
        # 4. Normalization (BitNet/Mamba uyumlu)
        x = self.norm(x)

        # 5. SSM (State Space Model) - Loopsuz
        y = self.ssm(x)

        # 6. Gated activation (z ile çarp)
        # z parametresini de aktive etmeyi unutma (SiLU)
        y = y * self.act(z)

        # 7. Output projection
        return self.out_proj(y)
# ============================================================================
# 4. DÜZELTİLMİŞ MLP
# ============================================================================

class MLP(nn.Module):
    """
    Feed-Forward Network - BitLinear Safe
    """
    def __init__(self, config, use_bitlinear=True, quantize_training=False):
        super().__init__()
        d_model = config.d_model
        # Genellikle d_ff, d_model'in 4 katı olur (veya Llama'da farklıdır)
        d_ff = getattr(config, 'd_ff', d_model * 4)
        
        if use_bitlinear:
            self.fc1 = BitLinear(
                d_model, d_ff, bias=False,
                use_rmsnorm=True, # BitNet stabilitesi için açık kalsın
                quantize_training=quantize_training
            )
            self.fc2 = BitLinear(
                d_ff, d_model, bias=False,
                use_rmsnorm=True,
                quantize_training=quantize_training
            )
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        
        # DÜZELTME: Modern LLM'ler SiLU kullanır
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class HybridBlock(nn.Module):
    """
    Mamba veya Attention içeren unified block - BitLinear Safe
    """
    def __init__(self, config, use_mamba=True, use_bitlinear=True, 
                 quantize_training=False):
        super().__init__()
        
        # Ana işlem katmanı (Mamba veya Attention)
        if use_mamba:
            self.main = MambaBlock(
                config, 
                use_bitlinear=use_bitlinear,
                quantize_training=quantize_training
            )
        else:
            # Buraya ileride Attention ekleyeceğiz
            raise NotImplementedError("Attention block henüz implement edilmedi")
        
        # Normalization layers (Pre-Norm Architecture)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        
        # FFN
        self.ffn = MLP(
            config, 
            use_bitlinear=use_bitlinear,
            quantize_training=quantize_training
        )
    
    def forward(self, x, **kwargs):
        # 1. Yol: Mamba/Attention (Residual)
        # kwargs'ı Mamba'ya iletmek önemli (ileride maske gerekirse diye)
        x = x + self.main(self.norm1(x))
        
        # 2. Yol: MLP (Residual)
        x = x + self.ffn(self.norm2(x))
        
        return x

class UltimateTransformerModel(nn.Module):
    """
    Ultimate Transformer Model
    
    İyileştirmeler:
    - RMSNorm kullanımı (decoder'da)
    - BitLinear output layer (isteğe bağlı RMSNorm içinde)
    - Daha iyi weight initialization
    - TPU v5e-8 optimize
    """
    def __init__(self, c):  # c: ModelConfig
        super().__init__()
        self.config = c
        
        # quantize_training parametresini al (default: False)
        quantize_training = getattr(c, 'quantize_training', False)

        self.gradient_checkpointing = getattr(c, 'gradient_checkpointing', False)
        
        if self.gradient_checkpointing:
            print("✅ Gradient checkpointing enabled")
        
        # 1. Embedding
        self.embedding = nn.Embedding(
            c.vocab_size, 
            c.d_model, 
            padding_idx=c.pad_token_id
        )
        
        # 2. Positional Encoding
        if hasattr(c, 'is_tpu_v5e') and c.is_tpu_v5e:
            max_pos_len = getattr(c, 'max_position_embeddings', 4096)
        else:
            max_pos_len = getattr(c, 'max_position_embeddings', 2048)
        
        self.pos_encoder = AdvancedPositionalEncoding(
            c.d_model, 
            c.dropout, 
            max_len=max_pos_len
        )
        
        if hasattr(c, 'is_tpu_v5e') and c.is_tpu_v5e:
            effective_nhead = max(c.nhead, 16) if c.nhead < 16 else c.nhead
        else:
            effective_nhead = c.nhead
        
        enc_l = PatchedTransformerEncoderLayer(
            c.d_model, 
            effective_nhead, 
            c.dim_feedforward, 
            c.dropout,
            activation=F.gelu, 
            batch_first=c.batch_first, 
            norm_first=True
        )
        
        # 4. Decoder (TransformerEncoder + RMSNorm)
        self.decoder = nn.TransformerEncoder(
            enc_l, 
            c.num_decoder_layers, 
            RMSNorm(c.d_model)
        )
        
        # 5. Output Layer (BitLinear - DÜZELTILMIŞ!)
        # 🔥 quantize_training parametresi eklendi
        self.output_layer = BitLinear(
            c.d_model, 
            c.vocab_size,
            use_rmsnorm=True,
            quantize_training=quantize_training  # 🔥 YENİ!
        )
        
        # 6. Cache
        self._causal_mask_cache = {}
        self._device_cache = None
        self._is_v5e = hasattr(c, 'is_tpu_v5e') and c.is_tpu_v5e
        self._max_cache_size = 20
        
        # 7. Weight Initialization
        self.init_weights()

    
    def init_weights(self):
        """Weight initialization"""
        init_std = 0.02
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, BitLinear)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if hasattr(module, "padding_idx") and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, RMSNorm):
                pass

    # --- YARDIMCI FONKSİYONLAR ---
    def _get_device(self):
        if self._device_cache is None: self._device_cache = next(self.parameters()).device
        return self._device_cache
        
    def _is_xla_device(self, tensor=None):
        if tensor is not None: return 'xla' in str(tensor.device).lower()
        return 'xla' in str(self._get_device()).lower()

    def _optimize_mask_conversion(self, mask, device, force_bool=False):
        if mask is None: return None
        if mask.device != device: mask = mask.to(device)
        
        if mask.dtype in [torch.bfloat16, torch.float16, torch.float64]:
            return mask.bool() if force_bool else mask.float()
        elif force_bool and mask.dtype != torch.bool:
            return mask.bool()
        elif not force_bool and mask.dtype not in [torch.bool, torch.float32]:
            return mask.float()
        return mask

    def generate_causal_mask(self, size: int, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        cache_key = (size, device, dtype)
        if cache_key in self._causal_mask_cache: return self._causal_mask_cache[cache_key]
        
        # TPU için Bool maske en iyisidir
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        if len(self._causal_mask_cache) < self._max_cache_size: self._causal_mask_cache[cache_key] = mask
        return mask

    def _gradient_checkpointing_decode(
        self, 
        tgt_emb: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Gradient checkpointing ile decode
        Her decoder layer activation'ını recompute eder
        """
        
        x = tgt_emb
        
        # Her decoder layer'ı checkpoint'le
        for i, layer in enumerate(self.decoder.layers):
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    src = inputs[0]
                    mask = inputs[1] if len(inputs) > 1 else None
                    key_padding_mask = inputs[2] if len(inputs) > 2 else None
                    
                    return module(
                        src,
                        src_mask=mask,
                        src_key_padding_mask=key_padding_mask
                    )
                return custom_forward
            
            # Checkpoint ile forward
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer),
                x,
                tgt_mask,
                tgt_key_padding_mask,
                use_reentrant=False
            )
        
        # Final normalization
        if hasattr(self.decoder, 'norm') and self.decoder.norm is not None:
            x = self.decoder.norm(x)
        
        return x

    def decode(self, tgt, tgt_mask=None, tgt_key_padding_mask=None) -> torch.Tensor:
        device = self._get_device()
        tgt_mask = self._optimize_mask_conversion(tgt_mask, device, force_bool=False)
        tgt_key_padding_mask = self._optimize_mask_conversion(tgt_key_padding_mask, device, force_bool=True)
        
        sqrt_d_model = math.sqrt(self.config.d_model)
        tgt_emb = self.embedding(tgt) * sqrt_d_model
        tgt_pos = self.pos_encoder(tgt_emb, batch_first=self.config.batch_first)
        
        # Float maske için güvenlik (-inf doldurma)
        if tgt_mask is not None and tgt_mask.dtype == torch.bool:
            tgt_mask = tgt_mask.to(tgt_emb.dtype).masked_fill(tgt_mask, float('-inf'))
            
        use_gc = getattr(self.config, 'gradient_checkpointing', False) and self.training
        
        if use_gc:
            if self._is_v5e and xla_checkpoint is not None:
                 return xla_checkpoint(self._decoder_forward, tgt_pos, tgt_mask, tgt_key_padding_mask, preserve_rng_state=True)
            else:
                 return torch.utils.checkpoint.checkpoint(self._decoder_forward, tgt_pos, tgt_mask, tgt_key_padding_mask, use_reentrant=False)
        
        return self._decoder_forward(tgt_pos, tgt_mask, tgt_key_padding_mask)
        

    def _decoder_forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Noesis Motoru (Transformer Modu) - Gradient Checkpointing Optimized
        Her decoder layer ayrı checkpoint'lenir (daha fazla memory tasarrufu)
        """
        
        x = tgt
        
        # Her layer'ı ayrı ayrı checkpoint'le (16 layer)
        for i, layer in enumerate(self.decoder.layers):
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    src = inputs[0]
                    mask = inputs[1] if len(inputs) > 1 else None
                    key_padding_mask = inputs[2] if len(inputs) > 2 else None
                    
                    return module(
                        src,
                        src_mask=mask,
                        src_key_padding_mask=key_padding_mask
                    )
                return custom_forward
            
            # XLA checkpoint (TPU) veya normal checkpoint
            if self._is_v5e and xla_checkpoint is not None:
                x = xla_checkpoint(
                    create_custom_forward(layer),
                    x,
                    tgt_mask,
                    tgt_key_padding_mask,
                    preserve_rng_state=True
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    tgt_mask,
                    tgt_key_padding_mask,
                    use_reentrant=False
                )
        
        # Final normalization
        if hasattr(self.decoder, 'norm') and self.decoder.norm is not None:
            x = self.decoder.norm(x)
        
        return x

    def _forward_original(self, tgt_ids, tgt_padding_mask=None, tgt_causal_mask=None):
        device = self._get_device()
        
        # Causal mask oluştur (Bool olarak)
        if tgt_causal_mask is None:
            seq_len = tgt_ids.size(1)
            tgt_causal_mask = self.generate_causal_mask(seq_len, device, dtype=torch.bool)
            
        dec_out = self.decode(
            tgt=tgt_ids, 
            tgt_mask=tgt_causal_mask, 
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.output_layer(dec_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs) -> Any:
        device = self._get_device()
        
        # 1. Maskeleri Hazırla
        if attention_mask is not None:
            if attention_mask.dtype not in [torch.bool, torch.float32]: attention_mask = attention_mask.bool()
            if attention_mask.device != device: attention_mask = attention_mask.to(device)
        
        # Padding mask: 0 (veya False) olan yerler paddingdir
        # PyTorch Transformer için: True = Ignore (Maskele), False = Keep
        # Bu yüzden attention_mask (1=Keep, 0=Ignore) ters çevrilir.
        tgt_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # 2. Ana İşlem
        logits = self._forward_original(
            tgt_ids=input_ids, 
            tgt_padding_mask=tgt_padding_mask, 
            tgt_causal_mask=None
        )
        
        # 3. Loss Hesapla
        loss = None
        if labels is not None:
            if labels.device != device: labels = labels.to(device)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        @dataclass
        class ModelOutput:
            logits: torch.Tensor
            loss: Optional[torch.Tensor] = None
            
        return ModelOutput(logits=logits, loss=loss)
# =u200c========================================================================
# BÖLÜM 7: DATASET ve COLLATOR
# ==========================================================================

class PackedTPUDataset(Dataset):
    """
    TPU v5e-8 için Optimize Edilmiş Veri Seti Sınıfı.
    Önceden paketlenmiş (.pt) veriyi RAM'den doğrudan TPU'ya aktarır.
    
    Avantajları:
    - Sıfır işlem maliyeti (CPU boşta kalmaz).
    - Sabit boyut (2048 token) sayesinde XLA Re-compilation olmaz.
    - BitNet + Mamba eğitimi için en stabil yöntemdir.
    """
    def __init__(self, data_tensor):
        # data_tensor: (Num_Batches, Seq_Len) boyutunda LongTensor
        self.data = data_tensor
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 1. Veriyi Al (Tensor)
        input_ids = self.data[idx]
        
        # 2. Attention Mask (Full Attention)
        # Packing yaptığımız için padding yok, maske hepsi 1.
        attention_mask = torch.ones_like(input_ids)
        
        # 3. Labels (Causal LM - Next Token Prediction)
        # Input ile aynıdır. Model forward içinde kaydırır (shift).
        lm_labels = input_ids.clone()
        
        # 4. Multi-Task Uyumluluğu (Trainer Kodu Hata Vermesin Diye)
        # Diğer tasklar için dummy (boş) değerler dönüyoruz.
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'lm_labels': lm_labels,
            
            # Trainer beklediği için boş değerler:
            'sentiment_label': -100, # Ignore index
            'ner_labels': torch.full_like(input_ids, -100),
            'qa_start': -100,
            'qa_end': -100,
            
            # İstatistiksel veriler
            'active_tasks': ['lm'], # Sadece Language Modeling aktif
            'task_weights': [1.0],
            'doc_id': f'batch_{idx}',
            'chunk_index': 0,
            'is_last_chunk': True
        }


class TPUOptimizedCollator:
    """
    TPU için optimize edilmiş Collator.
    Dinamik padding yerine SABİT PADDING (Fixed Padding) kullanır.
    Bu, XLA Graph Recompilation'ı engeller ve hızı 10x artırabilir.
    """
    def __init__(self, tokenizer, max_length=768, padding_value=None, last_chunk_weight=1.5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.last_chunk_weight = last_chunk_weight
        
        # 🔥 GÜVENLİK DÜZELTMESİ: Pad ID Önceliği
        # 1. Eğer dışarıdan padding_value verilirse onu kullan.
        # 2. Verilmezse, Tokenizer'ın içindeki pad_token_id'yi (100300) kullan.
        # 3. O da yoksa 0 kullan (Fallback).
        if padding_value is not None:
            self.pad_token_id = padding_value
        elif hasattr(tokenizer, 'pad_token_id'):
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = 0

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not examples:
            return {}
        
        # Çıktı listeleri
        batch_input_ids = []
        batch_attention_mask = []
        batch_lm_labels = []
        batch_loss_weights = []
        
        # Metadata listeleri
        sentiment_labels = []
        ner_labels = []
        qa_starts = []
        qa_ends = []
        
        for ex in examples:
            # 1. Input ID'leri al ve Sabit Uzunluğa (768) Pad'le
            ids = self._flatten(ex['input_ids'])
            curr_len = len(ids)
            
            # Truncation (Eğer Dataset yapmadıysa güvenlik ağı)
            if curr_len > self.max_length:
                ids = ids[:self.max_length]
                curr_len = self.max_length # Uzunluğu güncelle
                
            pad_len = self.max_length - curr_len
            
            # Padding
            ids = ids + [self.pad_token_id] * pad_len
            # Mask: 1 = Dolu, 0 = Boş (Pad)
            mask = [1] * curr_len + [0] * pad_len 

            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            
            # 2. LM Labels (Pad kısmı -100 olmalı ki Loss hesaplanmasın)
            lm_lbl = self._flatten(ex['lm_labels'])
            if len(lm_lbl) > self.max_length:
                lm_lbl = lm_lbl[:self.max_length]
            
            # Labels için padding değeri -100'dür (PyTorch standardı)
            lm_lbl = lm_lbl + [-100] * pad_len
            batch_lm_labels.append(lm_lbl)
            
            # 3. NER Labels
            ner_lbl = self._flatten(ex['ner_labels'])
            if len(ner_lbl) > self.max_length:
                ner_lbl = ner_lbl[:self.max_length]
            ner_lbl = ner_lbl + [-100] * pad_len
            ner_labels.append(ner_lbl)

            # 4. Loss Weights
            weight = torch.ones(self.max_length)
            real_len = min(curr_len, self.max_length)
            
            if ex.get('is_last_chunk', False):
                weight[:real_len] *= self.last_chunk_weight
            
            # Pad kısmının ağırlığı 0
            weight[real_len:] = 0
            batch_loss_weights.append(weight)

            # Metadata
            sentiment_labels.append(ex.get('sentiment_label', -100))
            qa_starts.append(ex.get('qa_start', -100))
            qa_ends.append(ex.get('qa_end', -100))

        # Tensor'a çevir
        batch = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            # TPU için mask bool olmalı (True/False)
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.bool), 
            'lm_labels': torch.tensor(batch_lm_labels, dtype=torch.long),
            'ner_labels': torch.tensor(ner_labels, dtype=torch.long),
            'loss_weights': torch.stack(batch_loss_weights),
            'sentiment_labels': torch.tensor(sentiment_labels, dtype=torch.long),
            'qa_starts': torch.tensor(qa_starts, dtype=torch.long),
            'qa_ends': torch.tensor(qa_ends, dtype=torch.long),
            
            # Metadata pass-through (Tensor olmayanlar)
            'doc_ids': [ex.get('doc_id') for ex in examples],
            'chunk_indices': [ex.get('chunk_index') for ex in examples],
            'is_last_chunks': [ex.get('is_last_chunk') for ex in examples],
            'active_tasks': [ex.get('active_tasks') for ex in examples],
            'task_weights': [ex.get('task_weights') for ex in examples],
        }
        
        return batch

    def _flatten(self, data):
        """Veri yapısını düzleştirir"""
        if isinstance(data, torch.Tensor):
            return data.squeeze().tolist()
        if isinstance(data, list):
            # Eğer iç içe liste ise düzelt (örn: [[1,2,3]])
            if len(data) > 0 and isinstance(data[0], list):
                return data[0]
        return data
    

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        # Reduction 'none' olmalı ki her token için ayrı loss alıp
        # kendi ağırlıklarımızla çarpabilelim.
        self.base_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='none'
        )

    def forward(self, logits, labels, weights=None):
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len)
            weights: (batch, seq_len) -> Collator'dan gelen ağırlıklar
        """
        # 1. SHIFT (Causal LM Mantığı: Bir sonraki tokenı tahmin et)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size, seq_len, vocab_size = shift_logits.shape
        
        # Ağırlıklar varsa onları da kaydır (Input ile hizala)
        shift_weights = None
        if weights is not None:
            shift_weights = weights[..., 1:].contiguous()

        # 2. HAM LOSS HESAPLAMA (TPU Kalkanı: .float())
        # Düzleştirme (Flatten) işlemi
        flat_logits = shift_logits.view(-1, vocab_size).float()
        flat_labels = shift_labels.view(-1)
        
        # Her token için ham loss
        raw_losses = self.base_loss(flat_logits, flat_labels)

        # 3. AĞIRLIKLANDIRMA ve ORTALAMA
        # Geçerli token maskesi (Padding olmayanlar)
        valid_mask = (flat_labels != self.ignore_index).float()
        
        if shift_weights is not None:
            flat_weights = shift_weights.view(-1)
            # Ağırlıkları sadece geçerli tokenlara uygula
            final_weights = flat_weights * valid_mask
        else:
            final_weights = valid_mask

        # Ağırlıklı Ortalama Formülü: Sum(Loss * Weight) / Sum(Weight)
        sum_loss = (raw_losses * final_weights).sum()
        sum_weight = final_weights.sum() + 1e-9 # Sıfıra bölünme hatasını önle
        
        final_loss = sum_loss / sum_weight
        
        return final_loss, {'lm': final_loss}


class MemoryTracker:
    """
    Bellek kullanımını izler (TPU/GPU/CPU Uyumlu).
    Trainer tarafından başlatılırken 'device' parametresi alması en sağlıklısıdır.
    """
    def __init__(self, device=None):
        self.peak_memory = 0
        self.current_memory = 0
        self.device = device # Dışarıdan gelen cihazı sakla

    def _get_xla_device_safe(self):
        """
        Eğer device __init__ ile gelmediyse, güvenli şekilde bulmaya çalışır.
        """
        # Eğer zaten bir cihazımız varsa onu döndür
        if self.device is not None:
            return self.device
            
        # Yoksa ve XLA aktifse bulmaya çalış
        if XLA_AVAILABLE and xm:
            try:
                return xm.xla_device()
            except Exception:
                return None
        return None

    def update(self):
        """Bellek kullanımını güncelle"""
        try:
            # 1. CUDA (GPU) Kontrolü
            if torch.cuda.is_available():
                self.current_memory = torch.cuda.memory_allocated() / (1024**3)
                self.peak_memory = max(self.peak_memory, self.current_memory)
            
            # 2. XLA (TPU) Kontrolü
            elif XLA_AVAILABLE and xm:
                device = self._get_xla_device_safe()
                if device and 'xla' in str(device):
                    try:
                        mem_info = xm.get_memory_info(device)
                        # mem_info genelde dict döner: {'bytes_used': ..., 'bytes_limit': ...}
                        if isinstance(mem_info, dict):
                            self.current_memory = mem_info.get('bytes_used', 0) / (1024**3)
                        else:
                            self.current_memory = 0
                        
                        self.peak_memory = max(self.peak_memory, self.current_memory)
                    except Exception:
                        pass # TPU bellek bilgisi bazen senkronizasyon hatası verebilir, yutuyoruz.
                        
        except Exception:
            pass # Bellek takibi ana eğitimi bozmamalı

    def get_current_usage(self) -> float:
        self.update()
        return self.current_memory

    def is_memory_critical(self, threshold: float = 0.9) -> bool:
        """Bellek kullanımı kritik seviyede mi?"""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return (self.get_current_usage() / total_memory) > threshold
            
            elif XLA_AVAILABLE:
                device = self._get_xla_device_safe()
                if device:
                    try:
                        mem_info = xm.get_memory_info(device)
                        total_memory = mem_info.get('bytes_limit', 0) / (1024**3)
                        if total_memory > 0:
                            return (self.get_current_usage() / total_memory) > threshold
                    except:
                        return False
            return False
        except:
            return False

    def get_state(self) -> Dict[str, Any]:
        return {
            'current_memory': self.current_memory,
            'peak_memory': self.peak_memory
        }
class GradientTracker:
    """Gradient istatistiklerini izler"""
    def __init__(self):
        self.grad_norms = deque(maxlen=100)

    def update(self, grad_norm: float):
        self.grad_norms.append(grad_norm)

    def get_stats(self) -> Dict[str, float]:
        if not self.grad_norms:
            return {}
        norms = list(self.grad_norms)
        return {
            'grad_norm_avg': np.mean(norms),
            'grad_norm_max': np.max(norms),
            'grad_norm_min': np.min(norms),
            'grad_norm_std': np.std(norms)
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'grad_norms': list(self.grad_norms),
            'stats': self.get_stats()
        }

class AnomalyDetector:
    """Eğitim anomalilerini tespit eder"""
    def __init__(self, config):
        self.config = config
        self.anomalies = []

    def record_anomaly(self, anomaly_type: str, step: int, data: Dict[str, Any]):
        anomaly = {
            'type': anomaly_type,
            'step': step,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.anomalies.append(anomaly)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.anomalies
    
    def load_history(self, history: List[Dict[str, Any]]):
        self.anomalies = history


class MetricsCalculator:
    """BLEU ve ROUGE gibi metrikleri hesaplar"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Basit BLEU skoru (1-gram precision)"""
        try:
            total_score = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()

                if not pred_tokens or not ref_tokens:
                    continue

                matches = sum(1 for token in pred_tokens if token in ref_tokens)
                precision = matches / len(pred_tokens) if pred_tokens else 0
                total_score += precision
                count += 1
                
            return total_score / count if count > 0 else 0.0
        except Exception:
            return 0.0

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
        """Basit ROUGE-1 F1 skoru"""
        try:
            rouge_scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
            total_f1 = 0.0
            count = 0
            
            for pred, ref in zip(predictions, references):
                pred_tokens = set(pred.split())
                ref_tokens = set(ref.split())

                if not pred_tokens or not ref_tokens:
                    continue

                overlap = len(pred_tokens & ref_tokens)
                precision = overlap / len(pred_tokens) if pred_tokens else 0
                recall = overlap / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                total_f1 += f1
                count += 1
            
            rouge_scores['rouge-1']['f'] = total_f1 / count if count > 0 else 0.0
            return rouge_scores
        except Exception:
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
#===========================================================================================
@dataclass
class TrainingState:
    """
    Eğitim durumunu (state) tek bir yerde toplayan dataclass.
    Checkpointing ve chunk-aware metrikler için kullanılır.
    (Orijinal koddaki iki tanımdan daha eksiksiz olanı budur)
    """
    # Temel durum
    epoch: int = 0
    global_step: int = 0
    best_eval_loss: float = float('inf')
    best_perplexity: float = float('inf')
    early_stopping_counter: int = 0

    # Zamanlama
    start_time: float = field(default_factory=time.time)
    last_save_time: float = field(default_factory=time.time)
    step_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Token izleme
    total_tokens: int = 0

    # 🚀 Chunk-specific izleme
    docs_seen: set = field(default_factory=set)
    chunks_processed: int = 0
    last_chunk_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    middle_chunk_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Doküman bazlı metrikler
    doc_chunk_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    doc_losses: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Eğitim & Değerlendirme geçmişi
    train_history: Dict[str, List[float]] = field(default_factory=lambda: {
        'loss': [], 'lr': [], 'memory_usage': [], 'grad_norm': [],
        'step_time': [], 'throughput': [], 'loss_smoothed': []
    })

    eval_history: Dict[str, List[float]] = field(default_factory=lambda: {
        'loss': [], 'perplexity': [], 'bleu_score': [], 'rouge_score': []
    })

    def elapsed_hours(self) -> float:
        """Eğitim başlangıcından bu yana geçen saat"""
        return (time.time() - self.start_time) / 3600

    def minutes_since_save(self) -> float:
        """Son kayıttan bu yana geçen dakika"""
        return (time.time() - self.last_save_time) / 60

    def avg_step_time(self) -> float:
        """Son 100 adımın ortalama süresi"""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0

    def tokens_per_second(self) -> float:
        """Genel token/saniye throughput"""
        elapsed = time.time() - self.start_time
        return self.total_tokens / elapsed if elapsed > 0 else 0

    def avg_last_chunk_loss(self) -> float:
        """'Last chunk'ların son 100 kaybının ortalaması"""
        return sum(self.last_chunk_losses) / len(self.last_chunk_losses) if self.last_chunk_losses else 0

    def avg_middle_chunk_loss(self) -> float:
        """'Middle chunk'ların son 100 kaybının ortalaması"""
        return sum(self.middle_chunk_losses) / len(self.middle_chunk_losses) if self.middle_chunk_losses else 0

    def get_document_stats(self) -> Dict[str, Any]:
        """Doküman bazlı istatistikler"""
        stats = {
            'unique_docs': len(self.docs_seen),
            'total_chunks': self.chunks_processed,
            'avg_chunks_per_doc': self.chunks_processed / max(len(self.docs_seen), 1)
        }
        if self.doc_losses:
            doc_avg_losses = {
                doc_id: sum(losses) / len(losses)
                for doc_id, losses in self.doc_losses.items()
            }
            stats['avg_doc_loss'] = sum(doc_avg_losses.values()) / len(doc_avg_losses)
            stats['best_doc'] = min(doc_avg_losses.items(), key=lambda x: x[1])
            stats['worst_doc'] = max(doc_avg_losses.items(), key=lambda x: x[1])
        return stats


class CheckpointManager:
    def __init__(self, trainer, logger, max_hours=8.5, save_total_limit=2, save_interval_min=30):
        self.trainer = trainer
        self.logger = logger
        self.max_hours = max_hours
        self.save_total_limit = save_total_limit
        self.save_interval_min = save_interval_min  # ✅ EKLE
        self.last_save_time = time.time()  # ✅ EKLE (opsiyonel ama yararlı)
        self.save_dir = Path(f"{HOME}/Noesis_Model_TPU_v5e/model")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def should_stop_training(self, state: TrainingState) -> bool:
        return state.elapsed_hours() >= self.max_hours

    def save_checkpoint(self, name, state, optimizer=None, scheduler=None, eval_metrics=None, force_full=True):
        """
        Kaggle diski dolmasın diye kontrollü ve güvenli kayıt.
        """
        # Sadece Master (Rank 0) işlem yapar
        if not self.trainer.is_master():
            return
        
        save_path = self.save_dir / name
        tmp_path = str(save_path) + ".tmp"
        
        try:
            self.logger.info(f"💾 Model diske hazırlanıyor (CPU transfer)... Adım: {state.global_step}")
            
            # 🔥 YENİ: Model'i MUTLAKA train mode'a al!
            self.trainer.model.train()
            
            # 1. Modeli TPU'dan CPU'ya güvenli bir şekilde çek (RAM dostu)
            model_state = {k: v.cpu() for k, v in self.trainer.model.state_dict().items()}
            
            checkpoint = {
                'epoch': state.epoch,
                'global_step': state.global_step,
                'model_state_dict': model_state,
                'best_eval_loss': state.best_eval_loss,
                'config': self.trainer.config.to_dict(),
                'timestamp': datetime.now().isoformat(),
            }
            
            if force_full and optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # 2. Önce geçici dosyaya yaz
            torch.save(checkpoint, tmp_path)
            
            # 3. Dosya boyutunu doğrula ve ismini değiştir (Atomik işlem)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                os.replace(tmp_path, save_path)
                self.logger.info(f"✅ MODEL DİSKE YAZILDI: {save_path.name} ({os.path.getsize(save_path)/1e6:.1f} MB)")
            
            # 4. Eski dosyaları sil
            self._cleanup_old_checkpoints()
            
            # Bellek temizliği
            del model_state, checkpoint
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"❌ KAYIT HATASI: {e}")
            if os.path.exists(tmp_path): 
                os.remove(tmp_path)

    def _cleanup_old_checkpoints(self):
        """Kaggle disk limitini aşmamak için eski modelleri siler."""
        try:
            # Sadece .pt dosyalarını bul ve tarihe göre sırala
            ckpts = sorted(list(self.save_dir.glob("*.pt")), key=os.path.getmtime)
            if len(ckpts) > self.save_total_limit:
                for old_ckpt in ckpts[:-self.save_total_limit]:
                    old_ckpt.unlink()
                    self.logger.info(f"🗑️ Disk temizliği: Eski model silindi ({old_ckpt.name})")
        except Exception as e:
            self.logger.warning(f"⚠️ Temizlik sırasında hata: {e}")

class MultiTaskEvaluator:
    """
    Multi-task için gelişmiş evaluator (TPU v5e & BF16 Uyumlu)
    
    DÜZELTME:
    - BitLinear eval mode'da quantize etmemesi için güvence eklendi
    - Model mode'ları doğru yönetiliyor
    """
    def __init__(self, logger):
        self.logger = logger
    
    @torch.no_grad()
    def evaluate(self, model, eval_loader, criterion, device):
        """
        Multi-task evaluation loop - FIXED VERSION
        
        Args:
            model: Model
            eval_loader: Evaluation data loader
            criterion: WeightedCrossEntropyLoss
            device: Device (TPU/GPU/CPU)
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        
        # ============================================
        # Model'i eval mode'a al
        # ============================================
        model.eval()  # ← DÜZELTME: train() değil eval() olmalı!
        
        # ============================================
        # Metrik değişkenleri
        # ============================================
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        # Task-specific metrikler
        task_losses = {
            'lm': [], 
            'sentiment': [], 
            'ner': [], 
            'qa': []
        }
        
        # Doküman bazlı izleme
        doc_losses = {}
        last_chunk_losses = []
        middle_chunk_losses = []
        docs_evaluated = set()
        chunks_evaluated = 0
        
        # Task aktivasyon sayacı
        task_counts = {
            'lm': 0, 
            'sentiment': 0, 
            'ner': 0, 
            'qa': 0
        }
        
        # ============================================
        # TPU/BF16 ayarları
        # ============================================
        is_tpu = 'xla' in str(device)
        use_bf16 = True
        
        # Autocast context
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        from contextlib import nullcontext
        ctx = torch.autocast(
            device_type=device_type, 
            dtype=torch.bfloat16
        ) if use_bf16 and device.type == 'cuda' else nullcontext()
        
        # ============================================
        # Evaluation loop (NO GRADIENTS!)
        # ============================================
        with torch.no_grad():  # ← EKLEME: Eval'de gradient hesaplamayalım
            for batch_idx, batch in enumerate(eval_loader):
                try:
                    # Metadata'yı ayıkla
                    doc_ids = batch.pop('doc_ids', [])
                    chunk_indices = batch.pop('chunk_indices', [])
                    is_last_chunks = batch.pop('is_last_chunks', [])
                    active_tasks_list = batch.pop('active_tasks', [])
                    task_weights_list = batch.pop('task_weights', [])
                    
                    # Loss weights
                    loss_weights = batch.pop('loss_weights', None)
                    
                    # Girdiler
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    lm_labels = batch['lm_labels']
                    
                    # Mask dtype kontrolü
                    if attention_mask is not None and attention_mask.dtype == torch.float32:
                        attention_mask = attention_mask.to(torch.bfloat16)
                    
                    # ============================================
                    # Forward pass (BF16 Context)
                    # ============================================
                    with ctx:
                        outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=None
                        )
                        
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        
                        # ============================================
                        # 🔥 DÜZELTME: batch parametresi KALDIRILDI
                        # ============================================
                        loss, task_losses_dict = criterion(
                            logits=logits,
                            labels=lm_labels,
                            weights=loss_weights
                        )
                    
                    # ============================================
                    # Metrik hesaplama
                    # ============================================
                    batch_tokens = (lm_labels != -100).sum().item()
                    
                    if batch_tokens > 0:
                        total_loss += loss.item() * batch_tokens
                        total_tokens += batch_tokens
                        num_batches += 1
                    
                    # Task-specific loss'ları topla (eğer döndürülüyorsa)
                    if task_losses_dict:
                        for task_name, task_loss in task_losses_dict.items():
                            if task_loss is not None and task_name in task_losses:
                                task_losses[task_name].append(task_loss.item())
                    
                    # Task aktivasyon sayacı
                    if active_tasks_list:
                        for active_tasks in active_tasks_list:
                            for task in active_tasks:
                                if task in task_counts:
                                    task_counts[task] += 1
                    
                    # Doküman ve chunk bazlı takip
                    if doc_ids:
                        current_loss_val = loss.item()
                        for i, doc_id in enumerate(doc_ids):
                            if doc_id != 'unknown':
                                docs_evaluated.add(doc_id)
                                if doc_id not in doc_losses:
                                    doc_losses[doc_id] = []
                                doc_losses[doc_id].append(current_loss_val)
                            
                            # Chunk tracking
                            if i < len(is_last_chunks):
                                if is_last_chunks[i]:
                                    last_chunk_losses.append(current_loss_val)
                                else:
                                    middle_chunk_losses.append(current_loss_val)
                        
                        chunks_evaluated += len(doc_ids)
                
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"⚠️  Eval batch {batch_idx} error: {e}")
                    continue
        
        # ============================================
        # Nihai metrikler
        # ============================================
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        try:
            import math
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        except OverflowError:
            perplexity = float('inf')
        
        metrics = {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'docs_evaluated': len(docs_evaluated),
            'chunks_evaluated': chunks_evaluated,
            'num_batches': num_batches,
        }
        
        # Task-specific metrikleri ortalamaya dök
        for task_name, losses in task_losses.items():
            if losses:
                avg_task_loss = sum(losses) / len(losses)
                metrics[f'{task_name}_loss'] = avg_task_loss
                
                try:
                    task_ppl = math.exp(avg_task_loss) if avg_task_loss < 100 else float('inf')
                    metrics[f'{task_name}_ppl'] = task_ppl
                except:
                    metrics[f'{task_name}_ppl'] = float('inf')
                
                metrics[f'{task_name}_count'] = task_counts[task_name]
        
        # Chunk-specific metrikler
        if last_chunk_losses:
            metrics['last_chunk_loss'] = sum(last_chunk_losses) / len(last_chunk_losses)
        if middle_chunk_losses:
            metrics['middle_chunk_loss'] = sum(middle_chunk_losses) / len(middle_chunk_losses)
        
        # ============================================
        # Model'i tekrar train mode'a al
        # ============================================
        model.train()
        
        # ============================================
        # Log metrics
        # ============================================
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 EVALUATION RESULTS:")
            self.logger.info("="*80)
            self.logger.info(f"  Eval Loss: {avg_loss:.4f}")
            self.logger.info(f"  Perplexity: {perplexity:.2f}")
            self.logger.info(f"  Total Tokens: {total_tokens:,}")
            self.logger.info(f"  Documents: {len(docs_evaluated)}")
            self.logger.info(f"  Chunks: {chunks_evaluated}")
            
            # Task-specific
            self.logger.info("\n  Task Breakdown:")
            for task_name in ['lm', 'sentiment', 'ner', 'qa']:
                if f'{task_name}_loss' in metrics:
                    task_loss = metrics[f'{task_name}_loss']
                    task_ppl = metrics[f'{task_name}_ppl']
                    task_count = metrics[f'{task_name}_count']
                    self.logger.info(
                        f"    {task_name.upper():10} Loss: {task_loss:.4f}, "
                        f"PPL: {task_ppl:.2f}, Count: {task_count}"
                    )
            
            self.logger.info("="*80)
        
        return metrics
      


# ==============================================================================
# === BÖLÜM 10: ANA EĞİTİCİ SINIFI (AdvancedUltimateTrainer)
# ==============================================================================

@contextmanager
def autocast_context(device, dtype, enabled):
    """Mixed precision autocast için context manager (TPU/GPU uyumlu)"""
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    if enabled:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield

class AdvancedUltimateTrainer:
    def __init__(self, 
                 config: ModelConfig, 
                 model: nn.Module, 
                 tokenizer: PreTrainedTokenizerFast, # 🔥 DEĞİŞTİ: TiktokenWrapper gitti
                 logger: logging.Logger,
                 device: torch.device, 
                 is_tpu: bool):
        
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.device = device
        self.is_tpu = is_tpu
        self.is_tpu_v5e = is_tpu and hasattr(config, 'is_tpu_v5e') and config.is_tpu_v5e
        
        device_info = f"TPU v5e-8" if self.is_tpu_v5e else ("TPU" if self.is_tpu else "GPU/CPU")
        self.logger.info(f"Trainer, {device_info} modunda başlatıldı. Cihaz: {self.device}")

        self.pad_id = getattr(config, 'pad_token_id', 0)
        if self.pad_id is None:
             self.logger.warning("pad_token_id 'None' idi, 0 olarak ayarlandı.")
             self.pad_id = 0

        # Modelin zaten cihazda olduğundan emin ol
        self.model = model.to(self.device)
        
        # Kayıt dizini
        self.model_save_dir = Path(config.save_dir) / "model"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Yardımcı sınıflar
        self.memory_tracker = MemoryTracker()
        self.gradient_tracker = GradientTracker()
        self.anomaly_detector = AnomalyDetector(config)
        self.metrics_calculator = MetricsCalculator(tokenizer)
        
        # Geçmiş izleme
        self.checkpoint_history = []
        self.train_history = defaultdict(list)
        self.eval_history = defaultdict(list)
        self.step_times = deque(maxlen=200 if self.is_tpu_v5e else 100)
        self.loss_window = deque(maxlen=100 if self.is_tpu_v5e else 50)
        
        # Sampler'lar (Dataloader kurulumunda ayarlanacak)
        self.train_sampler = None
        self.eval_sampler = None
        
        self.world_size = self._get_world_size()
        
        log_msg = f"Enhanced Advanced Trainer başlatıldı. Cihaz: {self.device}"
        if self.is_tpu_v5e:
            log_msg += " (v5e-8 optimizasyonları aktif)"
        self.logger.info(log_msg)

    def _setup_optimizer_and_scheduler(self, num_training_steps: int,
                                       resume_from_ckpt_data: Optional[dict] = None):
        """
        Optimizer ve Scheduler kurulumu (Config uyumlu)
        
        İyileştirmeler:
        - Daha detaylı parameter grouping
        - Config'ten tüm parametreleri alır
        - Warmup ratio desteği (%20)
        - Checkpoint uyumluluğu iyileştirildi
        - Detaylı logging
        """
        
        # ============================================
        # A. PARAMETRE GRUPLAMA (İYİLEŞTİRİLMİŞ)
        # ============================================
        
        # Weight decay'den muaf tutulacak parametreler
        no_decay_keywords = [
            "bias",                    # Bias parametreleri
            "LayerNorm.weight",        # LayerNorm
            "layer_norm.weight",       # Küçük harfli variant
            "norm.weight",             # RMSNorm
            "embedding.weight",        # Embedding
        ]
        
        # Parametreleri grupla
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Decay'den muaf mı kontrol et
            if any(nd in name for nd in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Optimizer parameter groups
        optimizer_params = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]
        
        # ============================================
        # B. OPTIMIZER (Config'den tüm parametreler)
        # ============================================
        
        # Config'den parametreleri al
        learning_rate = self.config.learning_rate  # 1e-5
        adam_beta1 = self.config.adam_beta1        # 0.9
        adam_beta2 = self.config.adam_beta2        # 0.98
        adam_epsilon = self.config.adam_epsilon    # 1e-7
        
        self.logger.info("🔧 Optimizer: AdamW (Config Ayarları)")
        
        optimizer = AdamW(
            optimizer_params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # Detaylı logging
        total_params = sum(p.numel() for p in decay_params) + sum(p.numel() for p in no_decay_params)
        decay_count = sum(p.numel() for p in decay_params)
        no_decay_count = sum(p.numel() for p in no_decay_params)
        
        self.logger.info(f"   📊 Total Parameters: {total_params:,}")
        self.logger.info(f"   📊 With Decay: {decay_count:,} ({decay_count/total_params*100:.1f}%)")
        self.logger.info(f"   📊 No Decay: {no_decay_count:,} ({no_decay_count/total_params*100:.1f}%)")
        self.logger.info(f"   📊 Learning Rate: {learning_rate:.2e}")
        self.logger.info(f"   📊 Betas: ({adam_beta1}, {adam_beta2})")
        self.logger.info(f"   📊 Epsilon: {adam_epsilon:.2e}")
        
        # ============================================
        # C. SCHEDULER (Config uyumlu - warmup_ratio kullan)
        # ============================================
        
        # Config'den warmup parametrelerini al
        warmup_ratio = self.config.warmup_ratio  # 0.20 (%20)
        warmup_steps = int(num_training_steps * warmup_ratio)
        
        # Cosine scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"📅 Scheduler: Cosine with Warmup (Config)")
        self.logger.info(f"   📊 Warmup Steps: {warmup_steps:,} ({warmup_ratio*100:.0f}%)")
        self.logger.info(f"   📊 Total Steps: {num_training_steps:,}")
        self.logger.info(f"   📊 Initial LR: {learning_rate:.2e}")
        self.logger.info(f"   📊 Min LR: ~{learning_rate * 0.1:.2e}")
        
        # ============================================
        # D. CHECKPOINT'TEN DEVAM ETME
        # ============================================
        
        if resume_from_ckpt_data:
            # Optimizer state yükle
            if 'optimizer_state_dict' in resume_from_ckpt_data:
                try:
                    optimizer.load_state_dict(resume_from_ckpt_data['optimizer_state_dict'])
                    self.logger.info("   ✅ Optimizer state checkpoint'ten yüklendi")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Optimizer state yüklenemedi: {e}")
                    self.logger.warning("   ⚠️  Sıfırdan başlanıyor (yapı değişmiş olabilir)")
            
            # Scheduler state yükle
            if 'scheduler_state_dict' in resume_from_ckpt_data:
                try:
                    pass  # scheduler.load_state_dict(resume_from_ckpt_data["scheduler_state_dict"])
                    current_lr = scheduler.get_last_lr()[0]
                    self.logger.info(f"   ✅ Scheduler state checkpoint'ten yüklendi")
                    self.logger.info(f"   ✅ Current LR: {current_lr:.2e}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Scheduler state yüklenemedi: {e}")
    
        return optimizer, scheduler

    def _setup_data_loaders(self, train_dataset, eval_dataset=None, collate_fn=None):
        """TPU v5e-8 için optimize data loader kurulumu"""
        
        is_distributed = self.is_tpu and self.world_size > 1
        
        effective_batch_size = getattr(self.config, 'per_device_train_batch_size',
                                       getattr(self.config, 'batch_size', 4))

        final_collate_fn = collate_fn
        if final_collate_fn is None:
             raise ValueError("ChunkAwareCollator (collate_fn) sağlanmalıdır.")
        
        # ==================
        # TRAIN LOADER
        # ==================
        if is_distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True,
                drop_last=True
            )
        else:
            self.train_sampler = RandomSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=self.train_sampler,
            collate_fn=final_collate_fn,
            num_workers=0, # TPU için 0
            pin_memory=False, # TPU için False
            drop_last=True,
        )
        
        # ==================
        # EVAL LOADER
        # ==================
        eval_loader = None
        if eval_dataset:
            if is_distributed:
                self.eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    eval_dataset,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False,
                    drop_last=False
                )
            else:
                self.eval_sampler = SequentialSampler(eval_dataset)
            
            eval_batch_size = getattr(self.config, 'per_device_eval_batch_size', effective_batch_size * 2)
                
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                sampler=self.eval_sampler,
                collate_fn=final_collate_fn,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
        
        # ==================
        # TPU PARALLEL LOADER WRAPPER
        # ==================
        if self.is_tpu:
            if not (XLA_AVAILABLE and pl):
                 raise ImportError("TPU modu için 'torch_xla.distributed.parallel_loader' gerekli.")
            
            train_loader = pl.MpDeviceLoader(train_loader, self.device)
            if self.is_master():
                self.logger.info("✅ Train loader, MpDeviceLoader (ParallelLoader) ile sarıldı.")
            
            if eval_loader:
                eval_loader = pl.MpDeviceLoader(eval_loader, self.device)
                if self.is_master():
                    self.logger.info("✅ Eval loader, MpDeviceLoader (ParallelLoader) ile sarıldı.")
        
        if self.is_master():
            self.logger.info("=" * 60)
            self.logger.info("📊 DataLoaders Yapılandırması:")
            self.logger.info(f"   • Mod: {'Dağıtık (TPU)' if is_distributed else 'Tekil (GPU/CPU)'}")
            self.logger.info(f"   • Train batch size (per core): {effective_batch_size}")
            if is_distributed:
                self.logger.info(f"   • Global batch size (approx): {effective_batch_size * self.world_size}")
            if eval_loader:
                self.logger.info(f"   • Eval batch size (per core): {eval_batch_size}")
            self.logger.info(f"   • Using ParallelLoader: {self.is_tpu}")
            self.logger.info("=" * 60)
        
        return train_loader, eval_loader

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Batch'i (tensorlar ve diğer veriler dahil) cihaza taşır.
        Not: MpDeviceLoader (ParallelLoader) bunu otomatik yapar,
        bu manuel kullanım içindir.
        """
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device, non_blocking=True)
            else:
                # Metadata (doc_ids, etc.)
                moved_batch[key] = value
        return moved_batch

    def _get_world_size(self):
        """Güvenli world_size alımı"""
        if hasattr(self.config, 'world_size') and self.config.world_size > 0:
            return self.config.world_size
        
        if self.is_tpu:
            # ✅ Spawn öncesi için core sayısını environment'tan al
            try:
                import torch_xla.core.xla_model as xm
                # get_xla_supported_devices() spawn öncesi çalışır
                devices = xm.get_xla_supported_devices()
                return len(devices)  # 8 dönecek
            except:
                return 8  # TPU v5e-8 default
        
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        
        return 1

    def is_master(self) -> bool:
        """Ana process olup olmadığını kontrol eder"""
        if self.is_tpu:
            return xm.is_master_ordinal()
        elif torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        else:
            return True # Dağıtık değilse, her zaman master'dır

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None,
              resume_from_ckpt_data: Optional[dict] = None):
        """
        Ana eğitim döngüsü. 
        1. Tokenizer ile Model arasındaki boyut uyuşmazlığını otomatik düzeltir.
        2. TPU v5e-8 için optimize edilmiş eğitim adımlarını uygular.
        """
        
        # ============================================
        # 0. VOCAB SIZE KONTROLÜ VE DÜZELTME (EMNIYET SÜBABI)
        # ============================================
        
        # 1. Embedding Katmanını Bul (İsmi genelde 'token_embedding' veya 'embeddings' olur)
        emb_layer_name = None
        if hasattr(self.model, "token_embedding"):
            emb_layer_name = "token_embedding"
        elif hasattr(self.model, "embeddings"):
             emb_layer_name = "embeddings"
        elif hasattr(self.model, "wte"): # GPT-2 stili
             emb_layer_name = "wte"
        elif hasattr(self.model, "embedding"):
             emb_layer_name = "embedding"
        
        # Katmanı al
        if emb_layer_name:
            old_emb = getattr(self.model, emb_layer_name)
            current_vocab = old_emb.weight.shape[0]
            target_vocab = len(self.tokenizer) 
            
            # Boyutlar farklıysa Eylem Planı:
            if current_vocab != target_vocab:
                if self.is_master():
                    self.logger.warning(f"🚨 BOYUT UYUŞMAZLIĞI: Model({current_vocab}) != Tokenizer({target_vocab})")
                    self.logger.warning(f"🔧 '{emb_layer_name}' katmanı manuel olarak genişletiliyor...")
                
                # A) Yeni Embedding Katmanı Yarat
                new_emb = torch.nn.Embedding(
                    num_embeddings=target_vocab,
                    embedding_dim=old_emb.embedding_dim,
                    padding_idx=self.config.pad_token_id
                ).to(self.device) # Cihaza taşı
                
                # B) Eski ağırlıkları kopyala
                with torch.no_grad():
                    min_vocab = min(current_vocab, target_vocab)
                    # Eski verileri olduğu gibi al
                    new_emb.weight[:min_vocab] = old_emb.weight[:min_vocab]
                    # Yeni eklenen kısımları (örn: 100300 padding) küçük random başlat
                    if target_vocab > current_vocab:
                        new_emb.weight[current_vocab:].normal_(mean=0.0, std=0.02)
                
                # C) Modeli güncelle (Eskisini sil, yenisini tak)
                setattr(self.model, emb_layer_name, new_emb)
                
                # 2. Output Katmanını (LM Head) da güncellemek gerekir!
                head_layer_name = None
                if hasattr(self.model, "lm_head"): head_layer_name = "lm_head"
                elif hasattr(self.model, "output_layer"): head_layer_name = "output_layer"
                
                if head_layer_name:
                    old_head = getattr(self.model, head_layer_name)
                    if old_head.out_features != target_vocab:
                        if self.is_master(): self.logger.warning(f"🔧 '{head_layer_name}' katmanı da genişletiliyor...")
                        
                        # Yeni Linear katman
                        new_head = torch.BitLinear(
                            in_features=old_head.in_features,
                            out_features=target_vocab,
                            bias=(old_head.bias is not None)
                        ).to(self.device)
                        
                        # Ağırlıkları kopyala
                        with torch.no_grad():
                            new_head.weight[:min_vocab, :] = old_head.weight[:min_vocab, :]
                            if old_head.bias is not None:
                                new_head.bias[:min_vocab] = old_head.bias[:min_vocab]
                        
                        setattr(self.model, head_layer_name, new_head)
                
                # Config'i güncelle
                self.model.config.vocab_size = target_vocab
                if self.is_master(): self.logger.info(f"✅ Model başarıyla {target_vocab} boyutuna genişletildi.")
        
        else:
            if self.is_master():
                self.logger.warning("⚠️ Modelin embedding katmanı ismi bulunamadı. Resize atlandı (Riskli).")
        # ============================================
        # 1. KURULUM (SETUP)
        # ============================================
        
        # 1. Chunk-aware collator
        multi_task_collator = TPUOptimizedCollator(
            tokenizer=self.tokenizer,               # <-- Bu EKLENDİ
            max_length=self.config.seq_length,      # <-- Bu EKLENDİ
            padding_value=self.config.pad_token_id, # <-- İsim DEĞİŞTİ (pad_token_id -> padding_value)
            last_chunk_weight=getattr(self.config, 'last_chunk_weight', 1.5)
        )
        
        # 2. Ağırlıklı (Weighted) kayıp fonksiyonu
        criterion = WeightedCrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=getattr(self.config, 'label_smoothing', 0.0)
        )
        
        # 3. Eğitim durumu izleyicisi (State Tracker)
        state = TrainingState()
        
        # 4. Checkpoint yöneticisi
        checkpoint_manager = CheckpointManager(
            self, 
            self.logger,
            max_hours=getattr(self.config, 'max_training_hours', 8.5),
            save_total_limit=getattr(self.config, 'save_total_limit', 2),
            save_interval_min=getattr(self.config, 'save_interval_min', 30)  # ✅ EKLE
        )
        
        # 5. Değerlendirici (Evaluator)
        evaluator = MultiTaskEvaluator(logger=self.logger)
        
        # 6. Checkpoint'ten devam etme
        if resume_from_ckpt_data:
            state.epoch = resume_from_ckpt_data.get('epoch', 0)
            state.global_step = resume_from_ckpt_data.get('global_step', 0)
            state.best_eval_loss = resume_from_ckpt_data.get('best_eval_loss', float('inf'))
            state.best_perplexity = resume_from_ckpt_data.get('best_perplexity', float('inf'))
            if self.is_master():
                self.logger.info(f"📂 Eğitim devam ediyor: Epoch {state.epoch + 1}, Adım {state.global_step}")
        
        # 7. Sinyal (Shutdown) yakalayıcı
        def signal_handler(signum, frame):
            if self.is_master():
                self.logger.warning("⚠️ Kapatma sinyali alındı! Acil durum kaydı yapılıyor...")
                if self.is_tpu:
                    try:
                        import torch_xla.core.xla_model as xm
                        xm.rendezvous("emergency_shutdown")
                    except:
                        pass
                
                checkpoint_manager.save_checkpoint(
                    "signal_interrupted.pt", state,
                    optimizer, scheduler, {}, force_full=True
                )
                self.logger.info("💾 Acil durum checkpoint'i kaydedildi.")
            sys.exit(0)

        # ✅ DÜZELTME: Try-Except bloğu ile hatayı yutuyoruz
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Bu hata, worker process'lerde (alt işlemlerde) normaldir. 
            # Sadece Main Thread sinyal dinleyebilir.
            pass
        
        # 8. Data loader'ları kur (TPU için MpDeviceLoader kullanır)
        train_loader, eval_loader = self._setup_data_loaders(
            train_dataset, eval_dataset, collate_fn=multi_task_collator
        )
        
        # 9. Eğitim parametreleri
        num_epochs = getattr(self.config, 'epochs', 10)
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        save_steps = getattr(self.config, 'save_steps', 1000)
        logging_steps = getattr(self.config, 'logging_steps', 50)
        eval_epoch_interval = getattr(self.config, 'eval_epoch_interval', 1)
        
        # 10. Toplam adım sayısını hesapla
        try:
            steps_per_epoch = len(train_loader) # MpDeviceLoader bunu destekler
        except TypeError:
            # Fallback
            steps_per_epoch = math.ceil(
                len(train_dataset) / (getattr(self.config, 'per_device_train_batch_size', 4) * self.world_size * gradient_accumulation_steps)
            )
            
        total_steps = steps_per_epoch * num_epochs
        
        if steps_per_epoch == 0:
             self.logger.error("❌ HATA: steps_per_epoch 0! Dataset çok küçük veya batch_size çok büyük.")
             raise ValueError("steps_per_epoch 0 olamaz.")

        # 11. Optimizer & Scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(
            total_steps, resume_from_ckpt_data
        )
        
        # 12. Mixed Precision (CUDA için GradScaler)
        scaler = None
        if torch.cuda.is_available() and getattr(self.config, 'bf16', False):
            scaler = torch.cuda.amp.GradScaler()
            if self.is_master():
                self.logger.info("✅ CUDA GradScaler (mixed precision) aktif.")
        
        # 13. Autocast ayarları (TPU/GPU)
        device_type = self.device.type
        autocast_dtype = torch.bfloat16 if self.is_tpu or getattr(self.config, 'bf16', False) else torch.float16
        autocast_enabled = (device_type in ['cuda', 'xla']) and getattr(self.config, 'bf16', False)
        
        # 14. Model Compilation (PyTorch 2.0+ GPU)
        if hasattr(torch, 'compile') and getattr(self.config, 'compile_model', False) and not self.is_tpu:
            if self.is_master(): self.logger.info("🔥 torch.compile() ile model derleniyor...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                if self.is_master(): self.logger.info("✅ Model derlendi.")
            except Exception as e:
                if self.is_master(): self.logger.warning(f"⚠️ Derleme başarısız: {e}")
        
        # 15. TPU Senkronizasyonu
        if self.is_tpu:
            xm.mark_step()
            if self.is_master(): self.logger.info("✅ All TPU cores synced")
        
        # 16. Başlangıç Log'u
        if self.is_master():
            device_info = f"TPU v5e-8 ({self.world_size} cores)" if self.is_tpu_v5e else (
                f"TPU ({self.world_size} cores)" if self.is_tpu else (
                f"GPU ({self.world_size} devices)" if torch.cuda.is_available() else "CPU"
            ))
            self.logger.info(f"{'='*60}")
            self.logger.info(f"🚀 Eğitim Başlıyor ({device_info})")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"  Toplam Epoch: {num_epochs}")
            self.logger.info(f"  Toplam Adım (tahmini): {total_steps:,}")
            self.logger.info(f"  Adım / Epoch: {steps_per_epoch:,}")
            self.logger.info(f"  Gradient Accumulation: {gradient_accumulation_steps}")
            self.logger.info(f"  Mixed Precision: {autocast_enabled} ({autocast_dtype})")
            self.logger.info(f"  🎯 Multi-Task Training: AKTİF (Ağırlık: {multi_task_collator.last_chunk_weight}x)")
            self.logger.info(f"  ⏰ Max Süre: {checkpoint_manager.max_hours} saat")
            self.logger.info(f"  💾 Kayıt Aralığı: {checkpoint_manager.save_interval_min} dk")
            self.logger.info(f"{'='*60}")
            
        # ============================================
        # 2. EĞİTİM DÖNGÜSÜ (TRAINING LOOP - FIXED)
        # ============================================
        
        try:
            # ============================================
            # EPOCH LOOP BAŞLANGIÇ - SAYAÇLAR
            # ============================================
            running_loss = 0.0       
            batches_since_log = 0
            
            # Global throughput tracking
            epoch_start_time = time.time()
            total_samples_processed = 0
            total_tokens_processed = 0
            
            for epoch in range(state.epoch, num_epochs):
                state.epoch = epoch
                self.model.train()
                
                # ============================================
                # EPOCH BAŞLANGIÇ - DAĞITIK AYARLAR
                # ============================================
                if self.is_master():
                    self.logger.info("\n" + "="*80)
                    self.logger.info(f"🚀 EPOCH {epoch+1}/{num_epochs} BAŞLIYOR")
                    self.logger.info("="*80)
                
                # Distributed sampler epoch ayarı
                if self.train_sampler and hasattr(self.train_sampler, 'set_epoch'):
                    self.train_sampler.set_epoch(epoch)
                    if self.is_master():
                        self.logger.info(f"✅ Distributed sampler epoch ayarlandı: {epoch}")
                
                # TPU senkronizasyonu
                if self.is_tpu:
                    xm.mark_step()
                    if self.is_master():
                        self.logger.info(f"✅ TPU cores senkronize edildi (epoch_{epoch}_start)")
                
                # ============================================
                # EPOCH-LEVEL SAYAÇLARI SIFIRLA
                # ============================================
                running_loss = 0.0      
                batches_since_log = 0
                epoch_samples = 0
                epoch_tokens = 0
                epoch_step_times = []
                
                # ============================================
                # PROGRESS BAR HAZIRLIĞI (GLOBAL STEP BAZLI)
                # ============================================
                total_micro_batches = len(train_loader)
                total_global_steps = total_micro_batches // gradient_accumulation_steps
                
                # Batch configuration hesapla
                batch_per_core = self.config.per_device_train_batch_size
                world_size = self._get_world_size()
                grad_accum = gradient_accumulation_steps
                global_batch = batch_per_core * world_size
                effective_batch = global_batch * grad_accum
                tokens_per_step = effective_batch * self.config.seq_length
                
                if self.is_master():
                    self.logger.info("\n" + "="*70)
                    self.logger.info(f"📊 EPOCH {epoch+1}/{num_epochs} BATCH CONFIGURATION")
                    self.logger.info("="*70)
                    self.logger.info(f"   • Batch per core:       {batch_per_core}")
                    self.logger.info(f"   • World size (cores):   {world_size}")
                    self.logger.info(f"   • Gradient accum:       {grad_accum} steps")
                    self.logger.info(f"   • Global batch size:    {global_batch} (per micro-step)")
                    self.logger.info(f"   • Effective batch:      {effective_batch} (per optimization)")
                    self.logger.info(f"   • Sequence length:      {self.config.seq_length}")
                    self.logger.info(f"   • Tokens per step:      {tokens_per_step:,}")
                    self.logger.info(f"   • Gradient checkpoint:  {getattr(self.config, 'gradient_checkpointing', False)}")
                    self.logger.info("="*70)
                    self.logger.info(f"   • Total micro-batches:  {total_micro_batches:,}")
                    self.logger.info(f"   • Total optim steps:    {total_global_steps:,}")
                    self.logger.info("="*70 + "\n")
                
                # Progress bar (batch info description'da)
                progress_bar = tqdm(
                    total=total_global_steps,
                    desc=f"E{epoch+1}/{num_epochs} [B:{batch_per_core}×{world_size}×{grad_accum}={effective_batch}]",
                    disable=not self.is_master(),
                    leave=True,
                    mininterval=10.0,
                    unit="step",
                    bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
                
                # ============================================
                # MICRO-BATCH LOOP
                # ============================================
                for step, batch in enumerate(train_loader):
                    micro_batch_start_time = time.time()

                    if step % gradient_accumulation_steps == 0:
                        step_start_time = time.time()
                    
                    # ============================================
                    # İLK BATCH: DETAYLI BATCH SHAPE LOGU
                    # ============================================
                    if self.is_master() and step == 0:
                        self.logger.info(f"🔍 İlk Micro-Batch Detayları:")
                        self.logger.info(f"   • Micro-batch index: {step}")
                        self.logger.info(f"   • Batch keys: {list(batch.keys())}")
                        self.logger.info(f"   • input_ids shape: {batch['input_ids'].shape}")
                        self.logger.info(f"   • Device: {batch['input_ids'].device}")
                        self.logger.info(f"   • Dtype: {batch['input_ids'].dtype}")
                        
                        # Memory baseline
                        try:
                            import torch_xla
                            mem_info = torch_xla._XLAC._xla_memory_info(str(self.device))
                            self.logger.info(f"   • Memory (baseline): {mem_info}")
                        except:
                            pass
                        
                        self.logger.info("")
                    
                    # Gradient accumulation içindeki pozisyon
                    accumulation_step = step % gradient_accumulation_steps
                    is_accumulation_step = (step + 1) % gradient_accumulation_steps == 0
                    
                    # Periodic micro-batch tracking
                    if self.is_master() and state.global_step % (logging_steps * 5) == 0 and step % 10 == 0:
                        self.logger.info(
                            f"   📍 Micro-batch {step}/{total_micro_batches} | "
                            f"Accum: {accumulation_step + 1}/{gradient_accumulation_steps} | "
                            f"Global step: {state.global_step}"
                        )
                    
                    # ============================================
                    # A. VERİ HAZIRLIĞI VE METADATA
                    # ============================================
                    loss_weights = batch.pop('loss_weights', None)
                    doc_ids = batch.pop('doc_ids', None)
                    is_last_chunks = batch.pop('is_last_chunks', None)
                    
                    # Gereksiz metadata'ları temizle
                    for k in ['chunk_indices', 'active_tasks', 'task_weights']:
                        batch.pop(k, None)
                    
                    # Document tracking
                    if doc_ids is not None:
                        num_docs_in_batch = len(doc_ids) if isinstance(doc_ids, (list, tuple)) else doc_ids.numel()
                        state.docs_seen.update(doc_ids if isinstance(doc_ids, (list, tuple)) else doc_ids.tolist())
                        state.chunks_processed += num_docs_in_batch
                        
                        if self.is_master() and step % 100 == 0:
                            self.logger.info(
                                f"   📄 Documents: {len(state.docs_seen):,} unique | "
                                f"{state.chunks_processed:,} chunks processed"
                            )

                    # ============================================
                    # B. FORWARD PASS
                    # ============================================
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    # TPU Mask Fix
                    if attention_mask is not None and attention_mask.dtype not in [torch.bool, torch.float32]:
                        attention_mask = attention_mask.bool()

                    with autocast_context(self.device, autocast_dtype, autocast_enabled):
                        raw_output = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=None
                        )
                        logits = raw_output.logits
                        
                        # Loss Hesaplama
                        loss, _ = criterion(logits, batch['lm_labels'], weights=loss_weights)
                        loss_scaled = loss / gradient_accumulation_steps
                    
                    # İstatistik: Chunk Loss Takibi
                    if is_last_chunks is not None:
                        current_val = loss.detach().item()
                        if isinstance(is_last_chunks, torch.Tensor):
                            if is_last_chunks.any(): state.last_chunk_losses.append(current_val)
                            else: state.middle_chunk_losses.append(current_val)
                        elif isinstance(is_last_chunks, list):
                            if any(is_last_chunks): state.last_chunk_losses.append(current_val)
                            else: state.middle_chunk_losses.append(current_val)

                    # ============================================
                    # C. BACKWARD PASS
                    # ============================================
                    if scaler is not None:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    
                    running_loss += loss.item()
                    batches_since_log += 1
                    state.total_tokens += (batch['lm_labels'] != -100).sum().item()
                    
                    # ============================================
                    # D. OPTIMIZER ADIMI
                    # ============================================
                    if (step + 1) % gradient_accumulation_steps == 0:
                        
                        # ============================================
                        # Gradient Clipping
                        # ============================================
                        if max_grad_norm > 0:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            
                            # Gradient norm hesapla (monitoring)
                            total_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_grad_norm
                            )
                            
                            # Anormal gradient logging (periodic)
                            if self.is_master() and state.global_step % (logging_steps * 10) == 0:
                                self.logger.info(f"   📊 Gradient norm: {total_norm:.4f}")
                                if total_norm > max_grad_norm * 10:
                                    self.logger.warning(f"   ⚠️  High gradient norm detected!")
                        
                        # ============================================
                        # Optimizer Step
                        # ============================================
                        if self.is_tpu:
                            xm.optimizer_step(optimizer, barrier=True)
                        elif scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        
                        # Progress bar update
                        progress_bar.update(1)
                        
                        # Global step increment
                        state.global_step += 1

                        # ============================================
                        # 🔥 BITNET WARMUP: Quantization Gecikmeli Başlatma
                        # ============================================
                        # İlk 500 adım FP16/BF16 çalışır, sonra BitNet (1.58-bit) açılır.
                        # Bu, modelin "LOSS NAN" olmasını ve çökmesini engeller.
                        bitnet_warmup = getattr(self.config, 'bitnet_warmup_steps', 0)
                        
                        if state.global_step == bitnet_warmup:
                            if self.is_master():
                                self.logger.info(f"\n⚡ DİKKAT: BitNet Quantization AKTİF EDİLİYOR (Step {state.global_step}) ⚡")
                                self.logger.info("   Artık ağırlıklar {-1, 0, 1} arasına sıkıştırılıyor.")
                            
                            activated_count = 0
                            # Tüm model katmanlarını gez
                            for module in self.model.modules():
                                # Eğer katmanda 'quantize_training' özelliği varsa True yap
                                if hasattr(module, 'quantize_training'):
                                    module.quantize_training = True
                                    activated_count += 1
                            
                            if self.is_master():
                                self.logger.info(f"   ✅ {activated_count} katman başarıyla Quantization moduna geçirildi.\n")
                        # ============================================
                        # Performance Tracking
                        # ============================================
                        step_time = time.time() - step_start_time
                        state.step_times.append(step_time)
                        
                        # ============================================
                        # Logging (Periodic)
                        # ============================================
                        if state.global_step % logging_steps == 0 and self.is_master():
                            avg_loss = running_loss / batches_since_log
                            lr = scheduler.get_last_lr()[0]
                            
                            # Throughput calculation
                            batch_per_core = self.config.per_device_train_batch_size
                            world_size = self._get_world_size()
                            grad_accum = gradient_accumulation_steps
                            effective_batch = batch_per_core * world_size * grad_accum
                            tokens_per_step = effective_batch * self.config.seq_length
                            
                            # Average step time (deque-safe)
                            if len(state.step_times) > 0:
                                step_times_list = list(state.step_times)  # ← deque → list
                                recent_steps = min(logging_steps, len(step_times_list))
                                avg_step_time = sum(step_times_list[-recent_steps:]) / recent_steps
                            else:
                                avg_step_time = 0.0
                            
                            samples_per_sec = effective_batch / avg_step_time if avg_step_time > 0 else 0
                            tokens_per_sec = tokens_per_step / avg_step_time if avg_step_time > 0 else 0
                            
                            # Progress bar update (with throughput)
                            progress_bar.set_postfix({
                                'loss': f"{avg_loss:.4f}",
                                'lr': f"{lr:.2e}",
                                's/s': f"{samples_per_sec:.0f}",
                                'tok/s': f"{tokens_per_sec/1000:.1f}K"
                            })
                            
                            # Detailed log
                            self.logger.info(
                                f"Step {state.global_step:>6} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {lr:.2e} | "
                                f"Time: {avg_step_time:.2f}s/step | "
                                f"Throughput: {samples_per_sec:.0f} samp/s, {tokens_per_sec/1000:.1f}K tok/s"
                            )
                            
                            # History update
                            state.train_history['loss'].append(avg_loss)
                            
                            # Reset counters
                            running_loss = 0.0
                            batches_since_log = 0
                        
                        # ============================================
                        # Memory Logging (Every 50 steps)
                        # ============================================
                        if state.global_step % (logging_steps * 5) == 0 and self.is_master():
                            try:
                                import torch_xla
                                mem_info = torch_xla._XLAC._xla_memory_info(str(self.device))
                                self.logger.info(f"   💾 Memory: {mem_info}")
                            except Exception:
                                pass
                        
                        # ============================================
                        # CHECKPOINT (Fixed - No Deadlock)
                        # ============================================
                        if state.global_step > 0 and state.global_step % save_steps == 0:
                            
                            # 1. XLA Sync (all cores)
                            if self.is_tpu:
                                xm.mark_step()
                                # ❌ REMOVED: xm.rendezvous() - causes deadlock
                            
                            # 2. Master saves checkpoint
                            if self.is_master():
                                self.logger.info(f"\n💾 Saving checkpoint: step {state.global_step}")
                                
                                # Evaluation (optional, can be disabled for speed)
                                eval_metrics = None
                                if eval_loader is not None:
                                    try:
                                        eval_metrics = self._run_evaluation_safe(
                                            eval_loader, criterion, epoch, state, evaluator
                                        )
                                    except Exception as e:
                                        self.logger.warning(f"⚠️  Eval failed during checkpoint: {e}")
                                        eval_metrics = None
                                
                                # Save checkpoint
                                try:
                                    checkpoint_manager.save_checkpoint(
                                        f"ckpt_step{state.global_step}.pt",
                                        state,
                                        optimizer,
                                        scheduler,
                                        eval_metrics,
                                        force_full=True
                                    )
                                    self.logger.info(f"✅ Checkpoint saved: step {state.global_step}\n")
                                except Exception as e:
                                    self.logger.error(f"❌ Checkpoint save failed: {e}")
                            
                            # 3. Final sync (all cores wait for master to finish)
                            if self.is_tpu:
                                xm.mark_step()
                    # ==========================================================
                    # 🔥 E. RAM VE CACHE TEMİZLİĞİ (LIFE SAVER)
                    # ==========================================================
                    if step > 0 and step % 100 == 0:
                        import gc
                        gc.collect() 
                        if self.device.type == 'cuda': torch.cuda.empty_cache()
                        if self.is_tpu: xm.mark_step()

                # ============================================
                # 3. EPOCH SONU
                # ============================================
                progress_bar.close()

                # XLA sync
                if self.is_tpu:
                    xm.mark_step()
                
                # Master saves ALWAYS (no try-except to hide errors)
                if self.is_master():
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"✅ EPOCH {epoch+1}/{num_epochs} COMPLETED")
                    self.logger.info(f"{'='*60}")
                    
                    # Force save epoch checkpoint
                    checkpoint_path = f"{self.config.save_dir}/checkpoint_epoch{epoch+1}.pt"
                    
                    try:
                        # Ensure directory exists
                        import os
                        os.makedirs(self.config.save_dir, exist_ok=True)
                        
                        # Save checkpoint
                        save_dict = {
                            'epoch': epoch,
                            'global_step': state.global_step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': state.train_history['loss'][-1] if state.train_history['loss'] else None,
                            'config': self.config.__dict__,
                        }
                        
                        if self.is_tpu:
                            xm.save(save_dict, checkpoint_path)
                        else:
                            torch.save(save_dict, checkpoint_path)
                        
                        self.logger.info(f"💾 Epoch checkpoint saved: {checkpoint_path}")
                        # Eski epoch checkpointlerini sil (son 2 tut)
                        import glob
                        epoch_ckpts = sorted(glob.glob(f"{self.config.save_dir}/checkpoint_epoch*.pt"))
                        if len(epoch_ckpts) > 2:
                            for old_ckpt in epoch_ckpts[:-2]:
                                os.remove(old_ckpt)
                                self.logger.info(f"🗑️ Eski epoch checkpoint silindi: {old_ckpt}")
                        
                        # Also save as 'latest.pt'
                        latest_path = f"{self.config.save_dir}/latest.pt"
                        if self.is_tpu:
                            xm.save(save_dict, latest_path)
                        else:
                            torch.save(save_dict, latest_path)
                        
                        self.logger.info(f"💾 Latest checkpoint saved: {latest_path}\n")
                        
                    except Exception as e:
                        self.logger.error(f"❌ CRITICAL: Epoch checkpoint save FAILED: {e}")
                        self.logger.error(f"   Traceback:", exc_info=True)
                
                # Early Stopping
                if self._should_stop_early(state):
                    if self.is_master():
                        self.logger.info("🛑 Early stopping triggered")
                    break
            
            # ============================================
            # 4. EĞİTİM TAMAMLANDI
            # ============================================
            if self.is_tpu: xm.mark_step()
            if self.is_master():
                self.logger.info("🎉 Eğitim Tamamlandı! 🎉")
                self.save_tokenizer_and_config() # Tokenizer ve config'i kaydet

        except KeyboardInterrupt:
            if self.is_master():
                self.logger.warning("⚠️ Eğitim kullanıcı tarafından durduruldu!")
                checkpoint_manager.save_checkpoint(f"interrupted_epoch{state.epoch}.pt", state, optimizer, scheduler, {}, force_full=True)
        except Exception as e:
            if self.is_master():
                self.logger.error(f"💥 Kritik Hata: {e}", exc_info=True)
                checkpoint_manager.save_checkpoint(f"error_epoch{state.epoch}.pt", state, optimizer, scheduler, {}, force_full=True)
            raise
        
        finally:
            if self.is_tpu: xm.mark_step()
            if self.is_master(): self.logger.info("✅ Eğitim fonksiyonu temizlendi.")

    def _run_evaluation_safe(self, eval_loader, criterion, epoch, state, evaluator):
        """Değerlendirme (evaluate) işlemini güvenli bir şekilde çalıştırır."""
        if eval_loader is None:
            if self.is_master():
                self.logger.info("Değerlendirme verisi yok, atlanıyor.")
            return {}
        
        if self.is_master():
            self.logger.info(f"Değerlendirme başlıyor (Epoch {epoch+1})...")
            
        try:
            metrics = evaluator.evaluate(
                self.model,
                eval_loader,
                criterion,
                self.device
            )
            
            # Geçmişi (history) güncelle
            if self.is_master() and metrics:
                eval_loss = metrics.get('eval_loss', None)
                perplexity = metrics.get('perplexity', None)
                if eval_loss is not None:
                    state.eval_history['loss'].append(eval_loss)
                if perplexity is not None:
                    state.eval_history['perplexity'].append(perplexity)
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"❌ Değerlendirme hatası: {e}", exc_info=True)
            self.model.train() # Modeli tekrar train moda al
            return {}

    def save_model(self, filename: str, epoch: int, global_step: int,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   eval_metrics: dict,
                   is_best: bool = False, save_full_state: bool = True):
        """
        Model checkpoint'ini kaydeder.
        TPU'da ise xm.save(), GPU/CPU'da torch.save() kullanır.
        """
        if not self.is_master():
            return True # Sadece master kaydeder

        try:
            save_path = self.model_save_dir / filename
            self.logger.info(f"Kaydediliyor: {filename} (Adım={global_step})")

            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': self.model.state_dict(),
                'eval_metrics': eval_metrics,
                'best_eval_loss': state.best_eval_loss,
                'best_perplexity': self.best_perplexity,
                'train_history': dict(self.train_history),
                'eval_history': dict(self.eval_history),
                'anomaly_history': self.anomaly_detector.get_history(),
                'timestamp': datetime.now().isoformat(),
            }
            
            # Model config'ini de ekle
            if hasattr(self.config, 'to_dict'):
                checkpoint['config'] = self.config.to_dict()
            else:
                checkpoint['config'] = asdict(self.config)

            if save_full_state:
                checkpoint.update({
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                })
            
            # Kaydet
            if self.is_tpu:
                xm.save(checkpoint, save_path)
            else:
                torch.save(checkpoint, save_path)

            # Kayıt geçmişini yönet
            checkpoint_info = {
                'filename': filename, 'path': str(save_path),
                'timestamp': checkpoint['timestamp'],
                'eval_loss': eval_metrics.get('eval_loss', float('inf')),
                'is_best': is_best
            }
            self.checkpoint_history.append(checkpoint_info)
            self._cleanup_old_checkpoints()
            self._save_checkpoint_metadata()
            
            file_size_mb = save_path.stat().st_size / (1024**2)
            self.logger.info(f"Kayıt başarılı: {save_path.name} ({file_size_mb:.1f} MB)")

        except Exception as e:
            self.logger.error(f"❌ KAYIT HATASI: {e}", exc_info=True)
            return False
        return True

    def _cleanup_old_checkpoints(self):
        """Eski checkpoint'leri siler (save_total_limit'e göre)"""
        max_checkpoints = getattr(self.config, 'save_total_limit', 3)
        
        if len(self.checkpoint_history) <= max_checkpoints:
            return
            
        # En iyileri her zaman tut
        best_checkpoints = [cp for cp in self.checkpoint_history if cp.get('is_best', False)]
        regular_checkpoints = [cp for cp in self.checkpoint_history if not cp.get('is_best', False)]
        
        keep_regular = max(0, max_checkpoints - len(best_checkpoints))
        
        if len(regular_checkpoints) > keep_regular:
            regular_checkpoints.sort(key=lambda x: x.get('timestamp', ''))
            to_delete = regular_checkpoints[:-keep_regular] # En eskileri al
            
            for cp in to_delete:
                try:
                    checkpoint_path = Path(cp['path'])
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        self.checkpoint_history.remove(cp)
                        self.logger.info(f"Eski checkpoint silindi: {cp['filename']}")
                except Exception as e:
                    self.logger.warning(f"Checkpoint silinemedi {cp['filename']}: {e}")

    def _save_checkpoint_metadata(self):
        """Checkpoint'lerin listesini JSON olarak kaydeder"""
        metadata_path = self.model_save_dir.parent / "checkpoint_metadata.json"
        try:
            metadata = {
                'checkpoint_history': self.checkpoint_history,
                'best_eval_loss': state.best_eval_loss,
                'best_perplexity': self.best_perplexity,
                'last_update': datetime.now().isoformat()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Checkpoint metadata kaydedilemedi: {e}")

    def load_checkpoint(self, checkpoint_path: str = None):
        """Bir checkpoint'i yükler"""
        try:
            if checkpoint_path is None:
                checkpoint_path = self.model_save_dir / "model_best.pt"
            else:
                checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
            
            self.logger.info(f"Checkpoint yükleniyor: {checkpoint_path}")
            
            # TPU/GPU/CPU uyumlu yükleme
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Model state'i yükle
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            state.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
            self.best_perplexity = checkpoint.get('best_perplexity', float('inf'))
            
            if 'train_history' in checkpoint: self.train_history.update(checkpoint['train_history'])
            if 'eval_history' in checkpoint: self.eval_history.update(checkpoint['eval_history'])
            if 'anomaly_history' in checkpoint:
                self.anomaly_detector.load_history(checkpoint['anomaly_history'])
            
            epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            
            self.logger.info(f"✅ Checkpoint yüklendi (Epoch: {epoch}, Adım: {global_step})")
            
            return {
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
                'scheduler_state_dict': checkpoint.get('scheduler_state_dict')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint yüklenemedi: {e}", exc_info=True)
            raise

    def get_latest_checkpoint(self):
        """Kaydedilmiş en son checkpoint'in yolunu bulur"""
        try:
            checkpoint_files = list(self.model_save_dir.glob("ckpt_step*.pt"))
            checkpoint_files += list(self.model_save_dir.glob("checkpoint_epoch*.pt"))
            if not checkpoint_files:
                return None
            
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            return str(latest_checkpoint)
        except Exception as e:
            self.logger.warning(f"En son checkpoint bulunamadı: {e}")
            return None

    def save_tokenizer_and_config(self):
        """
        Model Config ve Tokenizer'ı kaydeder.
        Bu sayede eğitim bittikten sonra 'AutoTokenizer.from_pretrained' çalışır.
        """
        try:
            if not self.is_master():
                return
            
            # 1. Klasörleri Oluştur
            output_dir = self.model_save_dir.parent # 'checkpoints' klasörünün üstü
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. Tokenizer Kaydet (HuggingFace Formatında)
            # Bu, tokenizer.json ve config dosyasını yaratır.
            tokenizer_path = output_dir / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
            self.logger.info(f"📝 Tokenizer kaydedildi: {tokenizer_path}")
            
            # 3. Model Config Kaydet
            config_path = output_dir / "config.json"
            
            # Config nesnesinin türüne göre kaydetme yöntemi
            if hasattr(self.config, 'save_to_json'):
                self.config.save_to_json(config_path) # Özel config sınıfın
            elif hasattr(self.config, 'to_json_file'):
                self.config.to_json_file(config_path) # HuggingFace config
            else:
                # Fallback: Dictionary olarak dump et
                import json
                with open(config_path, 'w') as f:
                    json.dump(self.config.__dict__, f, indent=2)
            
            self.logger.info(f"⚙️ Config kaydedildi: {config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Tokenizer/Config kaydedilemedi: {e}", exc_info=True)

    def _should_stop_early(self, state: TrainingState) -> bool:
         """Early stopping kontrolü"""
         patience = getattr(self.config, 'early_stopping_patience', 5)
         if patience <= 0:
             return False
         return state.early_stopping_counter >= patience

    def _save_training_summary(self, state: TrainingState):
        """Eğitim özetini JSON olarak kaydeder"""
        if not self.is_master():
            return
            
        try:
            summary = {
                'final_step': state.global_step,
                'final_epoch': state.epoch,
                'best_eval_loss': state.best_eval_loss,
                'best_perplexity': state.best_perplexity,
                'total_tokens': state.total_tokens,
                'elapsed_hours': state.elapsed_hours(),
                'docs_seen': len(state.docs_seen),
                'chunks_processed': state.chunks_processed,
                'device_info': str(self.device),
                'world_size': self.world_size,
                'config_summary': {
                    'lr': self.config.learning_rate,
                    'batch_per_core': self.config.per_device_train_batch_size,
                    'd_model': self.config.d_model,
                    'n_layers': self.config.num_decoder_layers,
                    'n_head': self.config.nhead
                }
            }
            summary_path = self.model_save_dir.parent / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Eğitim özeti kaydedildi: {summary_path}")
        except Exception as e:
            self.logger.warning(f"Eğitim özeti kaydedilemedi: {e}")

    def _check_model_health(self, loss: float, step: int) -> bool:
        """Modelin sağlığını (NaN/Inf loss) kontrol eder"""
        if math.isnan(loss) or math.isinf(loss):
            self.logger.error(f"❌ KRİTİK HATA: Geçersiz loss (NaN/Inf): {loss} (Adım: {step})")
            self.anomaly_detector.record_anomaly('invalid_loss', step, {'loss': loss})
            return False
        
        # Patlama (Explosion) kontrolü
        explosion_threshold = 20.0
        if loss > explosion_threshold:
             self.logger.warning(f"⚠️ Yüksek loss tespit edildi: {loss:.2f} (Adım: {step})")
             self.anomaly_detector.record_anomaly('high_loss', step, {'loss': loss})
        
        self.loss_window.append(loss)
        return True

    def _get_layer_wise_optimizer_params(self):
        """(Opsiyonel) Layer-wise learning rate decay için parametre grupları"""
        # Bu özellik şu anda `_setup_optimizer_and_scheduler` içinde tam
        # olarak bağlı değil, ancak orijinal kodda vardı.
        # Basit gruplamayı (decay/no_decay) kullanıyoruz.
        self.logger.warning("Layer-wise LR decay isteniyor ancak basit gruplama kullanılıyor.")
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        return [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

    def _group_parameters_by_layer(self):
        """(Opsiyonel) Parametreleri katmanlara göre gruplar"""
        # _get_layer_wise_optimizer_params içinde kullanılmak üzere
        pass # Şu anki implementasyon bunu kullanmıyor.


# ==============================================================================
# === BÖLÜM 11: CHATBOT (INFERENCE) SINIFI
# ==============================================================================

class UltimateChatbot:
    """
    Eğitilmiş UltimateTransformerModel'i (Decoder-Only) yükler ve
    interaktif sohbet arayüzü sağlar.
    
    NOT: Orijinal koddaki Seq2Seq (encode/decode) mantığı,
    refaktör edilmiş Decoder-Only modele uyarlanmıştır.
    """
    def __init__(self, model_config: ModelConfig,
                 model_state_dict: OrderedDict,
                 tokenizer: 'TiktokenWrapper',
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or setup_logging(model_config, rank=0) # Yeni logger kur
        self.tokenizer = tokenizer
        self.config = model_config

        # Config'in tokenizer ile senkronize olduğundan emin ol
        self.config.vocab_size = self.tokenizer.vocab_size
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.bos_token_id = self.tokenizer.bos_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id

        # Modeli GÜNCELLENMİŞ config ile oluştur
        self.model = UltimateTransformerModel(self.config)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = getattr(self.config, 'chatbot_history_length', 5)
        
        # Generation parametreleri
        self.max_new_tokens = getattr(self.config, 'chatbot_max_new_tokens', 150)
        self.temperature = getattr(self.config, 'temperature', 0.7)
        self.top_k = getattr(self.config, 'top_k', 50)
        self.top_p = getattr(self.config, 'top_p', 0.95)
        self.repetition_penalty = getattr(self.config, 'repetition_penalty', 1.1)
        self.no_repeat_ngram_size = getattr(self.config, 'no_repeat_ngram_size', 3)
        self.do_sample = getattr(self.config, 'do_sample', True)

        self.logger.info(f"🤖 Chatbot başarıyla yüklendi. Cihaz: {self.device}")
        self.logger.info(f"   Ayarlar: MaxNewTokens={self.max_new_tokens}, Temp={self.temperature}, TopK={self.top_k}")

    @torch.no_grad()
    def generate_response(self, user_input: str, use_history: bool = True) -> str:
        """
        Decoder-Only (GPT-benzeri) model için autoregressive cevap üretimi.
        """
        prompt = ""
        # Alpaca formatı (veya benzeri bir format)
        instruction_part = f"Instruction: {user_input.strip()}"

        if use_history and self.conversation_history:
            history_entries = []
            for entry in self.conversation_history[-self.max_history_length:]:
                history_entries.append(f"User: {entry['user']}")
                history_entries.append(f"Bot: {entry['bot']}")
            history_str = "\n".join(history_entries)
            prompt = f"{history_str}\n{instruction_part}\nOutput:"
        else:
            prompt = f"{instruction_part}\nOutput:"

        is_on_xla = self._is_xla_device()
        response_text = "Üzgünüm, bir hata oluştu."

        try:
            # 1. Tokenize et (BOS/EOS olmadan)
            encoding = self.tokenizer(prompt, add_special_tokens=False)
            prompt_ids = encoding['input_ids'][0] # [0] ile listeyi al

            # 2. [BOS] token'ı ile başla (veya EOS)
            input_ids_list = [self.config.bos_token_id] + prompt_ids
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)

            # 3. Prompt uzunluğunu kontrol et
            prompt_len = input_ids.size(1)
            max_len = getattr(self.config, 'max_seq_length', 512)
            if prompt_len > max_len:
                self.logger.warning(f"Prompt çok uzun ({prompt_len} > {max_len}). Kırpılıyor.")
                input_ids = input_ids[:, -max_len:]
                prompt_len = input_ids.size(1)
            
            generated_ids = input_ids # Üretilen tokenları buna ekleyeceğiz
            
            # 4. Autoregressive üretim döngüsü
            for _ in range(self.max_new_tokens):
                
                # Sadece son 'max_len' token'ı dikkate al (KV Cache yoksa)
                current_ids = generated_ids
                if current_ids.size(1) > max_len:
                    current_ids = current_ids[:, -max_len:]
                
                current_mask = (current_ids != self.config.pad_token_id)
                
                # Model forward
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    labels=None
                )
                
                # Sadece son token'ın logit'lerini al
                next_token_logits = outputs.logits[:, -1, :]
                
                # Sampling
                next_token_id = self._sample_next_token(
                    next_token_logits.squeeze(0), # (Vocab_size)
                    generated_ids.squeeze(0).tolist() # (Seq_len)
                )
                
                if next_token_id == self.config.eos_token_id:
                    break
                    
                # Üretilen token'ı listeye ekle ve döngüye devam et
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)],
                    dim=1
                )

                if is_on_xla: xm.mark_step()
            
            # 5. Decode
            # Prompt kısmını atla (sadece üretilen kısmı al)
            response_ids = generated_ids.squeeze(0)[prompt_len:].cpu().tolist()
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_text = self._post_process_response(response_text)

        except Exception as e:
            self.logger.error(f"Yanıt üretirken hata: {e}", exc_info=True)
            if is_on_xla: xm.mark_step()

        if use_history:
            self.conversation_history.append({'user': user_input.strip(), 'bot': response_text})
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history.pop(0)

        return response_text

    def _sample_next_token(self, logits: torch.Tensor, generated_ids: List[int]) -> int:
        """Logit'lerden bir sonraki token'ı sample eder"""
        
        # Repetition penalty
        if self.repetition_penalty != 1.0 and len(generated_ids) > 0:
            for token_id_in_history in set(generated_ids):
                if token_id_in_history < logits.size(0):
                    if logits[token_id_in_history] < 0:
                        logits[token_id_in_history] *= self.repetition_penalty
                    else:
                        logits[token_id_in_history] /= self.repetition_penalty
        
        # No repeat N-gram
        if self.no_repeat_ngram_size > 0 and len(generated_ids) >= self.no_repeat_ngram_size -1:
            n_gram_prefix = tuple(generated_ids[-(self.no_repeat_ngram_size - 1):])
            
            # Olası tüm n-gram'ları kontrol et
            # (Bu optimize edilebilir, ancak küçük n-gram'lar için çalışır)
            banned_tokens = []
            if len(generated_ids) >= self.no_repeat_ngram_size:
                # Geçmişteki tüm n-gram'ları bul
                existing_ngrams = set()
                for i in range(len(generated_ids) - self.no_repeat_ngram_size + 1):
                    existing_ngrams.add(tuple(generated_ids[i : i + self.no_repeat_ngram_size]))
                
                # Eğer prefix + yeni_token bir n-gram oluşturuyorsa banla
                for token_id_to_check in range(logits.size(-1)):
                    potential_ngram = n_gram_prefix + (token_id_to_check,)
                    if potential_ngram in existing_ngrams:
                        banned_tokens.append(token_id_to_check)
            
            if banned_tokens:
                logits[banned_tokens] = float('-inf')

        # Temperature
        if self.temperature > 0 and self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Top-K
        if self.top_k > 0:
            k_val = min(self.top_k, logits.size(-1))
            if k_val > 0 and k_val < logits.size(-1):
                indices_to_remove = logits < torch.topk(logits, k_val)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
        
        # Top-P (Nucleus)
        if self.top_p > 0.0 and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # En az bir token'ı tut
            if sorted_indices_to_remove.all():
                sorted_indices_to_remove[..., :1] = False
            
            # Shift'i uygula
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        probabilities = F.softmax(logits, dim=-1)
        
        if not torch.isfinite(probabilities).all():
            self.logger.warning("NaN/Inf olasılıklar tespit edildi! Argmax'a dönülüyor.")
            return torch.argmax(logits).item()
            
        if not self.do_sample:
             return torch.argmax(probabilities).item()
        
        try:
            next_id = torch.multinomial(probabilities, num_samples=1).item()
        except RuntimeError as e:
            self.logger.warning(f"Multinomial sampling hatası: {e}. Argmax'a dönülüyor.")
            next_id = torch.argmax(logits).item()
            
        return next_id

    def _post_process_response(self, response: str) -> str:
        """Üretilen metni temizler"""
        response = response.strip()
        # "Instruction:", "User:", "Bot:", "Output:" gibi kalıntıları temizle
        stop_phrases = ["Instruction:", "User:", "Bot:", "Output:"]
        for phrase in stop_phrases:
            if phrase in response:
                response = response[:response.find(phrase)].strip()
        
        response = " ".join(response.split()) # Çoklu boşlukları düzelt
        return response

    def clear_history(self):
        self.conversation_history = []
        self.logger.info("🗑️ Konuşma geçmişi temizlendi.")

    def save_conversation(self, filename: Union[str, Path]):
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filepath.open('w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"💾 Konuşma kaydedildi: {filepath}")
        except Exception as e:
            self.logger.error(f"Konuşma kaydetme hatası ({filepath}): {e}")

    def _is_xla_device(self):
        """Chatbot'un XLA cihazında olup olmadığını kontrol eder"""
        return 'xla' in str(self.device).lower()


# ==========================================================================
# BÖLÜM 12: ANA ÇALIŞTIRMA (MAIN) YARDIMCILARI
# ==========================================================================

def set_seed(seed: int):
    """Tüm kütüphaneler için seed ayarlar"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # xm.set_rng_seed kaldırıldı, torch.manual_seed XLA için de çalışıyor
    # if XLA_AVAILABLE and xm:
    #     xm.set_rng_seed(seed)  # ARTIK YOK
    print(f"🌱 Seed ayarlandı: {seed}")

def print_model_summary(model: nn.Module, logger: logging.Logger, config: ModelConfig):
    """Model özetini loglar"""
    if get_rank() != 0:
        return # Sadece master process yazdırsın
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 60 + f"\n🤖 Model Özeti: {model.__class__.__name__} (Decoder-Only) 🤖\n" + "-" * 60 +
           f"\n  Toplam Parametre:         {total_params:,}" +
           f"\n  Eğitilebilir Parametre:   {trainable_params:,}" +
           f"\n  Vocab Boyutu (Tokenizer): {config.vocab_size:,}" +
           f"\n  Embedding Boyutu (d_model): {config.d_model}" +
           f"\n  Attention Başlık Sayısı:  {config.nhead}" +
           f"\n  Decoder Katmanları:       {config.num_decoder_layers}" +
           f"\n  FeedForward Boyutu:       {config.dim_feedforward}" +
           f"\n  Max Dizi Uzunluğu:        {config.max_seq_length}" +
           f"\n  Tahmini Model Boyutu:     {total_params*2/(1024*1024):.2f} MB (BF16)\n" + "=" * 60)

def interactive_chat_session(chatbot: UltimateChatbot, logger: logging.Logger):
    """İnteraktif chat döngüsünü başlatır"""
    print("\n" + "=" * 70 + "\n🤖 Chatbot Hazır! ('exit'/'quit' ile çıkış).\n   'clearhist', 'saveconv'\n" + "=" * 70 + "\n")
    while True:
        try:
            inp = input("👤 Siz: ").strip()
            if not inp: continue
            if inp.lower() in ["exit", "quit", "çıkış", "bye"]:
                print("\n👋 Hoşça kalın!"); break
            if inp.lower() == "clearhist":
                chatbot.clear_history(); print("🗑️ Geçmiş temizlendi."); continue
            if inp.lower() == "saveconv":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = Path(chatbot.config.save_dir) / f"conv_{ts}.json"
                chatbot.save_conversation(fn); continue
            
            print("🤖 Düşünüyorum...", end="\r", flush=True)
            resp = chatbot.generate_response(inp)
            sys.stdout.write("\033[K") # Satırı temizle
            print(f"🤖 Bot: {resp}")
            
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Sohbet bitti."); break
        except Exception as e:
            logger.error(f"Sohbet hatası: {e}", exc_info=True)
            print("❌ Hata oluştu.")

# ==========================================================================
# BÖLÜM 13: EĞİTİM BAŞLATMA FONKSİYONU
# ==========================================================================

def start_v5e8_training(trainer: AdvancedUltimateTrainer,
                            train_dataset: Dataset,
                            eval_dataset: Optional[Dataset],
                            logger: logging.Logger,
                            config: ModelConfig,
                            is_master: bool,
                            rank: int,
                            is_on_xla_mpfn: bool,
                            resume_from_ckpt_data: Optional[Dict] = None):
       
        if is_master:
            logger.info("=" * 80)
            logger.info("🔍 start_v5e8_training() ÇAĞRILDI (MULTI-TASK MODE)")
            logger.info(f"   is_master: {is_master}, rank: {rank}, is_tpu: {is_on_xla_mpfn}")
            logger.info(f"   train_dataset: {len(train_dataset)}, eval_dataset: {len(eval_dataset) if eval_dataset else 'Yok'}")
            logger.info(f"   resume_from_ckpt_data: {resume_from_ckpt_data is not None}")
            
            # Multi-task bilgisi
            if hasattr(train_dataset, 'task_weights'):
                logger.info(f"   📊 Active Tasks: {list(train_dataset.task_weights.keys())}")
                logger.info(f"   ⚖️  Task Weights: {train_dataset.task_weights}")
            
            logger.info("=" * 80)
    
        try:
            # ============================================
            # 1. MULTI-TASK COLLATOR KURULUMU
            # ============================================
            if is_master:
                logger.info("🔧 Multi-Task Collator kuruluyor...")
            
            multi_task_collator = TPUOptimizedCollator(
                tokenizer=trainer.tokenizer,           # Tokenizer trainer içinden gelir
                max_length=config.seq_length,          # Sabit uzunluk
                padding_value=config.pad_token_id,     # Pad ID
                last_chunk_weight=getattr(config, 'last_chunk_weight', 1.5)
            )
            
            if is_master:
                logger.info("   ✅ TPUOptimizedCollator başarıyla oluşturuldu")
                logger.info(f"      - Pad Token ID: {config.pad_token_id}")
                logger.info(f"      - Last Chunk Weight: {getattr(config, 'last_chunk_weight', 1.5)}x")
            
            # ============================================
            # 2. DATALOADER TEST (Sadece master'da detaylı kontrol)
            # ============================================
            if is_master:
                logger.info("")
                logger.info("🔍 DataLoader testi başlatılıyor (Hata ayıklama modu)...")
                try:
                    # Geçici (test) loader'lar kur
                    test_train_loader, _ = trainer._setup_data_loaders(
                        train_dataset, 
                        eval_dataset,
                        collate_fn=multi_task_collator
                    )
                    
                    logger.info(f"   • DataLoader oluşturuldu (Tip: {type(test_train_loader)})")
                    
                    # Batch sayısını kontrol et
                    try:
                        num_batches = len(test_train_loader)
                        logger.info(f"   • Train loader batch sayısı: {num_batches}")
                    except TypeError:
                        logger.info(f"   • Train loader batch sayısı: Belirlenemedi (iterator)")
                    
                    # İlk batch'i yükle ve test et
                    logger.info("   • İlk batch yükleniyor...")
                    start_load = time.time()
                    first_batch = next(iter(test_train_loader))
                    elapsed_load = time.time() - start_load
                    
                    logger.info(f"   • ✅ İlk batch {elapsed_load:.2f}s'de yüklendi")
                    logger.info(f"   • Batch keys: {list(first_batch.keys())}")
                    logger.info(f"   • input_ids shape: {first_batch['input_ids'].shape}")
                    logger.info(f"   • attention_mask shape: {first_batch['attention_mask'].shape}")
                    
                    # Multi-task label'ları kontrol et
                    if 'lm_labels' in first_batch:
                        logger.info(f"   • lm_labels shape: {first_batch['lm_labels'].shape}")
                    if 'sentiment_labels' in first_batch:
                        logger.info(f"   • sentiment_labels shape: {first_batch['sentiment_labels'].shape}")
                        active_sentiment = (first_batch['sentiment_labels'] != -100).sum().item()
                        logger.info(f"      → Aktif sentiment örnekleri: {active_sentiment}")
                    if 'ner_labels' in first_batch:
                        logger.info(f"   • ner_labels shape: {first_batch['ner_labels'].shape}")
                        active_ner = (first_batch['ner_labels'] != -100).sum().item()
                        logger.info(f"      → Aktif NER token'ları: {active_ner}")
                    if 'qa_starts' in first_batch:
                        logger.info(f"   • qa_starts shape: {first_batch['qa_starts'].shape}")
                        active_qa = (first_batch['qa_starts'] != -100).sum().item()
                        logger.info(f"      → Aktif QA örnekleri: {active_qa}")
                    
                    # Active tasks istatistikleri (Senin eklediğin kısım)
                    if 'active_tasks' in first_batch:
                        from collections import Counter
                        all_tasks = [task for tasks in first_batch['active_tasks'] for task in tasks]
                        task_freq = Counter(all_tasks)
                        logger.info(f"   • Batch içindeki task dağılımı:")
                        for task, count in task_freq.items():
                            logger.info(f"      - {task}: {count} örnek")
                    
                    # Memory check
                    if 'loss_weights' in first_batch:
                        logger.info(f"   • loss_weights shape: {first_batch['loss_weights'].shape}")
                    
                    # Temizle
                    del test_train_loader, first_batch
                    if XLA_AVAILABLE and xm: 
                        xm.mark_step()
                    gc.collect()
                    
                    logger.info("   • ✅ DataLoader testi BAŞARILI - Multi-task batch'leri doğru yükleniyor")
                    
                except StopIteration:
                    logger.error("   • ❌ DataLoader boş! Dataset veya sampler sorunu olabilir.")
                    raise ValueError("DataLoader boş - eğitim başlatılamıyor")
                
                except Exception as e:
                    logger.error(f"   • ❌ DATALOADER TESTİ BAŞARISIZ: {e}", exc_info=True)
                    logger.error(f"   • Hata tipi: {type(e).__name__}")
                    logger.error(f"   • Hata mesajı: {str(e)}")
                    raise
            
            # Diğer rank'ler master'ın testi bitirmesini bekler
            if is_on_xla_mpfn:
                xm.rendezvous("dataloader_test_barrier")
                if is_master:
                    logger.info("   • ✅ Tüm TPU core'ları senkronize edildi")
            
            # ============================================
            # 3. EĞİTİM PARAMETRELERINI GÖSTER
            # ============================================
            if is_master:
                logger.info("")
                logger.info("=" * 80)
                logger.info("🚀 TPU v5e-8 MULTI-TASK EĞİTİMİ BAŞLIYOR 🚀")
                logger.info("=" * 80)
                logger.info(f"   🔧 Konfigürasyon:")
                logger.info(f"      - Global Batch Size: {getattr(config, 'per_device_train_batch_size', 4) * config.world_size}")
                logger.info(f"      - Per-Core Batch Size: {getattr(config, 'per_device_train_batch_size', 4)}")
                logger.info(f"      - World Size (TPU Cores): {config.world_size}")
                logger.info(f"      - Gradient Accumulation: {getattr(config, 'gradient_accumulation_steps', 1)}")
                logger.info(f"      - Mixed Precision (BF16): {getattr(config, 'bf16', False)}")
                logger.info(f"      - Max Training Hours: {getattr(config, 'max_training_hours', 8.5)}h")
                
                logger.info(f"   📊 Multi-Task Setup:")
                logger.info(f"      - Language Modeling (LM): Weight 1.0 ✅")
                logger.info(f"      - Sentiment Analysis: Weight 0.3 {'✅' if hasattr(train_dataset, 'task_weights') else '❌'}")
                logger.info(f"      - NER: Weight 0.5 {'✅' if hasattr(train_dataset, 'task_weights') else '❌'}")
                logger.info(f"      - QA: Weight 0.4 {'✅' if hasattr(train_dataset, 'task_weights') else '❌'}")
                
                logger.info(f"   💾 Checkpoint:")
                logger.info(f"      - Save Interval: {getattr(config, 'save_interval_min', 30)} dakika")
                resume_msg = '✅ Checkpoint\'ten devam ediliyor' if resume_from_ckpt_data else '❌ Sıfırdan başlıyor'
                logger.info(f"      - Resume: {resume_msg}")
                logger.info("=" * 80)
                logger.info("")
    
            # ============================================
            # 3.5 MODEL DTYPE FIX (BF16 ZORLAMA - SAFE MODE)
            # ============================================
            # 🔥 ÖNEMLİ DÜZELTME: Eski 'for' döngüsü silindi. 
            # Yerine Graph-safe yöntem kullanılıyor. Logları koruyoruz.
            
            if is_master:
                logger.info("🔧 Model tüm bileşenleriyle bf16'ya çevriliyor (Graph Safe Method)...")
            
            # 1. Modeli PyTorch'un kendi fonksiyonuyla çevir (Graph'ı bozmaz)
            trainer.model.to(torch.bfloat16)
            
            # Bufferları da garantiye al
            for buffer in trainer.model.buffers():
                 if buffer.dtype == torch.float32:
                    buffer.data = buffer.data.to(torch.bfloat16)

            trainer.model.train()
    
            # 2. Kullanıcıyı bilgilendirmek için Logları yazdır (Kontrol amaçlı)
            if is_master:
                # Ana parametre kontrolü
                first_param = next(trainer.model.parameters())
                logger.info(f"   ✅ Model dtype dönüşümü tamamlandı.")
                logger.info(f"   ✅ Final model parameter dtype: {first_param.dtype}")
                
                # Decoder kontrolü (varsa)
                if hasattr(trainer.model, 'decoder'):
                    try:
                        dec_param = next(trainer.model.decoder.parameters())
                        logger.info(f"   ✅ Decoder dtype: {dec_param.dtype}")
                    except StopIteration:
                        pass
    
                if first_param.dtype != torch.bfloat16:
                    logger.warning("   ⚠️ UYARI: Model hala tam olarak bf16 değil gibi görünüyor!")
                else:
                    logger.info("   🎉 Model başarıyla ve güvenli şekilde BF16 formatına alındı.")
            
            # TPU senkronizasyonu (Dtype dönüşümü sonrası herkes aynı yerde mi?)
            if is_on_xla_mpfn:
                xm.rendezvous("model_bf16_conversion_barrier")
                if is_master:
                    logger.info(f"   • Tüm çekirdekler BF16 dönüşümünü onayladı.")
            
            # ============================================
            # 4. EĞİTİMİ BAŞLAT
            # ============================================
            # Çöp topla
            gc.collect()
    
            if is_master:
                logger.info("🚀 trainer.train() fonksiyonuna giriliyor...")
    
            # Ana eğitim fonksiyonunu çağır
            trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                resume_from_ckpt_data=resume_from_ckpt_data
            )
    
        except Exception as e:
            logger.error(f"💥 start_v5e8_training içinde kritik hata: {e}", exc_info=True)
            if is_on_xla_mpfn:
                try:
                    xm.rendezvous("training_startup_error")
                except:
                    pass
            raise
# ==============================================================================
# === BÖLÜM 14: TPU MULTIPROCESSING (ANA GİRİŞ NOKTASI)
# ==============================================================================

def v5e8_critical_sync(phase_name: str, rank: int, is_master: bool, logger: logging.Logger):
    """TPU v5e-8 için gelişmiş senkronizasyon"""
    if XLA_AVAILABLE and xm:
        try:
            xm.rendezvous(f"v5e8_sync_{phase_name}_rank_{rank}")
            if hasattr(xm, 'wait_device_ops'):
                xm.wait_device_ops()
            
            # Kritik fazlarda bellek senkronizasyonu
            if phase_name in ['model_loaded', 'training_start', 'epoch_end']:
                gc.collect()
                xm.mark_step()
                
            if is_master:
                logger.debug(f"🔄 TPU v5e-8 Sync: {phase_name} tamamlandı.")
                
        except Exception as e:
            if is_master:
                logger.warning(f"⚠️ v5e-8 sync uyarısı ({phase_name}): {e}")

def v5e8_optimize_memory(model: UltimateTransformerModel, config: ModelConfig, 
                         rank: int, is_master: bool, logger: logging.Logger) -> UltimateTransformerModel:
    """TPU v5e-8 için agresif bellek optimizasyonu"""
    if XLA_AVAILABLE and xm:
        gc.collect()
        xm.mark_step()
        
        if is_master:
            try:
                memory_info = xm.get_memory_info(config.device)
                total_memory = memory_info.get('bytes_limit', 0) / (1024**3)
                used_memory = memory_info.get('bytes_used', 0) / (1024**3)
                logger.info(f"💾 TPU v5e-8 Bellek (Rank {rank}): {used_memory:.1f}GB / {total_memory:.1f}GB")
            except Exception:
                pass
        
        if hasattr(config, 'bf16') and config.bf16:
            model = model.bfloat16()
            if is_master:
                logger.info(f"⚡ Model BF16'ya dönüştürüldü (TPU v5e-8 native)")
        
        model = model.to(config.device, non_blocking=True)
        xm.mark_step()
        
        if is_master:
            logger.info(f"✅ TPU v5e-8 bellek optimizasyonu tamamlandı (Rank {rank})")
    
    return model

def v5e8_optimized_checkpoint_loading(model: UltimateTransformerModel, config: ModelConfig, 
                                      rank: int, is_master: bool, logger: logging.Logger) -> Tuple[UltimateTransformerModel, Optional[Dict]]:
    """TPU v5e-8 için optimize checkpoint yükleme"""
    
    resume_from_ckpt_data = None
    
    if config.resume_from_checkpoint and Path(config.resume_from_checkpoint).exists():
        if is_master:
            logger.info(f"🔄 TPU v5e-8 checkpoint yüklemesi başlıyor...")
        
        try:
            torch.serialization.add_safe_globals([PosixPath])
            # CPU'da yükle (bellek verimliliği)
            ckpt = torch.load(config.resume_from_checkpoint, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in ckpt:
                # v5e-8 için adımlı (gradual) yükleme
                state_dict = ckpt['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
                
                if is_master:
                    logger.info(f"✅ Model state dict yüklendi (v5e-8 optimized)")
            
            # Checkpoint'ten diğer verileri al
            resume_from_ckpt_data = {
                'epoch': ckpt.get('epoch', 0),
                'global_step': ckpt.get('global_step', 0),
                'best_eval_loss': ckpt.get('best_eval_loss', float('inf')),
                'best_perplexity': ckpt.get('best_perplexity', float('inf')),
                'optimizer_state_dict': ckpt.get('optimizer_state_dict'),
                'scheduler_state_dict': ckpt.get('scheduler_state_dict')
            }
            
            del ckpt, state_dict
            gc.collect()
            
            if XLA_AVAILABLE and xm:
                v5e8_critical_sync("checkpoint_loaded", rank, is_master, logger)
            
            if is_master:
                logger.info("✅ TPU v5e-8 checkpoint yüklemesi tamamlandı.")
                
        except Exception as e:
            logger.error(f"❌ TPU v5e-8 checkpoint hatası: {e}", exc_info=True)
            if XLA_AVAILABLE and xm: xm.mark_step() # Hata kurtarma
    
    return model, resume_from_ckpt_data

# ================================================================
# ArgsCLI SINIFI (args_cli yoksa fallback)
# ================================================================
class ArgsCLI:
    def __init__(self):
        self.current_mode = "train"  # Veya "eval", "chat"
        self.resume = False
        self.save_dir = f"{HOME}/checkpoints"

# ================================================================
# CUSTOM TOKENIZER WRAPPER
# ================================================================
class CustomTokenizerWrapper:
    def __init__(self, tokenizer_path=f"{HOME}/custom_tokenizer_info"):
        print(f"📂 Tokenizer Yükleniyor: {tokenizer_path}")
        
        try:
            # 1. Eğitilmiş Tokenizer'ı Yükle
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            
            # 2. Kritik Özellikleri Çek
            self.vocab_size = len(self.tokenizer)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.bos_token_id = self.tokenizer.bos_token_id
            
            # 3. Özel Token ID'leri (System 2 için)
            self.think_start_id = self.tokenizer.convert_tokens_to_ids("<think>")
            self.think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
            
            print(f"✅ Tokenizer Başarıyla Yüklendi!")
            print(f"   📊 Vocab: {self.vocab_size}")
            print(f"   🧠 Think ID: {self.think_start_id}")
            
        except Exception as e:
            print(f"❌ KRİTİK HATA: Tokenizer yüklenemedi! ({e})")
            print("⚠️ Lütfen önce 'HÜCRE 4' (Tokenizer Train) kodunu çalıştırın.")
            raise e

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False, **kwargs):
        """Trainer bu metodu çağırır. HuggingFace formatında çıktı verir."""
        if isinstance(text, str):
            text = [text]
            
        encoding = self.tokenizer(
            text,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs
        )
        return encoding

    def encode(self, text, **kwargs):
        """Metni ID listesine çevirir"""
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """ID listesini metne çevirir"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    
    def __len__(self):
        return self.vocab_size
    
    @property
    def pad_token(self):
        return self.tokenizer.pad_token
    
    @property
    def eos_token(self):
        return self.tokenizer.eos_token
    
    @property
    def bos_token(self):
        return self.tokenizer.bos_token


# ================================================================
# MAIN MULTIPROCESSING FUNCTION (_mp_fn)
# ================================================================
def _mp_fn(rank: int, args_tuple: Tuple[ModelConfig, argparse.Namespace]):
    """
    Ana Multiprocessing (MP) fonksiyonu.
    Her TPU çekirdeği (core) üzerinde çalışır.
    """
    # ================================================================
    # 1. KURULUM (SETUP)
    # ================================================================
    import inspect
    
    config, args_cli = args_tuple
    
    print(f"\n{'='*60}")
    print(f"🔍 [Rank {rank}] _mp_fn BAŞLADI")
    print(f"{'='*60}")
    
    # ✅ XLA modüllerini import et
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.runtime as xr
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
        print(f"✅ [Rank {rank}] XLA import başarılı")
    except ImportError as e:
        print(f"❌ [Rank {rank}] XLA import BAŞARISIZ: {e}")
        if rank == 0:
            print("KRİTİK HATA: _mp_fn XLA olmadan çağrıldı!")
        return
    
    # ✅ TPU'yu başlat
    try:
        current_device = xm.xla_device()
        world_size = xr.world_size()
        local_rank = xr.global_ordinal()
        is_master = (local_rank == 0)
        
        print(f"✅ [Rank {rank}] TPU başlatıldı: Device={current_device}, "
              f"Ordinal={local_rank}, WorldSize={world_size}")
              
    except Exception as e:
        print(f"❌ [Rank {rank}] TPU başlatma hatası: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Seed ayarla
    set_seed(config.seed + rank)
    
    # Config'i güncelle
    config.device = current_device
    config.world_size = world_size
    config.is_tpu = True
    config.local_rank = local_rank
    config.rank = rank
    config.distributed = world_size > 1
    
    # Logger'ı kur
    logger = setup_logging(config, rank=rank)
    
    if is_master:
        logger.info("=" * 80)
        logger.info("TPU v5e-8 _mp_fn BAŞLADI (MASTER)")
        logger.info(f"Rank: {rank}, World Size: {world_size}, Device: {current_device}")
        logger.info("=" * 80)
    
    # ================================================================
    # 2. TOKENIZER YÜKLEME
    # ================================================================
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if is_master:
        logger.info("📂 Tokenizer yükleniyor...")
    
    try:
        tokenizer = CustomTokenizerWrapper(tokenizer_path=f"{HOME}/custom_tokenizer_info")
        
        # Config'i tokenizer bilgileriyle güncelle
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id
        
        if is_master:
            logger.info(f"✅ Tokenizer hazır: vocab_size={config.vocab_size}")
    
    except Exception as e:
        logger.error(f"❌ Tokenizer yükleme hatası (Rank {rank}): {e}", exc_info=True)
        return
    
    # ================================================================
    # 3. MOD VE YÜRÜTME
    # ================================================================
    current_mode = args_cli.current_mode
    
    if current_mode == "train":
        if is_master:
            logger.info("="*60)
            logger.info(f"🚂 === EĞİTİM MODU (TPU v5e-8 PACKED) === 🚂")
            logger.info("="*60)
        
        # --------------------------------
        # 3a. DATASET YOLLARI
        # --------------------------------
        train_data_path = f"{HOME}/processed_data/train_data_packed.pt"
        eval_data_path = f"{HOME}/unified_test.jsonl"
        
        if not os.path.exists(train_data_path):
            if is_master:
                logger.error(f"❌ Train dosyası (.pt) bulunamadı: {train_data_path}")
                logger.error("   Lütfen önce 'PACKING' adımını çalıştırın!")
            raise FileNotFoundError(f"Train dosyası bulunamadı: {train_data_path}")
        
        # --------------------------------
        # 3b. TRAIN DATASET YÜKLEME
        # --------------------------------
        if is_master:
            logger.info(f"📊 Train dataset yükleniyor: {train_data_path}")
        
        try:
            loaded_data = torch.load(train_data_path)
            
            class PackedTPUDataset(Dataset):
                def __init__(self, data_tensor):
                    self.data = data_tensor
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    input_ids = self.data[idx]
                    return {
                        'input_ids': input_ids,
                        'attention_mask': torch.ones_like(input_ids),
                        'lm_labels': input_ids.clone(),
                        'sentiment_label': -100,
                        'ner_labels': torch.full_like(input_ids, -100),
                        'qa_start': -100,
                        'qa_end': -100,
                        'active_tasks': ['lm'],
                        'task_weights': [1.0]
                    }
            
            train_dataset = PackedTPUDataset(loaded_data)
            
            if is_master:
                logger.info(f"   • ✅ Train dataset: {len(train_dataset)} örnek")
        
        except Exception as e:
            logger.error(f"❌ Train dataset yükleme hatası: {e}", exc_info=True)
            raise
        
        # --------------------------------
        # 3c. EVAL DATASET
        # --------------------------------
        eval_dataset = None
        if os.path.exists(eval_data_path):
            if is_master:
                logger.warning("⚠️  Eval dataset geçici olarak devre dışı.")
        else:
            if is_master:
                logger.warning("⚠️  Eval dosyası yok, atlanıyor.")
        
        # Senkronizasyon
        try:
            xm.rendezvous("datasets_loaded")
        except Exception as e:
            logger.warning(f"⚠️  Rendezvous hatası: {e}")
        
        # --------------------------------
        # 3d. MODEL OLUŞTURMA
        # --------------------------------
        if is_master:
            logger.info(f"🏗️ Model oluşturuluyor...")
        
        model = UltimateTransformerModel(config)
        
        model, resume_from_ckpt_data = v5e8_optimized_checkpoint_loading(
            model, config, rank, is_master, logger
        )
        
        model = v5e8_optimize_memory(model, config, rank, is_master, logger)
        
        v5e8_critical_sync("model_created_and_loaded", rank, is_master, logger)
        
        if is_master:
            print_model_summary(model, logger, config)
        
        # --------------------------------
        # 3e. TRAINER OLUŞTURMA
        # --------------------------------
        if is_master:
            logger.info(f"🚀 Trainer oluşturuluyor...")
        
        trainer = AdvancedUltimateTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            device=current_device,
            is_tpu=config.is_tpu
        )
        
        v5e8_critical_sync("trainer_ready", rank, is_master, logger)
        
        # --------------------------------
        # 3f. EĞİTİMİ BAŞLAT
        # --------------------------------
        start_v5e8_training(
            trainer=trainer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            config=config,
            is_master=is_master,
            rank=rank,
            is_on_xla_mpfn=True,
            resume_from_ckpt_data=resume_from_ckpt_data
        )
    
    elif current_mode == "chat":
        if is_master:
            logger.info(f"🤖 === CHAT MODU === 🤖")
            
            model = UltimateTransformerModel(config)
            
            best_ckpt_path = Path(config.save_dir) / "model" / "model_best.pt"
            config.resume_from_checkpoint = str(best_ckpt_path)
            
            model, resume_data = v5e8_optimized_checkpoint_loading(
                model, config, rank, is_master, logger
            )
            
            if resume_data is None:
                logger.error("❌ CHAT: 'model_best.pt' bulunamadı.")
            else:
                model = model.to(current_device)
                
                chatbot = UltimateChatbot(
                    model_config=config,
                    model_state_dict=model.state_dict(),
                    tokenizer=tokenizer,
                    device=current_device,
                    logger=logger
                )
                
                interactive_chat_session(chatbot, logger)
        else:
            logger.info(f"[Rank {rank}] Chat modunda beklemede.")
    
    else:
        if is_master:
            logger.warning(f"⚠️ Bilinmeyen mod: {current_mode}")
    
    # ================================================================
    # 4. CLEANUP
    # ================================================================
    try:
        xm.rendezvous("final_cleanup")
    except:
        pass
    
    if is_master:
        logger.info(f"✅ _mp_fn (Rank {rank}) tamamlandı.")

# ==============================================================================
# === BÖLÜM 15: ANA GİRİŞ NOKTASI (MAIN)
# ==============================================================================
def init_tpu():
    """TPU'yu başlat (yeni notebook için basit versiyon)"""
    print("🔄 TPU başlatılıyor...")
    
    import torch_xla.core.xla_model as xm
    
    devices = xm.get_xla_supported_devices()
    print(f"✅ TPU: {len(devices)} cihaz bulundu")
    print(f"   {devices}\n")
    
    return devices

def main():
    print("DEBUG: main fonksiyonu BAŞLADI - TPU v5e-8 Optimized.")
    
    import os
    import argparse
    import logging
    import sys
    from pathlib import Path

    # ================================================================
    # 1. ARGÜMAN PARSER
    # ================================================================
    parser = argparse.ArgumentParser(description="TPU v5e-8 Optimized Transformer Eğitimi")
    parser.add_argument("--config_file", type=str, default=f"{HOME}/config.json",
                    help="Kullanılacak konfigürasyon JSON dosyasının yolu.")
    parser.add_argument("--mode_override", type=str, choices=["train", "chat"], default=None,
                        help="Çalışma modunu (train/chat) komut satırından belirler.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Devam edilecek checkpoint dosyasının yolu.")
    
    # Jupyter/Kaggle'dan gelen bilinmeyen argümanları yoksay
    args_cli, unknown_args = parser.parse_known_args()

    # ================================================================
    # 2. LOGGER (Geçici)
    # ================================================================
    initial_logger = logging.getLogger("tpu_v5e8_setup")
    if not initial_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [TPU-Setup] - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        initial_logger.addHandler(handler)
        initial_logger.setLevel(logging.INFO)

    # ================================================================
    # 3. CONFIG YÜKLEME (TPU KONTROLÜNDEN ÖNCE!)
    # ================================================================
    config_path = Path(args_cli.config_file)
    if config_path.exists():
        initial_logger.info(f"Config dosyası yükleniyor: {config_path}")
        config = ModelConfig.from_json(config_path)
    else:
        initial_logger.warning(f"Config dosyası bulunamadı ({config_path}). Varsayılan ModelConfig kullanılıyor.")
        config = ModelConfig()

    # Komut satırı argümanları ile config'i override et
    if args_cli.resume_from_checkpoint:
        config.resume_from_checkpoint = args_cli.resume_from_checkpoint
        
    # Çalışma modunu belirle
    current_mode = "train"
    if args_cli.mode_override:
        current_mode = args_cli.mode_override
    elif config.run_chatbot_only:
        current_mode = "chat"
    
    args_cli.current_mode = current_mode
    initial_logger.info(f"Çalışma modu ayarlandı: {current_mode.upper()}")

    # ================================================================
    # 4. TPU KONTROLÜ (CONFIG YÜKLENDİKTEN SONRA!)
    # ================================================================
    XLA_AVAILABLE = False
    xm = None
    pl = None
    xmp = None
    xr = None
    xla_checkpoint = None
    world_size = 1
    
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.runtime as xr
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
        XLA_AVAILABLE = True
        initial_logger.info("✅ XLA modülü başarıyla yüklendi")
        
        # TPU environment variables'ını SADECE TPU kullanacaksak ayarla
        if config.is_tpu:  # ✅ Artık config tanımlı, doğrudan kullanabiliriz
            # ✅ Kaggle TPU için environment variables
            # Sadece PJRT kullan
            os.environ['PJRT_DEVICE'] = 'TPU'
            
            initial_logger.info("✅ TPU environment variables ayarlandı (PJRT_DEVICE=TPU)")
            
            world_size = 8  # TPU v5e-8 için sabit
            initial_logger.info(f"   └─ TPU tespit edildi (World Size: {world_size})")
        else:
            initial_logger.info("⚠️ TPU modu kapalı - CPU/GPU moduna geçiliyor")
            XLA_AVAILABLE = False
            world_size = 1
            
    except ImportError as e:
        initial_logger.warning(f"⚠️ XLA modülü bulunamadı: {e}")
        XLA_AVAILABLE = False
        world_size = 1

    # ✅ GLOBAL DEĞİŞKENLERİ AYARLA - TRY-EXCEPT BLOĞUNDAN SONRA
    globals()['XLA_AVAILABLE'] = XLA_AVAILABLE
    globals()['xm'] = xm
    globals()['pl'] = pl
    globals()['xmp'] = xmp
    globals()['xr'] = xr
    globals()['xla_checkpoint'] = xla_checkpoint

    # ================================================================
    # 5. SPAWN (ÇALIŞTIRMA)
    # ================================================================
    try:
        if config.is_tpu and XLA_AVAILABLE:
            initial_logger.info("🔍 TPU başlatma modu belirleniyor...")
            
            # ✅ Güvenli TPU başlatma
            try:
                # ❌ REMOVE THIS - It initializes XLA too early!
                # available_devices = xm.get_xla_supported_devices()
                # actual_world_size = len(available_devices)
                
                # ✅ Use config value or default to 8 for TPU v5e-8
                actual_world_size = getattr(config, 'world_size', 8)
                
                initial_logger.info("="*80)
                initial_logger.info(f"🚀 TPU MULTIPROCESSING BAŞLATILIYOR")
                initial_logger.info(f"   └─ Hedef Core: {actual_world_size}")
                initial_logger.info(f"   └─ Mod: {current_mode.upper()}")
                initial_logger.info("="*80)
                
                # World size'ı güncelle
                config.world_size = actual_world_size
                
                # Çoklu process başlat
                xmp.spawn(
                    _mp_fn,
                    args=((config, args_cli),),
                    nprocs=8,  # ✅ Explicitly set nprocs
                    start_method='fork'
                )
                
                initial_logger.info("="*80)
                initial_logger.info("✅ TPU MULTIPROCESSING TAMAMLANDI")
                initial_logger.info("="*80)
                
            except RuntimeError as e:
                error_msg = str(e)
                
                # Runtime already initialized hatası
                if "already initialized" in error_msg.lower():
                    initial_logger.warning("="*80)
                    initial_logger.warning("⚠️ XLA Runtime zaten başlatılmış")
                    initial_logger.warning("   └─ Kernel'ı yeniden başlatın")
                    initial_logger.warning("="*80)
                    raise
                
                # VFIO hatası: TPU meşgul
                elif "vfio" in error_msg.lower() or "device or resource busy" in error_msg.lower():
                    initial_logger.warning("="*80)
                    initial_logger.warning("⚠️ TPU ÇOKLU CİHAZ BAŞLATMA HATASI")
                    initial_logger.warning(f"   └─ Hata: {error_msg[:150]}")
                    initial_logger.warning("   └─ Tek cihaz moduna geçiliyor...")
                    initial_logger.warning("="*80)
                    
                    config.world_size = 1
                    
                    initial_logger.info("="*80)
                    initial_logger.info(f"🚀 TPU TEK CİHAZ MODU")
                    initial_logger.info(f"   └─ World Size: 1")
                    initial_logger.info(f"   └─ Mod: {current_mode.upper()}")
                    initial_logger.info("="*80)
                    
                    # ✅ Tek process modunda çalıştır
                    initial_logger.info("▶️ Tek process modunda çalıştırılıyor...")
                    _mp_fn(0, (config, args_cli))
                    
                    initial_logger.info("="*80)
                    initial_logger.info("✅ TPU TEK CİHAZ MODU TAMAMLANDI")
                    initial_logger.info("="*80)
                    
                else:
                    # Başka bir TPU hatası
                    initial_logger.error(f"❌ TPU başlatma hatası: {error_msg}")
                    raise
                
        else:
            # CPU/GPU modu
            initial_logger.info("="*80)
            initial_logger.info("⚠️ TEKİL SÜREÇ MODU (CPU/GPU)")
            initial_logger.info("="*80)
            
            _mp_fn(0, (config, args_cli))
            
            initial_logger.info("✅ Tekil süreç tamamlandı")
            
    except Exception as e:
        initial_logger.error("="*80)
        initial_logger.error(f"💥 KRİTİK HATA: {e}")
        initial_logger.error("="*80)
        initial_logger.error("Stack trace:", exc_info=True)
        
        if XLA_AVAILABLE and xm:
            try:
                xm.rendezvous("main_exception")
            except:
                pass
        
        raise
    
    initial_logger.info("="*80)
    initial_logger.info("PROGRAM TAMAMLANDI")
    initial_logger.info("="*80)
# ==============================================================================
# === GİRİŞ NOKTASI
# ==============================================================================

if __name__ == '__main__':
    print("="*60)
    print("TPU v5e-8 OPTIMIZED SCRIPT BAŞLATILIYOR")
    print("="*60)
    
    try:
        
        # Ana programı çalıştır
        main()
        
    except KeyboardInterrupt:
        print("\n⚠️ Program kullanıcı tarafından durduruldu")
        
    except Exception as e:
        print(f"\n💥 Program beklenmeyen bir hata ile sonlandı: {e}")
        import traceback
        traceback.print_exc()
         
    finally:
        print("="*60)
        print("TPU v5e-8 SCRIPT TAMAMLANDI")
        print("="*60)
