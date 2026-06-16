import os, sys

print("=" * 60)
print("PYTHON & ORTAM")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Dizin: {os.getcwd()}")
print(f"Dosyalar: {os.listdir('.')}")

print("\nKUTUPHANELER")
print("=" * 60)

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    try:
        import torch_xla.core.xla_model as xm
        print(f"TPU (XLA): {xm.xla_device()}")
    except:
        print("torch_xla yok")
except:
    print("PyTorch yok")

try:
    import jax
    print(f"JAX: {jax.__version__}")
    print(f"Cihazlar: {jax.devices()}")
except:
    print("JAX yok")

print("\nMODEL DOSYALARI")
print("=" * 60)
for path in ["./model", "./checkpoints", "./noesis", "./weights", "./output"]:
    print(f"{'VAR' if os.path.exists(path) else 'YOK'} -> {path}")

os.system("cat /proc/cpuinfo | grep 'model name' | head -1")
os.system("free -h")
os.system("df -h .")
