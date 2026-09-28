"""check_env.py - read-only health check of the GPU environment. Run: python scripts\\check_env.py"""
import importlib, shutil, subprocess, sys
print('python      :', sys.executable, sys.version.split()[0])
print('conda env ok:', 'aquavision' in sys.executable.lower(), '(should be True)')
try:
    import torch
    print('torch       :', torch.__version__, '| CUDA available:', torch.cuda.is_available(),
          '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
except Exception as e:
    print('torch       : NOT IMPORTABLE', e)
for m in ('pycocotools', 'ultralytics', 'mmdet', 'mmcv', 'mmengine', 'matplotlib', 'PIL', 'lxml', 'numpy'):
    try:
        mod = importlib.import_module(m); print(f'{m:12s}:', getattr(mod, '__version__', 'installed'))
    except Exception as e:
        print(f'{m:12s}: MISSING ({type(e).__name__})')
if shutil.which('nvidia-smi'):
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                          '--format=csv'], capture_output=True, text=True).stdout)
else:
    print('nvidia-smi not on PATH')
