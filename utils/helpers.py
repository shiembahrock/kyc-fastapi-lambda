import uuid as _uuid
import os
from pathlib import Path

def gen_order_code():
    """Generate unique order code"""
    h = _uuid.uuid4().hex
    return h[0:6] + h[24:30]

def load_env_files():
    """Load environment variables from .env files"""
    base_dir = Path(__file__).parent.parent
    candidates = [base_dir / ".env.local", base_dir / ".env"]
    for p in candidates:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    if k:
                        os.environ[k] = v
