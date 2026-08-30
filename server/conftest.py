import sys
from pathlib import Path

# server/ is a flat app dir (main.py does `from errors import ...`); make it importable for tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
