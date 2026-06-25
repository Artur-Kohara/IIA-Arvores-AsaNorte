import os
import sys

# Garante que `import utils` funcione ao rodar o pytest a partir de qualquer
# diretório (local ou na CI), apontando para a raiz do repositório.
RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_REPO not in sys.path:
    sys.path.insert(0, RAIZ_REPO)
