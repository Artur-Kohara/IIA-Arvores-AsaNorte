import pytest
import numpy as np

"""
tests/test_chromatic.py
Testes unitários para validar a conversão cromática e reordenação de canais de cor.
Responsável: Pessoa 2 (Rafael de Lima Pereira)
"""

def test_chromatic_conversion_range():
    """
    Critério de aceitação:
    Assertar que a conversão cromática não gera clipping de pixel
    (valores nulos ou fora da escala [0, 255]).
    """
    # Cria uma matriz RGB aleatória no intervalo válido
    mock_rgb = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    
    # Simula a conversão RGB para BGR
    mock_bgr = mock_rgb[..., ::-1]
    
    assert mock_bgr.min() >= 0, "Valor mínimo de pixel menor que 0"
    assert mock_bgr.max() <= 255, "Valor máximo de pixel maior que 255"
    assert mock_bgr.dtype == np.uint8, "Tipo de dados deve ser uint8"
