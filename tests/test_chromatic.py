import numpy as np

import utils

"""
tests/test_chromatic.py
Testes unitários para validar a conversão cromática e reordenação de canais de cor.
Responsável: Pessoa 2 (Rafael de Lima Pereira)
"""

def test_chromatic_conversion_range():
    """
    Critério de aceitação:
    Assertar que a conversão cromática não gera clipping de pixel
    (valores nulos ou fora da escala [0, 255]) e mantém o tipo uint8.
    """
    mock_rgb = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    mock_bgr = utils.converter_rgb_para_bgr(mock_rgb)

    assert mock_bgr.min() >= 0, "Valor mínimo de pixel menor que 0"
    assert mock_bgr.max() <= 255, "Valor máximo de pixel maior que 255"
    assert mock_bgr.dtype == np.uint8, "Tipo de dados deve ser uint8"


def test_chromatic_channel_swap():
    """
    A reordenação RGB -> BGR deve trocar os canais R e B e manter o canal G.
    """
    mock_rgb = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)

    mock_bgr = utils.converter_rgb_para_bgr(mock_rgb)

    assert np.array_equal(mock_bgr[..., 0], mock_rgb[..., 2]), "B deve receber o canal R original"
    assert np.array_equal(mock_bgr[..., 1], mock_rgb[..., 1]), "Canal G (verde) deve permanecer igual"
    assert np.array_equal(mock_bgr[..., 2], mock_rgb[..., 0]), "R deve receber o canal B original"


def test_chromatic_chw_to_hwc():
    """
    Tiles lidos pelo rasterio chegam em CHW (bandas, H, W); a conversão deve
    devolver o formato HWC (H, W, 3) esperado pelo OpenCV/HDF5.
    """
    mock_chw = np.random.randint(0, 256, (3, 640, 640), dtype=np.uint8)

    mock_bgr = utils.converter_rgb_para_bgr(mock_chw)

    assert mock_bgr.shape == (640, 640, 3), "Saída deve estar em formato HWC (640, 640, 3)"
    # entrada CHW em ordem [R, G, B]; saída BGR => canal 0 (B) == banda 2 (B), canal 2 (R) == banda 0 (R)
    assert np.array_equal(mock_bgr[..., 0], mock_chw[2]), "Canal B de saída deve vir da banda B (índice 2) de entrada"
    assert np.array_equal(mock_bgr[..., 2], mock_chw[0]), "Canal R de saída deve vir da banda R (índice 0) de entrada"
