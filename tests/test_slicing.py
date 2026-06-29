import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import utils

"""
tests/test_slicing.py
Testes unitários para validar a rotina de fatiamento de ortofotos GeoTIFF.
Responsável: Pessoa 1 (Luidgi Varela Carneiro)
"""

@pytest.fixture
def synthetic_geotiff(tmp_path):
    """Gera um GeoTIFF sintético pequeno (1280x1280) para testes."""
    file_path = tmp_path / "test_orthophoto.tif"
    width, height = 1280, 1280
    bands = 3
    data = np.random.randint(0, 256, (bands, height, width), dtype=np.uint8)
    
    transform = from_origin(-47.89, -15.75, 0.00001, 0.00001)
    crs = 'EPSG:4326'
    
    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
        
    return file_path

def test_slice_geotiff_dimensions(synthetic_geotiff, tmp_path):
    """
    Critério de aceitação:
    Assertar que o tamanho de saída de cada tile é exatamente (640, 640)
    e que nenhuma imagem recortada possui dimensões diferentes de 640x640.
    """
    output_dir = tmp_path / "tiles"
    tile_size = 640
    
    # Executa a função real de fatiamento
    tiles = utils.slice_geotiff(synthetic_geotiff, output_dir, tile_size=tile_size)
    
    # Com imagem de 1280x1280 e tile_size de 640, devemos ter exatamente 4 tiles
    assert len(tiles) == 4, f"Esperava 4 tiles, obteve {len(tiles)}"
    
    for tile_path in tiles:
        assert Path(tile_path).exists(), f"Tile {tile_path} não foi gravado no disco."
        with rasterio.open(tile_path) as src:
            assert src.width == tile_size, f"Largura inválida: {src.width}"
            assert src.height == tile_size, f"Altura inválida: {src.height}"
            assert src.count == 3, f"Canais inválidos: {src.count}"
            assert src.crs is not None, "Metadata CRS ausente no tile"
            assert not src.transform.is_identity, "Transformação geométrica ausente no tile"
