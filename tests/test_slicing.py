import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent))
import utils

"""
tests/test_slicing.py
Testes unitários para validar a rotina de fatiamento de ortofotos GeoTIFF.
Responsável: Pessoa 1 (Luidgi Varela Carneiro)
"""

_CRS = "EPSG:4326"
_TRANSFORM = from_bounds(-47.89, -15.78, -47.88, -15.77, 1280, 1280)


def _criar_geotiff_sintetico(path, width=1280, height=1280, bands=3):
    """Cria um GeoTIFF sintético em disco para uso nos testes."""
    data = np.random.randint(0, 255, (bands, height, width), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": bands,
        "crs": _CRS,
        "transform": _TRANSFORM,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_slice_geotiff_numero_de_tiles(tmp_path):
    """Uma imagem 1280×1280 deve gerar exatamente 4 tiles 640×640 (2×2)."""
    src = _criar_geotiff_sintetico(tmp_path / "imagem.tif", width=1280, height=1280)
    tiles = utils.slice_geotiff(src, tmp_path / "tiles", tile_size=640)
    assert len(tiles) == 4


def test_slice_geotiff_dimensoes(tmp_path):
    """Todos os tiles gerados devem ter exatamente 640×640 pixels."""
    src = _criar_geotiff_sintetico(tmp_path / "imagem.tif", width=1280, height=1280)
    tiles = utils.slice_geotiff(src, tmp_path / "tiles", tile_size=640)
    for tile_path in tiles:
        with rasterio.open(tile_path) as t:
            assert t.width == 640
            assert t.height == 640


def test_slice_geotiff_preserva_crs(tmp_path):
    """Cada tile deve herdar o CRS do GeoTIFF original."""
    src = _criar_geotiff_sintetico(tmp_path / "imagem.tif", width=1280, height=1280)
    tiles = utils.slice_geotiff(src, tmp_path / "tiles", tile_size=640)
    for tile_path in tiles:
        with rasterio.open(tile_path) as t:
            assert t.crs is not None
            assert str(t.crs) == _CRS


def test_slice_geotiff_transform_nao_identidade(tmp_path):
    """Cada tile deve ter um transform georreferenciado (não identidade)."""
    src = _criar_geotiff_sintetico(tmp_path / "imagem.tif", width=1280, height=1280)
    tiles = utils.slice_geotiff(src, tmp_path / "tiles", tile_size=640)
    for tile_path in tiles:
        with rasterio.open(tile_path) as t:
            assert not t.transform.is_identity


def test_slice_geotiff_ignora_bordas_incompletas(tmp_path):
    """Uma imagem 1000×1000 não é múltiplo de 640, logo gera apenas 1 tile."""
    src = _criar_geotiff_sintetico(tmp_path / "imagem.tif", width=1000, height=1000)
    tiles = utils.slice_geotiff(src, tmp_path / "tiles", tile_size=640)
    assert len(tiles) == 1


def test_slice_geotiff_arquivo_inexistente(tmp_path):
    """Deve lançar FileNotFoundError para caminhos inválidos."""
    with pytest.raises(FileNotFoundError):
        utils.slice_geotiff(tmp_path / "nao_existe.tif", tmp_path / "tiles")


def test_slice_geotiff_tile_size_invalido(tmp_path):
    """tile_size <= 0 deve lançar ValueError."""
    src = _criar_geotiff_sintetico(tmp_path / "imagem.tif")
    with pytest.raises(ValueError):
        utils.slice_geotiff(src, tmp_path / "tiles", tile_size=0)
