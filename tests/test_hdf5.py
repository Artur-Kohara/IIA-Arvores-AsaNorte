import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import h5py
import numpy as np

from utils import load_tile_mapping
from utils import create_geographical_split

"""
tests/test_hdf5.py
Testes unitários para validar a integridade dos arquivos HDF5 gerados.
Responsáveis: Pessoa 2 (Rafael de Lima Pereira), Pessoa 3 e Pessoa 6 (Artur Kohara Guerra)
"""

def test_hdf5_structure_and_compression():
    """
    Critério de aceitação (Card 1.2):
    Certificar que as dimensões do dataset são correspondentes ao array original,
    e que a compressão gzip e os chunks de tamanho unitário estão configurados.
    """
    # Teste de verificação teórica/mock da chamada h5py
    pass

def test_geographical_split():
    """
    Critério de aceitação (Card 2.3):
    O teste deve verificar se nenhuma imagem de treino compartilha
    as mesmas coordenadas espaciais que as imagens de validação.
    """
    # Simula dados espaciais de treino e validação
    train_tiles, val_tiles = create_geographical_split(
        "data/tile_mapping.csv"
    )

    assert len(set(train_tiles) & set(val_tiles)) == 0

def test_all_tiles_used():

    mapping = load_tile_mapping("data/tile_mapping.csv")

    train_tiles, val_tiles = create_geographical_split(
        "data/tile_mapping.csv"
    )

    assert len(train_tiles) + len(val_tiles) == len(mapping)

def test_sectors():

    mapping = load_tile_mapping("data/tile_mapping.csv")

    train_tiles, val_tiles = create_geographical_split(
        "data/tile_mapping.csv"
    )

    train_mapping = mapping[mapping.tile_name.isin(train_tiles)]
    val_mapping = mapping[mapping.tile_name.isin(val_tiles)]

    train_sectors = set(train_mapping["sector"])
    val_sectors = set(val_mapping["sector"])

    assert train_sectors.isdisjoint(val_sectors)
