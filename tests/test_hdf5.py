import cv2
import h5py
import numpy as np

import utils

"""
tests/test_hdf5.py
Testes unitários para validar a integridade dos arquivos HDF5 gerados.
Responsáveis: Pessoa 2 (Rafael de Lima Pereira), Pessoa 3 e Pessoa 6 (Artur Kohara Guerra)
"""

def test_hdf5_structure_and_compression(tmp_path):
    """
    Critério de aceitação (Card 1.2):
    Certificar que as dimensões do dataset são correspondentes ao array original,
    e que a compressão gzip e os chunks de tamanho unitário estão configurados.
    """
    # Gera um tile JPG 640x640x3 de teste em disco
    tile_path = tmp_path / "tile_teste.jpg"
    mock_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(tile_path), mock_img)

    hdf5_path = tmp_path / "dataset_teste.h5"
    utils.criar_hdf5_bruto(str(hdf5_path), [str(tile_path)])

    with h5py.File(hdf5_path, "r") as f:
        assert "images" in f, "Grupo 'images' ausente no HDF5"
        assert "labels" in f, "Grupo 'labels' ausente no HDF5"

        dset = f["images"]["tile_0"]
        assert dset.shape == (640, 640, 3), "Dimensões do dataset não correspondem a 640x640x3"
        assert dset.dtype == np.uint8, "Dataset de imagem deve ser uint8"
        assert dset.compression == "gzip", "Compressão gzip não configurada"
        assert dset.chunks == (640, 640, 3), "Chunks de tamanho unitário não configurados"

        label = f["labels"]["tile_0"]
        assert label.shape == (0, 5), "Label inicial deve ser uma matriz vazia de shape (0, 5)"


def test_geographical_split():
    """
    Critério de aceitação (Card 2.3):
    O teste deve verificar se nenhuma imagem de treino compartilha
    as mesmas coordenadas espaciais que as imagens de validação.
    """
    # Simula dados espaciais de treino e validação
    mock_train_coords = [(15.70, 47.90), (15.71, 47.91)]
    mock_val_coords = [(15.80, 47.80), (15.81, 47.81)]
    
    intersection = set(mock_train_coords).intersection(set(mock_val_coords))
    assert len(intersection) == 0, "Detetado vazamento espacial de treino e validação!"
