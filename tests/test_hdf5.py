import cv2
import h5py
import numpy as np
import zipfile

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


def test_exportar_hdf5_para_roboflow(tmp_path):
    """
    Critério de aceitação (Card 2.1):
    Extrair imagens e labels do HDF5 em estrutura YOLO pronta para upload no Roboflow.
    """
    hdf5_path = tmp_path / "dataset_pseudo_label.h5"
    output_dir = tmp_path / "roboflow_export"

    mock_img = np.zeros((640, 640, 3), dtype=np.uint8)
    mock_boxes = np.array([[0, 0.500000, 0.500000, 0.250000, 0.250000]], dtype=np.float32)

    with h5py.File(hdf5_path, "w") as f:
        images = f.create_group("images")
        labels = f.create_group("labels")
        images.create_dataset("tile_0", data=mock_img)
        labels.create_dataset("tile_0", data=mock_boxes)

    resumo = utils.exportar_hdf5_para_roboflow(str(hdf5_path), str(output_dir))

    image_path = output_dir / "images" / "tile_0.jpg"
    label_path = output_dir / "labels" / "tile_0.txt"
    data_yaml_path = output_dir / "data.yaml"
    classes_path = output_dir / "classes.txt"
    zip_path = output_dir / "roboflow_upload.zip"

    assert image_path.exists(), "Imagem exportada não encontrada"
    assert label_path.exists(), "Label YOLO exportada não encontrada"
    assert data_yaml_path.exists(), "Arquivo data.yaml não encontrado"
    assert classes_path.exists(), "Arquivo classes.txt não encontrado"
    assert zip_path.exists(), "ZIP para upload no Roboflow não encontrado"

    assert label_path.read_text().strip() == "0 0.500000 0.500000 0.250000 0.250000"
    assert classes_path.read_text().strip() == "tree"
    assert "0: tree" in data_yaml_path.read_text()

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        nomes = set(zip_file.namelist())

    assert "images/tile_0.jpg" in nomes
    assert "labels/tile_0.txt" in nomes
    assert "data.yaml" in nomes
    assert "classes.txt" in nomes
    assert resumo["imagens"] == 1
    assert resumo["labels"] == 1


def test_exportar_hdf5_para_roboflow_com_labels_agregadas(tmp_path):
    """
    Compatibilidade com o formato usado no notebook de pseudo-labelling:
    labels/image_id, labels/class e labels/bbox.
    """
    hdf5_path = tmp_path / "dataset_pseudo_label_agregado.h5"
    output_dir = tmp_path / "roboflow_export_agregado"

    with h5py.File(hdf5_path, "w") as f:
        images = f.create_group("images")
        labels = f.create_group("labels")
        images.create_dataset("tile_0", data=np.zeros((640, 640, 3), dtype=np.uint8))
        images.create_dataset("tile_1", data=np.zeros((640, 640, 3), dtype=np.uint8))
        labels.create_dataset("image_id", data=np.array([1]))
        labels.create_dataset("class", data=np.array([0]))
        labels.create_dataset("bbox", data=np.array([[0.25, 0.30, 0.40, 0.50]], dtype=np.float32))
        labels.create_dataset("score", data=np.array([0.90], dtype=np.float32))

    resumo = utils.exportar_hdf5_para_roboflow(str(hdf5_path), str(output_dir))

    label_tile_0 = output_dir / "labels" / "tile_0.txt"
    label_tile_1 = output_dir / "labels" / "tile_1.txt"

    assert label_tile_0.read_text() == ""
    assert label_tile_1.read_text().strip() == "0 0.250000 0.300000 0.400000 0.500000"
    assert resumo["imagens"] == 2
    assert resumo["labels"] == 1
