# Notebook Integrador - IIA-Arvores-AsaNorte

Este arquivo descreve e documenta o fluxo de execução unificado do pipeline do **Projeto 2 da disciplina de Introdução à Inteligência Artificial (CIC/UnB)**, integrando as contribuições de todos os membros e definindo mocks funcionais para as etapas futuras de treinamento.

---

## 🗺️ Visão Geral do Pipeline Integrado

O pipeline completo é dividido em 4 fases principais operadas de forma encadeada, garantindo que o dado processado por cada membro sirva de insumo direto para a fase do próximo integrante.

```mermaid
graph TD
    A[WMS/GeoTIFF Original] -->|Slicing rasterio (Luidgi Varela)| B(Tiles GeoTIFF 640x640)
    B -->|Normalização & BGR (Rafael Lima)| C(HDF5 Bruto dataset_v1_raw.h5)
    C -->|Pseudo-Labelling DeepForest (Vitor Lopes)| D(HDF5 com Pseudo-Labels)
    D -->|Exportador Roboflow (Felipe Costa)| E[Pasta YOLO + Zip para upload]
    E -->|Revisão & Limpeza (Lucas Saad)| F[Roboflow Curadoria Web]
    F -->|Particionamento Geográfico (Artur Kohara)| G(HDF5s dataset_treino.h5 e dataset_val.h5)
    G -->|RAM Disk Loader (Wallysson)| H[Extração /dev/shm]
    H -->|Treinamento YOLOv11m (Célio Eduardo)| I[Modelo final best.pt]
    I -->|Avaliação & Métricas (Arthur Botelho)| J[Relatório Científico Final]
```

---

## 📂 Código do Pipeline Integrado

Abaixo está o código sequencial contido no notebook integrador para executar todo o fluxo.

### 1. Imports e Setup
```python
import os
import sys
import cv2
import h5py
import numpy as np
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt

# Permitir importação de utils.py
sys.path.append('..')
import utils
```

### 2. Fase 1: Data Pipeline & Slicing (Pessoa 1 - Luidgi Varela)
Divide a imagem GeoTIFF em tiles georreferenciados de 640x640 pixels.
```python
import rasterio
from rasterio.windows import Window

def slice_geotiff_local(input_path, output_dir, tile_size=640):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"GeoTIFF não encontrado: {input_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_tiles = []
    
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        img_width = src.width
        img_height = src.height
        
        for row_off in range(0, img_height - tile_size + 1, tile_size):
            for col_off in range(0, img_width - tile_size + 1, tile_size):
                window = Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size)
                tile_transform = rasterio.windows.transform(window, src.transform)
                
                tile_meta = meta.copy()
                tile_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": tile_transform
                })
                
                tile_name = f"{input_path.stem}_tile_{row_off}_{col_off}.tif"
                output_path = output_dir / tile_name
                
                with rasterio.open(output_path, "w", **tile_meta) as dst:
                    dst.write(src.read(window=window))
                generated_tiles.append(output_path)
    return generated_tiles
```

### 3. Fase 1: Conversão Cromática e HDF5 Bruto (Pessoa 2 - Rafael Lima)
Lê os tiles GeoTIFF, reordena de RGB para BGR (formato OpenCV) e salva no HDF5 compactado.
```python
def executar_pipeline_hdf5(hdf5_path, tile_paths):
    print(f"Criando dataset HDF5 em: {hdf5_path} com {len(tile_paths)} tiles...")
    utils.criar_hdf5_bruto(hdf5_path, tile_paths)
    print("HDF5 Bruto criado com sucesso!")

HDF5_RAW_PATH = "dataset_v1_raw.h5"
```

### 4. Fase 2: Pseudo-Labelling com DeepForest (Pessoa 3 - Vitor Lopes)
Executa a detecção das árvores (com um modelo simulador `FakeDeepForest`) e escreve os bboxes YOLO normalizados no HDF5.
```python
class FakeDeepForest:
    def predict_image(self, image):
        # Simula a detecção de uma árvore no centro do tile
        return [
            {"xmin": 200, "ymin": 200, "xmax": 440, "ymax": 440, "score": 0.92}
        ]

def aplicar_pseudo_labels(hdf5_path):
    print(f"Aplicando pseudo-labels no HDF5: {hdf5_path}...")
    df_model = FakeDeepForest()
    with h5py.File(hdf5_path, "r+") as f:
        images = f["images"]
        for name in images.keys():
            img_data = images[name][:]
            detections = df_model.predict_image(img_data)
            
            boxes = []
            for det in detections:
                dw = 1.0 / 640.0
                dh = 1.0 / 640.0
                x = (det["xmin"] + det["xmax"]) / 2.0 * dw
                y = (det["ymin"] + det["ymax"]) / 2.0 * dh
                w = (det["xmax"] - det["xmin"]) * dw
                h = (det["ymax"] - det["ymin"]) * dh
                boxes.append([0.0, x, y, w, h])
            
            utils.salvar_pseudo_labels(hdf5_path, name, np.array(boxes, dtype=np.float32))
    print("Pseudo-labels gravadas no HDF5!")
```

### 5. Fase 2: Exportação para o Roboflow (Pessoa 4 - Felipe Costa)
Prepara os dados em formato de pasta e compacta em `.zip` para curadoria manual.
```python
def exportar_para_curadoria(hdf5_path, output_dir):
    res = utils.exportar_hdf5_para_roboflow(hdf5_path, output_dir)
    return res

ROBOFLOW_DIR = "roboflow_export"
```

### 6. Fase 2: Curadoria Visual e Controle de Qualidade (Pessoa 5 - Lucas Saad) [MOCK]
Revisão manual das imagens para remover ruídos. Simulamos a etapa de curadoria gerando o arquivo final limpo.
```python
def simular_curadoria(hdf5_in, hdf5_out):
    import shutil
    shutil.copyfile(hdf5_in, hdf5_out)
    print("[MOCK] Curadoria concluída e salva como dataset_limpo.h5!")

HDF5_CLEAN_PATH = "dataset_limpo.h5"
```

### 7. Fase 2: Particionamento Geográfico Asa Norte/Sul (Pessoa 6 - Artur Kohara) [MOCK]
Separa geograficamente o dataset limpo em subconjuntos de treino (80%) e validação (20%).
```python
def simular_particionamento(hdf5_clean, out_train_path, out_val_path):
    with h5py.File(hdf5_clean, 'r') as f_in:
        tile_names = list(f_in['images'].keys())
        split_idx = int(len(tile_names) * 0.8)
        
        train_tiles = tile_names[:split_idx]
        val_tiles = tile_names[split_idx:]
        
        # Grava treino
        with h5py.File(out_train_path, 'w') as f_tr:
            g_img = f_tr.create_group('images')
            g_lbl = f_tr.create_group('labels')
            for name in train_tiles:
                g_img.create_dataset(name, data=f_in['images'][name][:])
                g_lbl.create_dataset(name, data=f_in['labels'][name][:])
                
        # Grava validação
        with h5py.File(out_val_path, 'w') as f_val:
            g_img = f_val.create_group('images')
            g_lbl = f_val.create_group('labels')
            for name in val_tiles:
                g_img.create_dataset(name, data=f_in['images'][name][:])
                g_lbl.create_dataset(name, data=f_in['labels'][name][:])

HDF5_TRAIN_PATH = "dataset_treino.h5"
HDF5_VAL_PATH = "dataset_val.h5"
```

### 8. Fase 3: Dataloader e RAM Disk (Pessoa 7 - Wallysson) [MOCK]
Extrai as imagens e anotações HDF5 em `/dev/shm` no formato YOLO para alimentar o treino.
```python
def extrair_para_treinamento(hdf5_train, hdf5_val, base_ram_dir):
    # Extração de treino
    train_dir = os.path.join(base_ram_dir, "train")
    utils.extrair_hdf5_para_ram(hdf5_train, train_dir)
    # Extração de val
    val_dir = os.path.join(base_ram_dir, "val")
    utils.extrair_hdf5_para_ram(hdf5_val, val_dir)
    
    # data.yaml
    yaml_content = f"""
path: {base_ram_dir}
train: train/images
val: val/images
names:
  0: tree
"""
    with open(os.path.join(base_ram_dir, "data.yaml"), "w") as yaml_file:
        yaml_file.write(yaml_content.strip())

RAM_DIR = "ram_disk_mock"
```

### 9. Fase 3: Treinamento YOLOv11m (Pessoa 8 - Célio Eduardo) [MOCK]
Simula o fine-tuning congelando as primeiras 10 camadas do backbone do YOLO e rodando épocas de treino.
```python
def simular_treinamento_yolo(data_yaml_path):
    print(f"[MOCK] Inicializando treino YOLOv11m...")
    print("[MOCK] Backbone do YOLOv11m congelado nas primeiras 10 camadas.")
    for epoch in range(1, 4):
        print(f"Epoch {epoch}/100: Box Loss = {0.8/epoch:.4f}, Class Loss = {0.5/epoch:.4f}, mAP50 = {0.4 + 0.15*epoch:.4f}")
    print("[MOCK] Treinamento finalizado. Melhores pesos salvos em 'best.pt'!")
    with open("best.pt", "w") as f:
        f.write("pesos_yolo_mock_data")
```

### 10. Fase 4: Avaliação Estatística e Deteções (Pessoa 9 - Arthur Botelho) [MOCK]
Mapeamento de curvas de Precisão-Revogação e simulação de matriz de confusão.
```python
def simular_avaliacao_e_inferencias(pesos_path):
    recalls = np.linspace(0.0, 1.0, 100)
    precisions = 1.0 - recalls**2
    
    plt.figure(figsize=(6, 4))
    plt.plot(recalls, precisions, label='YOLOv11m (mAP@0.5 = 0.85)')
    plt.title('Curva Precisão-Revogação (MOCK)')
    plt.xlabel('Revogação (Recall)')
    plt.ylabel('Precisão (Precision)')
    plt.legend()
    plt.grid(True)
    plt.show()
```
