# Card 2.1 - Exportacao HDF5 para Roboflow

Responsavel: Pessoa 4, Felipe Costa.

Esta etapa recebe um HDF5 com imagens e pseudo-labels e gera uma estrutura YOLO que pode ser importada no Roboflow para curadoria manual.

## Entrada Esperada

O arquivo HDF5 real deve vir do Google Drive do projeto, na pasta:

```text
/02_Datasets_HDF5/
```

Baixe o arquivo localmente para uma pasta ignorada pelo Git, por exemplo:

```text
data/dataset_v1_raw.h5
```

O exportador aceita dois formatos de labels encontrados no projeto.

Formato original por tile:

```text
images/tile_0
labels/tile_0
```

em que `labels/tile_0` e uma matriz `Nx5`:

```text
class_id x_center y_center width height
```

Formato agregado usado no notebook `feat/vitor`:

```text
images/tile_0
labels/image_id
labels/class
labels/bbox
labels/score
```

## Exportar Dataset Real

Na raiz do projeto:

```bash
source .venv/bin/activate
python scripts/exportar_roboflow.py --hdf5 data/dataset_v1_raw.h5 --output roboflow_export
```

Saida gerada:

```text
roboflow_export/
├── images/
├── labels/
├── data.yaml
├── classes.txt
└── roboflow_upload.zip
```

## Upload no Roboflow

Use o Roboflow normal, nao o Roboflow Rapid:

```text
https://app.roboflow.com
```

Crie um projeto com:

```text
Project Type: Object Detection
Tool: Traditional
Annotation Group / Classe: tree
```

Na tela `Upload Data`, use uma destas opcoes:

1. Arrastar `roboflow_export/roboflow_upload.zip`.
2. Clicar em `Select Folder` e selecionar uma pasta contendo imagens `.jpg` e labels `.txt`.
3. Clicar em `Select File(s)` e selecionar juntos os `.jpg`, `.txt`, `classes.txt` e `data.yaml`.

Se o ZIP nao for aceito pela interface, gere uma pasta flat para upload:

```bash
mkdir -p roboflow_upload_folder
cp roboflow_export/images/*.jpg roboflow_upload_folder/
cp roboflow_export/labels/*.txt roboflow_upload_folder/
cp roboflow_export/classes.txt roboflow_upload_folder/
cp roboflow_export/data.yaml roboflow_upload_folder/
```

Depois use `Select Folder` na pasta:

```text
roboflow_upload_folder/
```

## Como Validar

Depois do upload, a tela deve mostrar as imagens como anotadas. Para um mock com 3 imagens, por exemplo:

```text
All Images: 3
Annotated: 3
Not Annotated: 0
```

Abra uma imagem no Roboflow e confirme que existe uma bounding box com classe `tree`.

## Mock Local

Enquanto o HDF5 real nao existir localmente, gere um HDF5 de teste compativel com o contrato original:

```bash
source .venv/bin/activate
python scripts/criar_mock_hdf5.py --output data/dataset_v1_raw_mock.h5 --formato-labels por_tile
python scripts/exportar_roboflow.py --hdf5 data/dataset_v1_raw_mock.h5 --output roboflow_export
```

Para simular o formato agregado da branch `feat/vitor`:

```bash
python scripts/criar_mock_hdf5.py --output data/dataset_v1_raw_mock_vitor.h5 --formato-labels agregado
python scripts/exportar_roboflow.py --hdf5 data/dataset_v1_raw_mock_vitor.h5 --output roboflow_export_vitor
```

O mock serve apenas para validar a integracao HDF5 -> YOLO -> Roboflow. Ele nao substitui o dataset real.

## Arquivos Implementados

```text
utils.py
scripts/exportar_roboflow.py
scripts/criar_mock_hdf5.py
tests/test_hdf5.py
```

## Testes

Rode:

```bash
source .venv/bin/activate
python -m pytest -q
```

O teste relevante do Card 2.1 valida:

- criacao de imagens em `images/`;
- criacao de labels YOLO em `labels/`;
- geracao de `data.yaml`;
- geracao de `classes.txt`;
- geracao de `roboflow_upload.zip`;
- compatibilidade com labels por tile e labels agregadas.
