# IIA-Arvores-AsaNorte
Projeto 2 da disciplina de Introdução à Inteligência Artificial do departamento de Ciência da Computação (CIC) da Universidade de Brasília (UnB)

## Card 2.1 - Exportacao para Roboflow

Documentacao completa: [docs/card_2_1_roboflow.md](docs/card_2_1_roboflow.md)

A Pessoa 4 exporta o HDF5 com pseudo-labels para uma estrutura YOLO pronta para upload no Roboflow:

```bash
source .venv/bin/activate
python scripts/exportar_roboflow.py --hdf5 dataset_v1_raw.h5 --output roboflow_export
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

O arquivo `roboflow_export/roboflow_upload.zip` deve ser enviado para um projeto Roboflow de Object Detection com a classe `tree`. Depois do upload, a Pessoa 4 compartilha o workspace com as Pessoas 5 e 6 para revisao manual das caixas e organizacao da divisao do dataset.

### Mock local para teste de integracao

Enquanto o HDF5 real nao estiver disponivel no Drive, e possivel gerar um HDF5 pequeno de teste:

```bash
source .venv/bin/activate
python scripts/criar_mock_hdf5.py --output data/dataset_v1_raw_mock.h5 --formato-labels por_tile
python scripts/exportar_roboflow.py --hdf5 data/dataset_v1_raw_mock.h5 --output roboflow_export
```

Para simular o formato agregado usado no notebook `feat/vitor`:

```bash
python scripts/criar_mock_hdf5.py --output data/dataset_v1_raw_mock_vitor.h5 --formato-labels agregado
python scripts/exportar_roboflow.py --hdf5 data/dataset_v1_raw_mock_vitor.h5 --output roboflow_export_vitor
```

Esses arquivos mock servem apenas para validar a integracao do exportador e o upload no Roboflow. O dataset real deve vir da pasta do Google Drive em `/02_Datasets_HDF5/`.
