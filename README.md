# IIA-Arvores-AsaNorte

Projeto 2 da disciplina de Introdução à Inteligência Artificial do Departamento de Ciência da Computação (CIC) da Universidade de Brasília (UnB).

## Fase 1 - Data pipeline

O notebook `notebooks/01_data_pipeline.ipynb` implementa a etapa da Pessoa 1 para as ortofotos da Asa Norte:

- leitura de ortofotos GeoTIFF com `rasterio`;
- tentativa opcional de baixar multiplos recortes/setores da Asa Norte via WMS;
- fatiamento em imagens `640x640` usando `rasterio.windows.Window`;
- preservacao dos metadados geograficos dos tiles;
- validacao das dimensoes dos tiles gerados;
- contagem e visualizacao de exemplos.

O notebook nao implementa DeepForest, YOLO, treinamento, HDF5, Roboflow, Kaggle ou conversao para JPEG.

## Como executar

1. Instale as dependencias principais:

```bash
pip install rasterio matplotlib numpy jupyter
```

Para usar a secao opcional de download via WMS, instale tambem:

```bash
pip install requests
```

2. Baixe as ortofotos GeoTIFF da Asa Norte no Geoportal IDE-DF, conforme indicado no PDF do projeto, ou use a secao opcional `Download de recorte via WMS` do notebook para tentar gerar varios recortes menores em `data/raw/asa_norte_setor_*.tif`. A rotina WMS nao sobrescreve um arquivo existente por padrao se ele ja tiver as dimensoes esperadas.

3. Coloque os arquivos `.tif` em:

```text
data/raw/
```

Exemplo:

```text
data/raw/asa_norte_setor_01.tif
data/raw/asa_norte_setor_02.tif
data/raw/asa_norte_setor_03.tif
```

4. Abra o notebook:

```bash
jupyter notebook notebooks/01_data_pipeline.ipynb
```

5. No notebook, ajuste a variavel `INPUT_GEOTIFF` para o nome da ortofoto baixada e execute as celulas.

Cada ortofoto/setor gera sua propria pasta de tiles:

```text
data/raw/asa_norte_setor_01.tif
-> data/tiles/asa_norte_setor_01/

data/raw/asa_norte_setor_02.tif
-> data/tiles/asa_norte_setor_02/

data/raw/asa_norte_setor_03.tif
-> data/tiles/asa_norte_setor_03/
```

O notebook pode processar todos os setores em sequencia. A validacao percorre cada pasta de saida e verifica dimensoes, CRS e transform georreferenciado dos tiles.

Os arquivos em `data/raw/` e `data/tiles/` sao dados locais gerados ou baixados para execucao do pipeline. Eles podem ser recriados pelo notebook e nao devem ser incluidos no commit da entrega.

## Estrutura sugerida

```text
IIA-Arvores-AsaNorte/
├── notebooks/
│   └── 01_data_pipeline.ipynb
├── data/
│   ├── raw/                       # GeoTIFFs originais da Asa Norte
│   │   ├── asa_norte_setor_01.tif
│   │   ├── asa_norte_setor_02.tif
│   │   ├── asa_norte_setor_03.tif
│   │   └── ...
│   └── tiles/                     # Tiles GeoTIFF 640x640 por ortofoto
│       ├── asa_norte_setor_01/
│       ├── asa_norte_setor_02/
│       ├── asa_norte_setor_03/
│       └── ...
└── README.md
```

## Card 2.1 - Exportação para Roboflow

Documentação completa: [docs/card_2_1_roboflow.md](docs/card_2_1_roboflow.md)

A Pessoa 4 exporta o HDF5 com pseudo-labels para uma estrutura YOLO pronta para upload no Roboflow:

```bash
source .venv/bin/activate
python scripts/exportar_roboflow.py --hdf5 dataset_v1_raw.h5 --output roboflow_export
```

Saída gerada:

```text
roboflow_export/
├── images/
├── labels/
├── data.yaml
├── classes.txt
└── roboflow_upload.zip
```

O arquivo `roboflow_export/roboflow_upload.zip` deve ser enviado para um projeto Roboflow de Object Detection com a classe `tree`. Depois do upload, a Pessoa 4 compartilha o workspace com as Pessoas 5 e 6 para revisão manual das caixas e organização da divisão do dataset.

### Mock local para teste de integração

Embora o HDF5 real não esteja disponível no Drive, é possível gerar um HDF5 pequeno de teste:

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

Esses arquivos mock servem apenas para validar a integração do exportador e o upload no Roboflow. O dataset real deve vir da pasta do Google Drive em `/02_Datasets_HDF5/`.
