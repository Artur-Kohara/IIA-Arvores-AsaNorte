# IIA-Arvores-AsaNorte

Projeto 2 da disciplina de Introducao a Inteligencia Artificial do Departamento de Ciencia da Computacao (CIC) da Universidade de Brasilia (UnB).

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
