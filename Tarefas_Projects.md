# Tarefas Detalhadas e Quadro do GitHub Projects (Com Foco em TDD e HDF5)

Este documento estabelece o mapeamento final de cartões de tarefas para o **GitHub Projects**, definindo as responsabilidades de cada um dos 9 integrantes, divididos em seus respectivos núcleos. 

Também detalha as estratégias técnicas acordadas no Brainstorm:
1. **O Guia do HDF5:** Como criar, validar e ler o dataset unificado.
2. **O Guia do Roboflow:** Como o Núcleo 2 fará a curadoria manual das imagens.
3. **A Infraestrutura de Testes (Pytest + CI):** Como garantir a qualidade do código antes de cada mesclagem.

---

## 🗄️ 1. O GUIA TÉCNICO DO HDF5: Uso, Validação e Manipulação

O arquivo HDF5 (`.h5`) armazena grandes volumes de tensores de forma compactada em um único arquivo binário. Isso resolve o problema de transferir milhares de JPEGs e arquivos de texto individuais na nuvem.

### Por que estamos usando o HDF5?
* **Facilidade de Transporte:** Em vez de fazer download/upload de 20.000 imagens avulsas de/para o Google Drive (o que geraria lentidão e estouro de inodes), trafegamos um arquivo único.
* **Leitura Direta na RAM:** No Kaggle, lemos o HDF5 e extraímos os dados diretamente no RAM disk (`/dev/shm`), evitando gargalos de I/O de disco durante o treino.

---

### Como manipular o HDF5 via Código (Contrato entre Núcleos)

#### 📝 Gravação do HDF5 Inicial (Núcleo 1 - Pessoa 2)
```python
import h5py
import numpy as np

def criar_hdf5_bruto(hdf5_path, image_paths):
    with h5py.File(hdf5_path, 'w') as f:
        # Criamos dois subgrupos principais
        grupo_images = f.create_group('images')
        grupo_labels = f.create_group('labels') # Vazio inicialmente
        
        for idx, img_path in enumerate(image_paths):
            # Carrega a imagem e garante tamanho 640x640x3 do tipo uint8
            img_data = carregar_imagem_jpg(img_path) 
            
            # Salva no HDF5 com compressão gzip balanceada (nível 4)
            grupo_images.create_dataset(
                name=f"tile_{idx}",
                data=img_data,
                compression="gzip",
                compression_opts=4,
                chunks=(640, 640, 3) # Otimiza leitura individual de cada imagem
            )
            
            # Cria dataset de label correspondente vazio (classe -1 ou vazio)
            # Formato: matriz Nx5 para caixas YOLO [class_id, x_center, y_center, w, h]
            grupo_labels.create_dataset(
                name=f"tile_{idx}",
                data=np.empty((0, 5), dtype=np.float32)
            )
```

#### 🏷️ Atualização das Pseudo-Labels no HDF5 (Núcleo 1 - Pessoa 3)
```python
def salvar_pseudo_labels(hdf5_path, tile_name, boxes):
    # boxes: numpy array shape (N, 5) com as coordenadas normalizadas
    with h5py.File(hdf5_path, 'r+') as f: # Modo leitura/escrita
        # Deleta a label vazia anterior e grava a nova
        del f['labels'][tile_name]
        f['labels'].create_dataset(name=tile_name, data=boxes, dtype=np.float32)
```

#### 📂 Extração para o RAM Disk do Kaggle (Núcleo 3 - Pessoa 7)
```python
import os
import cv2

def extrair_hdf5_para_ram(hdf5_path, output_dir="/dev/shm/dataset"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        
        for name in images.keys():
            # 1. Recupera os pixels da imagem e salva como JPG na RAM
            img_array = images[name][:]
            cv2.imwrite(f"{output_dir}/images/{name}.jpg", img_array)
            
            # 2. Recupera os boxes e salva em TXT formato YOLO na RAM
            box_array = labels[name][:]
            txt_path = f"{output_dir}/labels/{name}.txt"
            with open(txt_path, 'w') as txt_file:
                for box in box_array:
                    # Formato YOLO: class_id x_center y_center width height
                    txt_file.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
```

---

## 🎨 2. O GUIA DO ROBOFLOW: Curadoria Prática e Visual (Núcleo 2)

Como a equipe nunca utilizou o Roboflow, adotaremos um fluxo simplificado para revisão de dados sem complicar o pipeline de programação:

### Por que escolher o Roboflow?
* Oferece uma interface web onde você clica e arrasta para mover, deletar ou criar novas caixas delimitadoras sobre as imagens.
* Permite dividir o trabalho de revisão visual igualmente entre os 3 membros do Núcleo 2 (Pessoas 4, 5 e 6).

### Passo a Passo de Execução:
1. **Pessoa 4 (Configuração):** A partir do arquivo `dataset.h5` inicial (gerado com pseudo-labels), roda um script local simples que extrai as imagens e labels em pastas locais temporárias e as comprime em um arquivo `.zip`.
2. **Criação do Projeto:** A Pessoa 4 cria uma conta gratuita no [Roboflow](https://roboflow.com), cria um projeto de "Object Detection" (classe: `tree`) e faz o upload desse `.zip`. O Roboflow lerá as anotações do DeepForest automaticamente.
3. **Distribuição das Imagens:** No painel do Roboflow, divide o dataset em 3 partes e designa cada parte para um membro do Núcleo 2 revisar.
4. **Limpeza Visual (Pessoas 4, 5 e 6):** Cada integrante abre o navegador, inspeciona suas imagens atribuídas e:
   - Clica nas caixas erradas (ex: caixas em telhados de prédios) e aperta `Delete`.
   - Desenha caixas retangulares nas copas das árvores que o modelo pré-treinado do DeepForest deixou passar.
5. **Exportação:** Terminada a revisão, a Pessoa 6 exporta o dataset no formato **YOLOv8/YOLOv11** (gerando um novo link de download/zip).
6. **Consolidação HDF5 Limpo:** A Pessoa 6 baixa esse zip no Drive e roda o script para recompactar os dados curados nos arquivos finais `dataset_treino.h5` e `dataset_val.h5` no Drive.

---

## 🧪 3. ESTRUTURA DE TDD: Pytest e Integração Contínua (CI)

Adotaremos o framework **`pytest`** pela simplicidade de sintaxe e clareza dos relatórios de erros. A validação das tarefas no quadro do GitHub dependerá do sucesso dos testes.

### Arquitetura de Pastas do Repositório:
```text
projeto2-iia/
├── .github/
│   └── workflows/
│       └── tests.yml             <-- GitHub Actions (Executa testes automáticos em PRs)
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_pseudo_labelling.ipynb
│   ├── 03_yolo_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── notebook_integrador.ipynb <-- Notebook unificado integrando todo o pipeline
├── tests/
│   ├── test_slicing.py           <-- Valida scripts da Pessoa 1
│   ├── test_chromatic.py         <-- Valida scripts da Pessoa 2
│   ├── test_hdf5.py              <-- Valida a integridade do HDF5 (Pessoas 2, 3 e 6)
│   ├── test_yolo_format.py       <-- Valida conversão YOLO (Pessoa 3)
│   └── test_ram_disk.py          <-- Valida extração em RAM (Pessoa 7)
├── notebook_integrador.md        <-- Walkthrough e documentação do notebook integrador
├── utils.py                      <-- Contém toda a lógica codificada do projeto
└── requirements.txt
```

---

### Configuração do GitHub Actions (CI)
Para automatizar os testes em cada envio de código (Pull Request para a branch `dev`), a **Pessoa 9** criará o arquivo `.github/workflows/tests.yml` com a seguinte instrução:

```yaml
name: Test Suite CI

on:
  pull_request:
    branches: [ dev ]
  push:
    branches: [ dev ]

jobs:
  run-tests:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Código
      uses: actions/checkout@v3

    - name: Configurar Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Instalar Dependências
      run: |
        python -m pip install --upgrade pip
        pip install pytest h5py numpy opencv-python rasterio shapely

    - name: Executar Pytest
      run: |
        pytest tests/
```

---

## 🚀 4. A Estratégia do Notebook Integrador (Foco em Integração Contínua)

Para mitigar problemas de acoplamento e garantir que os entregáveis de cada fase converjam sem quebrar o fluxo geral, adotamos o [notebook_integrador.ipynb](file:///C:/Users/Eduardo/Documents/GitHub/IIA-Arvores-AsaNorte/notebooks/notebook_integrador.ipynb) na branch `dev` (cujo walkthrough e documentação estão em [notebook_integrador.md](file:///C:/Users/Eduardo/Documents/GitHub/IIA-Arvores-AsaNorte/notebook_integrador.md)).

### Diretrizes de Trabalho no Integrador:
1. **Mocks Iniciais:** Todas as etapas subsequentes à Fase 2 (curadoria, particionamento, carregamento em RAM, treino e avaliação) começam com **código simulado (mocks)** funcionais no notebook.
2. **Substituição de Mocks:** Quando um integrante finalizar sua tarefa específica, ele deve criar um PR para a branch `dev` e **atualizar a respectiva seção do notebook integrador** com o código real, substituindo o mock correspondente.
3. **Validação do Fluxo:** O integrante deve executar o notebook integrador por completo, garantindo que o seu novo código consuma perfeitamente os dados gerados pela etapa anterior e forneça os insumos corretos para a etapa seguinte.
4. **Alinhamento TDD:** Os testes unitários associados a cada etapa devem ser atualizados e validados com `pytest` localmente antes do merge.

---

## 👥 5. Mapeamento de Cartões do GitHub Projects (Tarefas Individuais)

O quadro do GitHub Projects terá as seguintes colunas:
`Backlog` ➔ `A Fazer` ➔ `Em Progresso` ➔ `QA & Testes` ➔ `Revisão de Código` ➔ `Concluído`

Abaixo estão os cartões que o coordenador deve cadastrar no quadro, especificando assignees e critérios.

---

### 🔵 NÚCLEO 1: ENGENHARIA DE DADOS & PIPELINE

#### 📌 Card 1.1: Download e Fatiamento de Ortofotos GeoTIFF
* **Responsável:** Pessoa 1
* **Dependência:** Nenhuma
* **Notebook correspondente:** `01_data_pipeline.ipynb`
* **Descrição:** 
  Criar rotina de download automatizado de ortofotos do Geoportal do DF no Kaggle. Desenvolver a função `slice_geotiff(input_path, output_dir, tile_size=640)` para fatiar as imagens em 640x640 utilizando a biblioteca `rasterio`. O script deve ignorar bordas menores que o tamanho delimitado.
* **Critérios de Aceitação TDD (Escrever teste em `tests/test_slicing.py`):**
  - O teste deve simular uma leitura de GeoTIFF fictícia.
  - Assertar que o tamanho de saída de cada tile é exatamente `(640, 640)`.
  - Assertar que nenhuma imagem recortada possui dimensões diferentes de 640x640.

---

#### 📌 Card 1.2: Conversão Cromática e Estruturação do HDF5 Bruto
* **Responsável:** Pessoa 2
* **Dependência:** Card 1.1 finalizado
* **Notebook correspondente:** `01_data_pipeline.ipynb`
* **Descrição:** 
  Criar a função para reordenar os canais das imagens GeoTIFF extraídas (de RGB para BGR) e salvá-las no formato JPG. Estruturar a criação do arquivo compilado `dataset.h5` no Google Drive, salvando as imagens fatiadas no grupo `images/` e inicializando o grupo `labels/` correspondente vazio.
* **Critérios de Aceitação TDD (Escrever testes em `tests/test_chromatic.py` e `tests/test_hdf5.py`):**
  - Assertar que a conversão cromática não gera clipping de pixel (valores nulos ou fora da escala `[0, 255]`).
  - Abrir o HDF5 criado e certificar que as dimensões do dataset são correspondentes ao array numpy original.
  - Assertar que a compressão `gzip` e os `chunks` de tamanho unitário estão configurados no HDF5.

---

#### 📌 Card 1.3: Geração de Pseudo-Labels com DeepForest no HDF5
* **Responsável:** Pessoa 3
* **Dependência:** Card 1.2 finalizado
* **Notebook correspondente:** `02_pseudo_labelling.ipynb`
* **Descrição:** 
  Implementar o script do modelo pré-treinado `DeepForest` (RetinaNet) para rodar sobre as imagens contidas no HDF5. Mapear as caixas preditas de formato absoluto `[xmin, ymin, xmax, ymax]` para o formato YOLO `[class_id, x_center_norm, y_center_norm, w_norm, h_norm]` normalizado. Salvar os resultados de volta no HDF5 sob o grupo `/labels/`.
* **Critérios de Aceitação TDD (Escrever teste em `tests/test_yolo_format.py`):**
  - O teste deve receber coordenadas fictícias absolutas e verificar se a conversão resulta em valores normalizados entre `0.0` e `1.0`.
  - Assertar que todos os rótulos de classe gerados são `0` (representando a classe única `tree`).
  - Garantir tratamento correto para imagens vazias (deve gravar matriz vazia de shape `(0, 5)`).

---

### 🟡 NÚCLEO 2: CURADORIA, CONTROLE DE QUALIDADE & DIVISÃO DE DADOS

#### 📌 Card 2.1: Exportação HDF5 e Configuração do Roboflow
* **Responsável:** Pessoa 4
* **Dependência:** Card 1.3 finalizado
* **Descrição:** 
  Baixar o HDF5 inicial do Drive. Desenvolver um utilitário simples em `utils.py` que extraia as imagens e labels em uma pasta local para upload no Roboflow. Criar o workspace colaborativo do grupo no Roboflow, subir os dados e dividir a carga de revisão entre as Pessoas 4, 5 e 6.
* **Critérios de Aceitação:**
  - Workspace do Roboflow criado e compartilhado com os integrantes do Núcleo 2.
  - Imagens com as caixas do DeepForest visíveis na plataforma.

---

#### 📌 Card 2.2: Curadoria Visual e Limpeza de Pseudo-Labels
* **Responsável:** Pessoa 5 (com auxílio de 4 e 6)
* **Dependência:** Card 2.1 finalizado
* **Descrição:** 
  Realizar o trabalho de revisão manual das imagens atribuídas. Excluir caixas com falsos positivos (ex: marcações em carros, piscinas, telhados) e desenhar manualmente as caixas nas árvores omitidas (falsos negativos).
* **Critérios de Aceitação:**
  - 100% das imagens do dataset marcadas como "Revisadas" no Roboflow.
  - Relatório sumário da proporção de caixas apagadas/adicionadas no final do processo para inclusão no PDF de entrega.

---

#### 📌 Card 2.3: Particionamento Geográfico e Consolidação do HDF5 Curado
* **Responsável:** Pessoa 6
* **Dependência:** Card 2.2 finalizado
* **Descrição:** 
  Exportar o dataset curado do Roboflow. Aplicar o particionamento espacial por setores (Asa Norte vs Asa Sul) para separar o dataset em conjuntos de Treino e Validação (garantindo zero vazamento de dados geográfico). Compactar os subconjuntos em dois arquivos HDF5 distintos: `dataset_treino.h5` e `dataset_val.h5`. Gerar o arquivo final `data.yaml` e subir todos na pasta oficial do Google Drive.
* **Critérios de Aceitação TDD (Escrever teste em `tests/test_hdf5.py` na branch correspondente):**
  - O teste deve verificar se nenhuma imagem de treino compartilha as mesmas coordenadas espaciais que as imagens de validação.
  - Assertar que o arquivo `data.yaml` aponta corretamente para os caminhos de RAM Disk que serão montados no Kaggle.

---

### 🟢 NÚCLEO 3: MODELAGEM, TREINAMENTO & AVALIAÇÃO

#### 📌 Card 3.1: Dataloader HDF5 e Extração para RAM Disk no Kaggle
* **Responsável:** Pessoa 7
* **Dependência:** Card 2.3 finalizado
* **Notebook correspondente:** `03_yolo_training.ipynb`
* **Descrição:** 
  Criar a rotina de download automático dos HDF5 curados (`dataset_treino.h5` e `dataset_val.h5`) do Drive para a sessão do Kaggle via `gdown`. Desenvolver a rotina de extração direta dessas imagens/anotações na pasta RAM do contêiner (`/dev/shm` ou `/tmp`) no formato clássico de pastas YOLO.
* **Critérios de Aceitação TDD (Escrever teste em `tests/test_ram_disk.py`):**
  - Medir o tempo de extração do HDF5 para o RAM disk (deve rodar em menos de 30 segundos).
  - Assertar que cada imagem `.jpg` extraída possui um arquivo de rótulo `.txt` correspondente de mesmo nome na pasta paralela.

---

#### 📌 Card 3.2: Otimização de Fine-Tuning do YOLOv11m
* **Responsável:** Pessoa 8
* **Dependência:** Card 3.1 finalizado
* **Notebook correspondente:** `03_yolo_training.ipynb`
* **Descrição:** 
  Codificar a chamada de treinamento do YOLOv11m no notebook Kaggle. Configurar o carregamento dos pesos pré-treinados (`yolov11m.pt`). Implementar a estratégia de congelamento do backbone (`freeze=10` nas primeiras épocas). Ativar e testar data augmentations específicas para dados aéreos (rotação livre, flip vertical/horizontal, mistura mosaic). Ajustar os parâmetros de parada antecipada (`patience=10`).
* **Critérios de Aceitação:**
  - Curva de perda do modelo em convergência suave nas primeiras épocas.
  - Salvamento automatizado dos pesos ótimos (`best.pt`) de volta para a pasta oficial do Google Drive.

---

#### 📌 Card 3.3: Avaliação Estatística, Git Guard e Inferência Visual
* **Responsável:** Pessoa 9
* **Dependência:** Card 3.2 finalizado
* **Notebook correspondente:** `04_evaluation.ipynb`
* **Descrição:** 
  Configurar a automação do Pytest no GitHub Actions. Validar todas as submissões de código do grupo. Criar o notebook de avaliação pós-treinamento, computando métricas como curvas de precisão-revogação, matriz de confusão, F1-score e mapas de detecção. Criar uma função de inferência qualitativa para passar imagens inéditas no modelo final e gerar imagens de controle visual.
* **Critérios de Aceitação:**
  - Pipeline de testes integrado e passando com sucesso no GitHub Actions CI.
  - Gráficos estatísticos gerados no notebook final e salvos como imagens no Drive para inclusão no relatório.
