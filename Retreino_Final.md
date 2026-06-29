# 🔄 Fluxo de Curadoria e Retreino Final (Fase 2 e 3)

Este documento detalha o fluxo de trabalho para a **curadoria manual dos dados, divisão geográfica e o retreino final do modelo YOLOv11m no Kaggle**, garantindo a preservação das versões anteriores dos dados.

---

## 📂 Organização no Google Drive Compartilhado

Para não sobrescrever nem alterar os dados gerados nas fases passadas, estruturamos uma nova pasta dedicada para a etapa de curadoria e retreino.

### Arquivo da Primeira Rodada (Bruta)
O arquivo compactado com as estatísticas, pesos e curvas de perda do primeiro treino (sem curadoria) foi salvo na raiz do Google Drive com o nome:
*   🔗 `/resultados_treino_sem_curadoria.zip` (Usado para comparação final de performance).

### Nova Estrutura de Pastas no Drive:
```text
[ Google Drive - Pasta do Projeto ]
├── /01_GeoTIFF_Original/          <-- (Anterior) Apenas leitura dos TIFFs originais.
├── /02_Datasets_HDF5/             <-- (Anterior) Contém o 'dataset_v1_raw.h5' bruto usado como insumo.
├── /03_Zips_Curadoria/            <-- (Anterior) Metadados e zips das fases passadas.
└── /04_Curadoria_e_Retreino/      <-- [NOVA PASTA] Diretório de trabalho do retreino.
    ├── dataset_treino.h5          <-- Dataset de treino final revisado.
    ├── dataset_val.h5             <-- Dataset de validação final revisado.
    └── data.yaml                  <-- Arquivo de configuração YOLO para o Kaggle.
```

---

## 👥 1. Distribuição de Tarefas por Integrante

### 🔵 Pessoa 4 - Felipe Costa (Exportação para o Roboflow)
* **Insumo Utilizado:** O arquivo `/02_Datasets_HDF5/dataset_v1_raw.h5` da pasta antiga (leitura).
* **Ação:**
  1. Baixe o arquivo `dataset_v1_raw.h5` do Google Drive para a sua máquina local.
  2. Execute o exportador na raiz do repositório:
     ```bash
     python scripts/exportar_roboflow.py --hdf5 dataset_v1_raw.h5 --output roboflow_export
     ```
  3. Envie o arquivo gerado em `roboflow_export/roboflow_upload.zip` para o projeto no Roboflow.
  4. Adicione a **Pessoa 5 (Lucas Saad)** ao projeto do Roboflow para início da limpeza.

### 🟡 Pessoa 5 - Lucas Saad (Curadoria Visual Web)
* **Insumo Utilizado:** O projeto de anotação criado pelo Felipe no Roboflow.
* **Ação:**
  1. **Limpeza Visual:** Remova as caixas delimitadoras sobre falsos positivos (carros, sombras, asfalto, telhados).
  2. **Inclusão:** Adicione marcações de árvores omitidas pelo DeepForest (falsos negativos).
  3. **Ajuste:** Redimensione caixas que não cobrem perfeitamente a copa das árvores.
  4. Ao finalizar, exporte o dataset revisado na plataforma no formato **YOLOv8** (como arquivo ZIP) e entregue para a **Pessoa 6 (Artur Kohara)**.

### 🔵 Pessoa 6 - Artur Kohara Guerra (Divisão Geográfica e Novos HDF5s)
* **Insumos Utilizados:** Os arquivos `.jpg` e `.txt` curados exportados do Roboflow e os GeoTIFFs originais de `/01_GeoTIFF_Original/` (para checagem de coordenadas dos setores).
* **Ação:**
  1. Divida as imagens curadas de forma geográfica (**80% treino / 20% validação**) de modo que fatias de uma mesma região não vazem de um conjunto para o outro.
  2. Crie os novos arquivos HDF5 limpos:
     - `dataset_treino.h5`
     - `dataset_val.h5`
  3. Atualize o arquivo `data.yaml` para mapear a classe única (`0: tree`).
  4. Salve esses três arquivos finais **exclusivamente dentro da nova pasta** do Google Drive:
     - 📁 **`/04_Curadoria_e_Retreino/`**

### 🟢 Pessoas 7 e 8 - Wallysson e Célio Eduardo (Retreino Real no Kaggle)
* **Insumos Utilizados:** Os arquivos novos contidos em `/04_Curadoria_e_Retreino/` (baixados automaticamente na GPU).
* **Ação:**
  1. Abra o notebook integrador no Kaggle com a aceleração de GPU ativa.
  2. Altere o endereço de download do Google Drive (ou aponte a busca de IDs de arquivo do `gdown`) para buscar os arquivos atualizados da nova pasta `/04_Curadoria_e_Retreino/`.
  3. Mude a variável da primeira célula do notebook para rodar o treino real:
     ```python
     EXECUTION_MODE = "KAGGLE_REAL"
     ```
  4. Rode todas as células. O notebook descompactará os arquivos na RAM do Kaggle (`/dev/shm/yolo_dataset`), executará o fine-tuning real de 100 épocas do YOLOv11m e exibirá as métricas finais.
  5. Rode a célula de exportação final e baixe o ZIP `resultados_treino_curado.zip` gerado na pasta `/kaggle/working/`.

### 🟢 Pessoa 9 - Arthur Botelho (Validação e Comparação)
* **Insumos Utilizados:** O arquivo `/resultados_treino_sem_curadoria.zip` (na raiz do Drive) e o novo `resultados_treino_curado.zip`.
* **Ação:**
  1. Descompacte os dois resultados de treino.
  2. Compare as curvas de F1-Score, precisão, recall e perda (loss) de ambos os experimentos.
  3. Confeccione o relatório de 5 páginas destacando o impacto crucial da curadoria visual na melhoria das métricas e mitigação de falsos positivos do YOLOv11m.
