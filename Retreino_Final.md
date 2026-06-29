# 🔄 Fluxo de Curadoria e Retreino Final (Fase 2 e 3)

Este documento detalha o fluxo de trabalho para a **curadoria manual dos dados, divisão geográfica e o retreino final do modelo YOLOv11m no Kaggle**.

Este pipeline garante a transição do dataset bruto com pseudo-rótulos ruidosos (gerados pelo DeepForest) para um dataset curado, resultando em um modelo final de alta precisão.

---

## 🗺️ Visão Geral do Fluxo

```mermaid
graph TD
    A[HDF5 Bruto dataset_v1_raw.h5] -->|1. Exportador (Felipe)| B(Pasta YOLO Flat + ZIP)
    B -->|2. Upload| C[Roboflow - Limpeza Manual (Lucas)]
    C -->|3. Exportar YOLOv8/v11| D(Dataset Revisado)
    D -->|4. Particionador Geográfico (Artur)| E(dataset_treino.h5, dataset_val.h5 e data.yaml)
    E -->|5. Google Drive| F[Google Drive Compartilhado]
    F -->|6. Download gdown| G[Kaggle GPU - Treinamento Real (Wallysson/Célio)]
    G -->|7. Pesos Finais| H(best.pt + Métricas Finais)
```

---

## 👥 1. Distribuição de Tarefas por Integrante

### 🔵 Pessoa 4 - Felipe Costa (Exportação para o Roboflow)
* **Objetivo:** Transformar o arquivo HDF5 bruto em uma estrutura de imagens JPG e rótulos TXT para o Roboflow.
* **Passos:**
  1. Baixe o arquivo `dataset_v1_raw.h5` do Google Drive para o seu computador.
  2. Execute o script de exportação na raiz do repositório:
     ```bash
     python scripts/exportar_roboflow.py --hdf5 dataset_v1_raw.h5 --output roboflow_export
     ```
  3. O script gerará automaticamente um arquivo ZIP em `roboflow_export/roboflow_upload.zip`.
  4. Crie um projeto no Roboflow (**Object Detection**, classe única: `tree`) e faça o upload desse arquivo ZIP. A plataforma reconhecerá todas as imagens e os pseudo-rótulos correspondentes.
  5. Compartilhe o acesso ao workspace do projeto no Roboflow com a **Pessoa 5 (Lucas Saad)**.

### 🟡 Pessoa 5 - Lucas Saad (Curadoria e QA Visual)
* **Objetivo:** Eliminar o ruído das anotações automáticas do DeepForest e refinar as caixas delimitadoras.
* **Por que fazer isso diretamente no Roboflow?** O Roboflow fornece uma interface web rápida e colaborativa para arrastar, apagar e desenhar caixas delimitadoras, o que é muito mais eficiente do que tentar editar os dados no HDF5 diretamente.
* **Passos de Curadoria (Interface Web do Roboflow):**
  1. **Remover Falsos Positivos:** Apague as caixas que o DeepForest desenhou incorretamente sobre sombras de edifícios, carros escuros, copas parciais de arbustos pequenos ou telhados.
  2. **Adicionar Falsos Negativos:** Desenhe caixas ao redor de copas de árvores reais que foram ignoradas pelo DeepForest (principalmente árvores sob sombreamento ou próximas a edifícios).
  3. **Ajustar Bounding Boxes:** Redimensione as caixas que ficaram grandes demais ou deslocadas em relação à copa real da árvore.
  4. Ao finalizar a curadoria de todas as imagens, clique em **Generate Version** no Roboflow e exporte o dataset final no formato **YOLOv8** (selecione a opção de baixar como arquivo ZIP).
  5. Entregue o ZIP exportado para a **Pessoa 6 (Artur Kohara)**.

### 🔵 Pessoa 6 - Artur Kohara Guerra (Particionamento Geográfico e Compilação Clean)
* **Objetivo:** Dividir o dataset curado geograficamente para evitar vazamento de dados espacial (data leakage) e gerar os novos arquivos HDF5 limpos.
* **Por que a divisão geográfica?** Se usarmos fatias adjacentes de uma mesma ortofoto no treino e na validação, o modelo decorará a textura do solo. Dividir por setores geográficos (ex: Asa Norte e Asa Sul, ou Setores específicos) garante uma validação robusta.
* **Passos:**
  1. Descompacte o ZIP exportado do Roboflow.
  2. Escreva/execute o script de particionamento (ou integre a lógica no `utils.py`) para ler os arquivos `.jpg` e `.txt` revisados.
  3. Separe **80% dos tiles para treino e 20% para validação** com base no setor ou coordenada geográfica do tile (que consta no nome do arquivo, ex: `asa_norte_setor_01_tile_row_col`).
  4. Compile esses subconjuntos em dois novos arquivos HDF5 estruturados:
     - `dataset_treino.h5`
     - `dataset_val.h5`
  5. Atualize o arquivo `data.yaml` para refletir as configurações da classe única (`0: tree`).
  6. Suba os três arquivos (`dataset_treino.h5`, `dataset_val.h5` e `data.yaml`) na pasta do Google Drive em `/02_Datasets_HDF5/`, **substituindo** ou versionando incrementalmente os arquivos existentes lá.

### 🟢 Pessoas 7 e 8 - Wallysson e Célio Eduardo (Retreino Real no Kaggle)
* **Objetivo:** Executar o fine-tuning definitivo do YOLOv11m com os dados limpos.
* **Passos:**
  1. Abra o notebook integrador no Kaggle com a aceleração de GPU ativada.
  2. Defina a variável na primeira célula:
     ```python
     EXECUTION_MODE = "KAGGLE_REAL"
     ```
  3. Execute o notebook. O notebook irá:
     - Baixar os arquivos curados (`dataset_treino.h5`, `dataset_val.h5` e `data.yaml`) do Drive via `gdown`.
     - Extrair tudo na memória RAM do Kaggle (`/dev/shm/yolo_dataset`).
     - Treinar o YOLOv11m real por 100 épocas.
     - Rodar a validação e gerar as métricas de inferência finais.
  4. Execute a última célula para gerar o ZIP `resultados_treino_curado.zip` e faça o download.

### 🟢 Pessoa 9 - Arthur Botelho (Validação e Comparação)
* **Objetivo:** Consolidar os relatórios e analisar a evolução das métricas.
* **Passos:**
  1. Extraia e compare os gráficos do primeiro treino bruto com o treinamento curado final.
  2. Analise a evolução do `mAP50`, curvas de F1-Score e a redução de falsos positivos na Matriz de Confusão.
  3. Colete os resultados qualitativos (imagens de inferência reais) para estruturar o relatório científico de 5 páginas.
