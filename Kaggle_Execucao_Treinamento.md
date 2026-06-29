# 🚀 Guia de Execução e Treinamento no Kaggle

Este guia orienta de forma estruturada como rodar o treinamento real do **YOLOv11m** utilizando GPUs gratuitas no **Kaggle** a partir do `notebooks/notebook_integrador.ipynb`.

---

## 📋 1. Pré-requisitos e Arquivos Necessários

Como os dados brutos e os datasets HDF5 (`dataset_treino.h5`, `dataset_val.h5`) são muito pesados, o download é feito direto do Google Drive para o Kaggle usando `gdown`. Portanto, você **só precisa enviar os arquivos de código** para o Kaggle.

### Arquivos que você deve enviar:
1. **`notebooks/notebook_integrador.ipynb`** (O notebook integrador adaptado).
2. **`utils.py`** (As funções de I/O em HDF5 e manipulação cromática/RAM Disk).

> [!IMPORTANT]
> **Manutenção da Estrutura de Pastas:**
> O notebook importa o `utils.py` subindo um nível de diretório (`sys.path.append('..')`).
> Para não quebrar essa importação, mantenha a mesma estrutura no ambiente de trabalho do Kaggle ou coloque ambos no mesmo diretório e ajuste a linha de `sys.path.append` no notebook.

---

## 📦 2. Métodos de Envio para o Kaggle

### Método A: Upload via Arquivo ZIP (Recomendado)
Este método preserva a estrutura exata do repositório.

1. Compacte os arquivos principais do seu projeto em um arquivo `.zip` (ex: `iia_projeto.zip`). **Não inclua** pastas locais `data/`, `.git/`, `.venv/` ou `__pycache__/`.
   - O ZIP deve conter a pasta `notebooks/` com o `notebook_integrador.ipynb` e o arquivo `utils.py` na raiz do ZIP.
2. Acesse o [Kaggle](https://www.kaggle.com).
3. Vá em **Create** -> **New Notebook**.
4. No menu superior esquerdo, clique em **File** -> **Upload data**.
5. Faça o upload do arquivo `iia_projeto.zip`. Ele será descompactado na pasta `/kaggle/input/` de leitura.
6. Copie os arquivos da pasta descompactada para a pasta de trabalho executável `/kaggle/working/` rodando a seguinte célula de código no topo do notebook:
   ```python
   import shutil
   import os
   # Copia tudo para o diretório de execução para permitir importações e escritas de arquivos
   shutil.copytree("/kaggle/input/iia-projeto-zip-name", "/kaggle/working", dirs_exist_ok=True)
   os.chdir("/kaggle/working/notebooks")
   ```

### Método B: Upload Direto
Caso queira subir os arquivos um a um de forma rápida.

1. Crie um **New Notebook** no Kaggle.
2. Na barra lateral direita do Kaggle, clique em **File** -> **Upload File** para subir o `utils.py`.
3. Copie as células do `notebook_integrador.ipynb` local e cole no notebook do Kaggle (ou use a opção de importar o arquivo `.ipynb` diretamente pelo menu **File** -> **Import Notebook**).
4. Como os arquivos estarão todos na raiz de `/kaggle/working/`, ajuste o import do `utils.py` na primeira célula do notebook de:
   ```python
   sys.path.append('..')
   ```
   para:
   ```python
   sys.path.append('.')
   ```

---

## ⚙️ 3. Configurações Críticas no Kaggle (Antes de Executar)

Na barra lateral direita do editor de Notebooks do Kaggle, você **obrigatoriamente** deve configurar:

1. **Accelerator (Acelerador de Hardware):**
   - Mude de **None** para **GPU T4 x2** (ou **GPU P100**). O YOLOv11m exige GPU para o fine-tuning em tempo viável.
2. **Internet:**
   - Ative a chave **Internet on**. Isso é **crucial** para permitir que o `gdown` acesse o Google Drive e para o YOLO baixar os pesos pré-treinados iniciais (`yolo11m.pt`).

---

## 🏃 4. Passos para Execução do Treinamento

1. Abra o notebook no Kaggle.
2. Na primeira célula de código, certifique-se de configurar a variável para rodar o pipeline real:
   ```python
   EXECUTION_MODE = "KAGGLE_REAL"
   ```
3. Execute todas as células em sequência (**Run All**).
4. **O que acontecerá na célula de execução final:**
   - **Download:** O `gdown` fará o download da pasta `yolo_project_iia` do Drive para a pasta `/kaggle/working/yolo_project_iia`.
   - **Extração na RAM (Dataloader):** As imagens dos HDF5s serão descompactadas instantaneamente para `/dev/shm/yolo_dataset` (RAM do Kaggle) estruturada em `train/` e `val/`.
   - **Treino:** O YOLOv11m começará o treinamento real usando a GPU por 100 épocas.
   - **Validação:** Ao término, a melhor época será validada no conjunto `val` e exibirá inferências reais anotadas com copas de árvores detectadas.

---

## 💾 5. Como Baixar os Resultados e os Pesos

Ao fim do treinamento, a biblioteca `ultralytics` salva os pesos, estatísticas e gráficos na pasta:
`/kaggle/working/notebooks/runs/detect/train/`

1. Os melhores pesos para usar em produção estarão no arquivo:
   `runs/detect/train/weights/best.pt`
2. Você pode baixar os arquivos clicando na barra lateral direita em **Data** -> **Output** -> localize `best.pt` -> clique nos três pontinhos e selecione **Download**.
3. Como alternativa de backup rápido, você pode zipar a pasta de resultados para baixar tudo de uma vez:
   ```python
   import shutil
   shutil.make_archive('resultados_yolo', 'zip', 'runs/detect/train')
   ```
   E baixar o arquivo `resultados_yolo.zip` gerado na raiz.
