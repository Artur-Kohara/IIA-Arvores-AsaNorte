# Proposta de Divisão de Trabalho - Revisada e Acoplada

Este plano de trabalho foi reestruturado para superar o caráter puramente linear da proposta original e incorporar a infraestrutura de **armazenamento centralizado no Google Drive** do coordenador, a execução baseada em uma **conta única do Kaggle** (com otimização via HDF5 e RAM disk `/dev/shm`), e a estratégia de **versionamento modular de notebooks** (com outputs limpos via `nbstripout`).

Projetos de Aprendizado de Máquina (ML) e Visão Computacional são altamente acoplados: **bugs no fatiamento invalidam as anotações, anotações de má qualidade impedem a convergência do modelo e métricas ruins exigem revisões no pré-processamento**. 

Para garantir o acoplamento e a colaboração contínua, os 9 integrantes do grupo estão divididos em **3 núcleos integrados**.

---

## ⚠️ AVISO CRÍTICO E POLÍTICA DE USO DO GOOGLE DRIVE

O repositório oficial de dados grandes (imagens e arquivos `.h5`) é:
🔗 [Google Drive - Pasta de Dados do Projeto](https://drive.google.com/drive/folders/11M2qj3BHYNDle173v8JT7gcz6kZtfLVY?usp=sharing)

> [!CAUTION]
> **POLÍTICA DE BACKUP ZERO (DADOS PESADOS):**
> Devido ao grande volume físico das imagens GeoTIFF e dos datasets compactados, **não haverá backup local e nem no GitHub destes arquivos**. O coordenador e o grupo NÃO manterão cópias de segurança fora desta pasta compartilhada. Como todos os membros possuem **acesso total de edição**, a exclusão acidental ou modificação incorreta de arquivos causará a **perda irremediável** do trabalho de fases anteriores.

### 🛡️ Regras Obrigatórias para Todo Integrante:
1. **Exclusão Estritamente Proibida:** Nunca utilize a opção de excluir ou "remover" arquivos na pasta compartilhada. Se julgar que um arquivo é obsoleto, envie uma mensagem no grupo e aguarde a aprovação coletiva antes de deletar.
2. **Versionamento Incrementado:** Nunca substitua/sobrescreva arquivos HDF5 existentes (ex: `dataset.h5`). Sempre salve versões novas adicionando sufixos claros (ex: `dataset_v1_raw.h5`, `dataset_v2_QA_limpo.h5`).
3. **Download Isolado:** Ao rodar experimentos locais ou no Kaggle, baixe os arquivos (ou utilize scripts com `gdown`). Nunca mova os arquivos compartilhados de pasta no Drive para evitar quebras nos caminhos dos scripts dos outros membros.
4. **Organização de Pastas:** Respeite a divisão:
   - `/01_GeoTIFF_Original/` -> Apenas leitura das imagens originais.
   - `/02_Datasets_HDF5/` -> Apenas os arquivos compilados (`.h5`).
   - `/03_Zips_Curadoria/` -> Arquivos de revisão e metadados.

---

## 🗺️ Fluxo de Trabalho Integrado e Acoplado

Para evitar falhas de integração e garantir que o código de cada integrante funcione como insumo direto para o próximo, o desenvolvimento será unificado e testado através de um **único notebook de fluxo completo**: o `notebooks/notebook_integrador.ipynb` (descrito em [notebook_integrador.md](file:///C:/Users/Eduardo/Documents/GitHub/IIA-Arvores-AsaNorte/notebook_integrador.md)) na branch `dev`. 

As partes não implementadas de cada fase serão simuladas com **mocks** temporários para que o fluxo de ponta a ponta execute sem falhas. Cada integrante, ao completar sua tarefa, deve substituir o respectivo mock pelo seu código de produção.

1. **Núcleo 1 (Engenharia de Dados) - Fases 1 e 2:**
   - Baixa GeoTIFFs da IDE-DF, fatia em 640x640, converte TIFF -> JPG (RGB -> BGR) e cria o arquivo compilado `dataset_v1_raw.h5` no Google Drive.
   - Executa o DeepForest nas imagens do HDF5 e insere as pseudo-labels no próprio HDF5.
   - *Seção do Integrador:* Seções 1, 2 e 3 do `notebooks/notebook_integrador.ipynb` e respectivos métodos em [utils.py](file:///C:/Users/Eduardo/Documents/GitHub/IIA-Arvores-AsaNorte/utils.py).

2. **Núcleo 2 (Curadoria & QA) - Fase 2:**
   - Exporta o HDF5 bruto para um `.zip` padrão YOLO, faz o upload no Roboflow para curadoria manual e baixa as anotações revisadas.
   - Divide o dataset geograficamente (Asa Norte/Asa Sul) e cria os datasets consolidados `dataset_treino.h5`, `dataset_val.h5` e `data.yaml`.
   - *Seção do Integrador:* Seções 4, 5 e 6 do `notebooks/notebook_integrador.ipynb` e respectivos métodos em [utils.py](file:///C:/Users/Eduardo/Documents/GitHub/IIA-Arvores-AsaNorte/utils.py).

3. **Núcleo 3 (Modelagem & Treinamento) - Fases 3 e 4:**
   - Baixa os HDF5s curados na sessão única do Kaggle e extrai instantaneamente na RAM (`/dev/shm`).
   - Configura e executa o fine-tuning do YOLOv11m (com congelamento do backbone e aumentos de dados).
   - Avalia estatisticamente (curvas Precision-Recall, matriz de confusão) e plota gráficos.
   - *Seção do Integrador:* Seções 7, 8 e 9 do `notebooks/notebook_integrador.ipynb` e respectivos métodos em [utils.py](file:///C:/Users/Eduardo/Documents/GitHub/IIA-Arvores-AsaNorte/utils.py).


---

## 👥 Divisão de Equipes e Papéis (9 Integrantes)

### 🔵 Núcleo 1: Engenharia de Dados & Geração de Pipeline (Pessoas 1, 2 e 3)
**Foco:** Infraestrutura do dado, fatiamento espacial e geração automatizada de caixas delimitadoras.

*   **Pessoa 1 - Luidgi Varela Carneiro (Coordenador do Pipeline):** Desenvolve o notebook `01_data_pipeline.ipynb`. Implementa o download das ortofotos da IDE-DF e o fatiamento em 640x640 usando `rasterio` e `Window`, garantindo o correto tratamento dos limites espaciais.
*   **Pessoa 2 - Rafael de Lima Pereira (Engenheiro de Formatação):** Desenvolve a conversão eficiente de bandas GeoTIFF RGB para JPEG via OpenCV, organizando as imagens JPG na estrutura inicial do HDF5. Compartilha as funções de conversão e escrita do HDF5 em `utils.py`.
*   **Pessoa 3 - Vitor Caldas Danelon Lopes (Operador DeepForest):** Desenvolve o notebook `02_pseudo_labelling.ipynb`. Instancia o `DeepForest` (RetinaNet) sobre as imagens do HDF5, extrai os boxes em coordenadas absolutas, faz a normalização de escala e grava as anotações formatadas para o YOLO no arquivo HDF5.

---

### 🟡 Núcleo 2: Curadoria, Controle de Qualidade & Divisão de Dados (Pessoas 4, 5 e 6)
**Foco:** Refinamento dos dados, eliminação de ruídos das pseudo-labels e estruturação de validação confiável.

*   **Pessoa 4 - Felipe Costa (Especialista em Anotação):** Configura a plataforma de curadoria (ex: Roboflow ou interface local de inspeção no notebook) para importar as imagens e pseudo-labels do HDF5, coordenando o esforço de limpeza do grupo.
*   **Pessoa 5 - Lucas Saad (Revisor de Qualidade):** Executa a revisão manual detalhada, apagando as caixas absurdas geradas pelo DeepForest (ex: sombras, carros, telhados) e adicionando copas de árvores omitidas (falsos negativos).
*   **Pessoa 6 - Artur Kohara Guerra (Estrategista de Particionamento):** Divide o HDF5 geograficamente por setores (Asa Norte/Asa Sul) para separar o conjunto de treino e validação, prevenindo vazamento de dados espacial. Gera o arquivo final `dataset_limpo.h5` e o arquivo de configuração `data.yaml`, salvando-os de volta no Google Drive.

---

### 🟢 Núcleo 3: Modelagem, Treinamento & Avaliação (Pessoas 7, 8 e 9)
**Foco:** Treinamento do YOLOv11m na conta única do Kaggle, sintonia de hiperparâmetros, avaliação rigorosa e documentação.

*   **Pessoa 7 - Wallysson (Engenheiro de Treinamento):** Cria o notebook `03_yolo_training.ipynb`. Implementa o download de `dataset_limpo.h5` do Google Drive via `gdown`, a extração rápida das imagens/labels para a memória RAM (`/dev/shm` ou `/tmp`) e inicializa a chamada de treino da biblioteca `ultralytics`.
*   **Pessoa 8 - Celio Eduardo (Estrategista de Fine-Tuning & Regularização):** Otimiza o script de treino do YOLOv11m. Configura o congelamento das primeiras camadas do backbone (`freeze=10`) para preservar os extratores primitivos de features e define aumentos de dados adequados (rotações de 90° e espelhamentos verticais/horizontais).
*   **Pessoa 9 - Arthur Menezes Botelho (Avaliador Métrico & Git Guard):** Configura a estrutura geral do repositório, assegura a ativação do filtro `nbstripout` nas máquinas locais, garante a padronização de `utils.py`, cria o notebook `04_evaluation.ipynb` para gerar as métricas finais (curva PR, matriz de confusão) e implementa a inferência qualitativa.

---

## 🧠 Teoria Aplicada à Prática: Decisões Críticas de ML

*   **Tile Size (640x640):** Mantido para coincidir com o tamanho de entrada nativo do YOLOv11m. Slicing direto preserva a escala original de pixels de Brasília (10cm-20cm/pixel), mantendo as copas das árvores com resolução suficiente (30 a 60 pixels de diâmetro) para detecção de borda e textura.
*   **HDF5 para I/O em Memória:** A leitura direta de arquivos HDF5 e extração direta no RAM disk (`/dev/shm`) em tempo de execução elimina leituras repetitivas do lento disco do Kaggle, mantendo a GPU com 100% de uso e poupando tempo da conta única de GPU.
*   **Fine-Tuning Inteligente:** O modelo é inicializado com `yolov11m.pt` (pesos COCO). O extrator de features do backbone é congelado nas primeiras 10 épocas para proteger os pesos contra esquecimento catastrófico gerado por erros iniciais de pseudo-labels do DeepForest.

---

## 📈 Plano de Entregas e Integração do Relatório

O relatório final de 5 páginas é escrito em paralelo à conclusão das fases de desenvolvimento:

*   **Seção 1: Introdução & Metodologia de Coleta (Núcleo 1):** Escrita logo após a finalização do fatiamento e estruturação do HDF5 inicial.
*   **Seção 2: Metodologia de Curadoria e Estatística de Rótulos (Núcleo 2):** Escrita durante a limpeza no Roboflow.
*   **Seção 3: Metodologia de Treino e Hiperparâmetros (Núcleo 3):** Escrita durante as rodadas de execução de testes no Kaggle.
*   **Seção 4: Resultados, Métricas Comparativas e Inferência Visual (Núcleo 3 + Núcleo 2):** Plota as curvas, tabelas comparativas e inferências visuais de controle.
*   **Seção 5: Conclusão & Integração (Todo o grupo):** Revisão e compilação do relatório final sob a coordenação da Pessoa 9.
