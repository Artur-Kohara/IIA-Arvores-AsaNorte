import os
import zipfile
import cv2
import h5py
import numpy as np

"""
utils.py - Funções utilitárias compartilhadas do projeto projeto2-iia.
Dividido entre as necessidades de pipeline de dados, pseudo-labelling e carregamento no Kaggle.
"""

# =====================================================================
# NÚCLEO 1: ENGENHARIA DE DADOS & PIPELINE
# =====================================================================

def slice_geotiff(input_path, output_dir, tile_size=640):
    """
    Fatia uma ortofoto GeoTIFF em blocos (tiles) de tamanho tile_size x tile_size.
    Implementado por: Pessoa 1
    """
    # TODO: Implementar fatiamento usando a biblioteca rasterio e Window.
    pass

def carregar_imagem_jpg(img_path):
    """
    Carrega imagem do disco.
    """
    # Exemplo simples de leitura
    img = cv2.imread(img_path)
    return img

def _normalizar_uint8(arr):
    """
    Garante que o array esteja em uint8 (0-255). Tiles GeoTIFF podem vir em
    16 bits; neste caso aplicamos uma normalização min-max para a escala de 8 bits.
    """
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    minimo = float(arr.min())
    maximo = float(arr.max())
    if maximo - minimo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - minimo) / (maximo - minimo) * 255.0
    return arr.astype(np.uint8)

def converter_rgb_para_bgr(img):
    """
    Reordena os canais de uma imagem de RGB para BGR (ordem nativa do OpenCV).
    Aceita tanto o formato HWC (H, W, C) quanto o formato CHW (C, H, W) devolvido
    pelo rasterio. Descarta o canal alfa (4ª banda) se presente e garante uint8.
    Implementado por: Pessoa 2

    Retorna array contíguo (H, W, 3) uint8 em ordem BGR.
    """
    # Detecta CHW (ex.: (3, 640, 640)) e transpõe para HWC (640, 640, 3)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))

    # Mantém apenas as 3 primeiras bandas (descarta alfa, se houver)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    # Reordena RGB -> BGR
    img_bgr = img[..., ::-1]
    return np.ascontiguousarray(img_bgr, dtype=np.uint8)

def carregar_tile(tile_path):
    """
    Carrega um tile gerado pelo fatiamento (Pessoa 1) e devolve um array
    (640, 640, 3) uint8 em ordem BGR, pronto para gravar no HDF5.

    - Tiles GeoTIFF (.tif/.tiff): lidos com rasterio (RGB, formato CHW),
      normalizados para uint8 e reordenados para BGR.
    - Tiles já rasterizados (.jpg/.jpeg/.png): lidos com cv2.imread, que já
      devolve BGR — não há reordenação adicional.

    Retorna None se a leitura falhar.
    Implementado por: Pessoa 2
    """
    ext = os.path.splitext(str(tile_path))[1].lower()

    if ext in (".tif", ".tiff"):
        import rasterio  # import lazy: só exigido para tiles GeoTIFF
        with rasterio.open(tile_path) as src:
            arr = src.read()  # (bandas, H, W) em ordem RGB
        arr = _normalizar_uint8(arr)
        return converter_rgb_para_bgr(arr)

    # JPG/PNG (ou fallback): cv2.imread já retorna BGR
    return cv2.imread(str(tile_path))

def criar_hdf5_bruto(hdf5_path, image_paths):
    """
    Reordena canais GeoTIFF e grava no formato compilado HDF5 inicial.
    Implementado por: Pessoa 2
    """
    with h5py.File(hdf5_path, 'w') as f:
        # Criamos dois subgrupos principais
        grupo_images = f.create_group('images')
        grupo_labels = f.create_group('labels') # Vazio inicialmente
        
        for idx, img_path in enumerate(image_paths):
            # Carrega o tile, reordena RGB->BGR e garante 640x640x3 uint8
            img_data = carregar_tile(img_path)

            if img_data is None:
                continue
                
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

def salvar_pseudo_labels(hdf5_path, tile_name, boxes):
    """
    Atualiza as pseudo-labels geradas pelo DeepForest no arquivo HDF5.
    boxes: numpy array shape (N, 5) com as coordenadas normalizadas.
    Implementado por: Pessoa 3
    """
    with h5py.File(hdf5_path, 'r+') as f: # Modo leitura/escrita
        # Deleta a label vazia anterior e grava a nova
        if tile_name in f['labels']:
            del f['labels'][tile_name]
        f['labels'].create_dataset(name=tile_name, data=boxes, dtype=np.float32)


# =====================================================================
# NÚCLEO 2: CURADORIA, CONTROLE DE QUALIDADE & DIVISÃO DE DADOS
# =====================================================================

def _labels_do_tile(labels, tile_name, tile_index):
    """
    Retorna labels YOLO Nx5 para um tile.

    Suporta dois contratos encontrados no projeto:
    - labels/tile_0 -> matriz Nx5 [class_id, x_center, y_center, w, h]
    - labels/{image_id,class,bbox,score} -> datasets agregados por imagem
    """
    if tile_name in labels:
        return labels[tile_name][:]

    campos_agregados = {"image_id", "class", "bbox"}
    if not campos_agregados.issubset(set(labels.keys())):
        return np.empty((0, 5), dtype=np.float32)

    image_ids = labels["image_id"][:]
    classes = labels["class"][:]
    bboxes = labels["bbox"][:]

    linhas = []
    for image_id, class_id, bbox in zip(image_ids, classes, bboxes):
        if isinstance(image_id, bytes):
            image_id = image_id.decode("utf-8")

        corresponde = image_id == tile_index or str(image_id) == tile_name
        if not corresponde:
            continue

        linhas.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    if not linhas:
        return np.empty((0, 5), dtype=np.float32)

    return np.asarray(linhas, dtype=np.float32)

def exportar_hdf5_para_roboflow(hdf5_path, output_dir):
    """
    Extrai as imagens e labels do HDF5 em pastas locais temporárias
    para upload no Roboflow.

    Gera a estrutura:
      output_dir/
        images/*.jpg
        labels/*.txt
        data.yaml
        classes.txt
        roboflow_upload.zip

    O formato dos rótulos é YOLO:
    class_id x_center y_center width height

    Implementado por: Pessoa 4
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    exportados = 0
    labels_exportadas = 0

    with h5py.File(hdf5_path, "r") as f:
        if "images" not in f or "labels" not in f:
            raise ValueError("HDF5 deve conter os grupos 'images' e 'labels'.")

        images = f["images"]
        labels = f["labels"]

        for idx, name in enumerate(sorted(images.keys())):
            img_array = images[name][:]
            image_path = os.path.join(images_dir, f"{name}.jpg")

            if not cv2.imwrite(image_path, img_array):
                raise IOError(f"Falha ao gravar imagem exportada: {image_path}")

            box_array = _labels_do_tile(labels, name, idx)
            label_path = os.path.join(labels_dir, f"{name}.txt")

            with open(label_path, "w") as txt_file:
                for box in box_array:
                    txt_file.write(
                        f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                    )

            exportados += 1
            labels_exportadas += len(box_array)

    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w") as classes_file:
        classes_file.write("tree\n")

    data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(data_yaml_path, "w") as data_yaml:
        data_yaml.write(
            "path: .\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            "  0: tree\n"
        )

    zip_path = os.path.join(output_dir, "roboflow_upload.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(output_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if file_path == zip_path:
                    continue
                arcname = os.path.relpath(file_path, output_dir)
                zip_file.write(file_path, arcname)

    return {
        "imagens": exportados,
        "labels": labels_exportadas,
        "output_dir": output_dir,
        "zip_path": zip_path,
    }


# =====================================================================
# NÚCLEO 3: MODELAGEM, TREINAMENTO & AVALIAÇÃO
# =====================================================================

def extrair_hdf5_para_ram(hdf5_path, output_dir="/dev/shm/dataset"):
    """
    Lê o HDF5 e extrai diretamente no RAM disk (/dev/shm) para treinamento YOLO acelerado.
    Implementado por: Pessoa 7
    """
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
