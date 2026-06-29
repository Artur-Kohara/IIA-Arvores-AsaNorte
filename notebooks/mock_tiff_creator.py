import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_synthetic_tiff(output_path):
    """
    Cria uma imagem GeoTIFF sintética de 1280x1280 pixels com 3 canais de cor
    para fins de teste local do pipeline.
    """
    width = 1280
    height = 1280
    bands = 3
    data = np.random.randint(0, 256, (bands, height, width), dtype=np.uint8)
    transform = from_origin(-47.9, -15.7, 0.1, 0.1)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=np.uint8,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(data)
