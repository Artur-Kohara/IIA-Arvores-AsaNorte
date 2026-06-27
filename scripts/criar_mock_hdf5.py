import argparse
import os

import cv2
import h5py
import numpy as np


def _imagem_mock(indice):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cor = ((40 + indice * 70) % 255, (120 + indice * 50) % 255, (200 + indice * 30) % 255)
    img[:] = cor
    centro = (180 + indice * 120, 220 + indice * 80)
    cv2.circle(img, centro, 55, (30, 180, 40), -1)
    cv2.rectangle(img, (centro[0] - 65, centro[1] - 65), (centro[0] + 65, centro[1] + 65), (0, 0, 255), 3)
    return img


def criar_mock_hdf5(output_path, total_imagens=3, formato_labels="por_tile"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        images = f.create_group("images")
        labels = f.create_group("labels")

        for idx in range(total_imagens):
            images.create_dataset(
                f"tile_{idx}",
                data=_imagem_mock(idx),
                compression="gzip",
                compression_opts=4,
                chunks=(640, 640, 3),
            )

        if formato_labels == "por_tile":
            for idx in range(total_imagens):
                boxes = np.array(
                    [[0, 0.35 + idx * 0.08, 0.40 + idx * 0.06, 0.20, 0.20]],
                    dtype=np.float32,
                )
                labels.create_dataset(f"tile_{idx}", data=boxes, dtype=np.float32)
        else:
            image_ids = np.arange(total_imagens, dtype=np.int32)
            classes = np.zeros(total_imagens, dtype=np.int32)
            bboxes = np.array(
                [[0.35 + idx * 0.08, 0.40 + idx * 0.06, 0.20, 0.20] for idx in range(total_imagens)],
                dtype=np.float32,
            )
            scores = np.full(total_imagens, 0.95, dtype=np.float32)

            labels.create_dataset("image_id", data=image_ids)
            labels.create_dataset("class", data=classes)
            labels.create_dataset("bbox", data=bboxes)
            labels.create_dataset("score", data=scores)


def main():
    parser = argparse.ArgumentParser(
        description="Cria um HDF5 mock compativel com o pipeline HDF5 -> Roboflow."
    )
    parser.add_argument(
        "--output",
        default="data/dataset_v1_raw_mock.h5",
        help="Caminho do HDF5 mock de saida.",
    )
    parser.add_argument(
        "--total-imagens",
        type=int,
        default=3,
        help="Quantidade de tiles mock a gerar.",
    )
    parser.add_argument(
        "--formato-labels",
        choices=("por_tile", "agregado"),
        default="por_tile",
        help="Formato de labels: por_tile e o contrato original; agregado simula o notebook feat/vitor.",
    )

    args = parser.parse_args()
    criar_mock_hdf5(args.output, args.total_imagens, args.formato_labels)

    print(f"HDF5 mock criado em: {args.output}")
    print(f"Total de imagens: {args.total_imagens}")
    print(f"Formato de labels: {args.formato_labels}")


if __name__ == "__main__":
    main()
