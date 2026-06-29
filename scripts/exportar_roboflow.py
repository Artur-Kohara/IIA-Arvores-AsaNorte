import argparse
import os
import sys


RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_REPO not in sys.path:
    sys.path.insert(0, RAIZ_REPO)

import utils


def main():
    parser = argparse.ArgumentParser(
        description="Exporta um HDF5 do projeto para estrutura YOLO pronta para upload no Roboflow."
    )
    parser.add_argument(
        "--hdf5",
        required=True,
        help="Caminho do arquivo HDF5 com grupos images/ e labels/.",
    )
    parser.add_argument(
        "--output",
        default="roboflow_export",
        help="Pasta de saida da exportacao. Padrao: roboflow_export.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.hdf5):
        print(f"Erro: HDF5 nao encontrado: {args.hdf5}", file=sys.stderr)
        print(
            "Baixe o arquivo .h5 do Google Drive ou informe o caminho completo com --hdf5.",
            file=sys.stderr,
        )
        print(
            "Exemplo: python scripts/exportar_roboflow.py --hdf5 data/dataset_v1_raw.h5 --output roboflow_export",
            file=sys.stderr,
        )
        return 1

    print(f"Lendo HDF5: {args.hdf5}")
    print(f"Exportando para: {args.output}")

    resumo = utils.exportar_hdf5_para_roboflow(args.hdf5, args.output)

    print("Exportacao finalizada.")
    print(f"Imagens exportadas: {resumo['imagens']}")
    print(f"Bounding boxes exportadas: {resumo['labels']}")
    print(f"ZIP para upload no Roboflow: {resumo['zip_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
