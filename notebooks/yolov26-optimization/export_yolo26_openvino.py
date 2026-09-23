import argparse
import shutil
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_ROOT = SCRIPT_DIR / "yolo26_export_model"
CALIBRATION_IMAGES = SCRIPT_DIR / "datasets/coco128/images/train2017"
MODEL_SIZES = {
    "yolo26n-pose": 384,
    "yolo26s-pose": 512,
    "yolo26m-pose": 576,
    "yolo26l-pose": 704,
}


def replace_directory(destination: Path, source: Path):
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def export_fp16(model_name: str, imgsz: int) -> Path:
    model_path = SCRIPT_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    print(f"\n=== Exporting {model_name} at {imgsz}x{imgsz} as FP16 ===")
    default_export_dir = Path(
        YOLO(str(model_path)).export(
            format="openvino",
            imgsz=imgsz,
            dynamic=False,
            half=True,
            end2end=True,
        )
    )
    target_dir = EXPORT_ROOT / model_name / "fp16"
    replace_directory(target_dir, default_export_dir)
    return target_dir


def quantize_int8(model_name: str, imgsz: int, fp16_dir: Path):
    model_xml = fp16_dir / f"{model_name}.xml"
    metadata = fp16_dir / "metadata.yaml"
    image_files = list(CALIBRATION_IMAGES.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(
            f"No calibration images in {CALIBRATION_IMAGES}. "
            "The notebook uses coco128/images/train2017 for NNCF quantization."
        )

    def transform_fn(image_path):
        image = cv2.imread(str(image_path))
        image = LetterBox((imgsz, imgsz))(image=image)
        image = image.transpose(2, 0, 1)
        image = image[::-1]
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(image, 0)

    print(f"=== Quantizing {model_name} as INT8 with {len(image_files)} coco128 images ===")
    quantized_model = nncf.quantize(
        ov.Core().read_model(model_xml),
        nncf.Dataset(image_files, transform_fn),
        preset=nncf.QuantizationPreset.MIXED,
        fast_bias_correction=False,
        ignored_scope=nncf.IgnoredScope(patterns=[".*one2one.*"]),
    )

    target_dir = EXPORT_ROOT / model_name / "int8"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    int8_model_xml = target_dir / f"{model_name}.xml"
    ov.save_model(quantized_model, str(int8_model_xml))
    shutil.copy2(metadata, target_dir / "metadata.yaml")
    print(f"INT8 model: {int8_model_xml}")


def export_fp32(model_name: str, imgsz: int):
    model_path = SCRIPT_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    print(f"\n=== Exporting {model_name} at {imgsz}x{imgsz} as FP32 ===")
    default_export_dir = Path(
        YOLO(str(model_path)).export(
            format="openvino",
            imgsz=imgsz,
            dynamic=False,
            end2end=True,
        )
    )
    replace_directory(EXPORT_ROOT / model_name / "fp32", default_export_dir)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26 OpenVINO models using the notebook conversion pipeline.")
    parser.add_argument("--model", choices=[*MODEL_SIZES, "all"], default="all")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8", "all"], default="all")
    args = parser.parse_args()

    selected_models = MODEL_SIZES if args.model == "all" else {args.model: MODEL_SIZES[args.model]}
    for model_name, imgsz in selected_models.items():
        fp16_dir = None
        if args.precision in {"fp16", "int8", "all"}:
            fp16_dir = export_fp16(model_name, imgsz)
        if args.precision in {"int8", "all"}:
            quantize_int8(model_name, imgsz, fp16_dir)
        if args.precision in {"fp32", "all"}:
            export_fp32(model_name, imgsz)

    print(f"\nAll exports are in {EXPORT_ROOT}")


if __name__ == "__main__":
    main()
