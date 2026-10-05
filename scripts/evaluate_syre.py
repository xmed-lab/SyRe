#!/usr/bin/env python3
"""Evaluate SyRe on the native ``2d_test`` split.

The loader and metric path mirror ``GLaMMedv16/eval/med_seg`` while using the
SyRe model implementation in this repository.  Only ``2d_test`` is exposed on
purpose; 3D and other GLaMM evaluation modes are outside this release script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .syre_2d_test import (
        MetricAccumulator,
        evaluate_output,
        load_2d_test_dataset,
        load_model,
        make_2d_test_loader,
        predict_batch,
        write_json,
    )
except ImportError:  # Direct invocation: python scripts/evaluate_syre.py
    from syre_2d_test import (
        MetricAccumulator,
        evaluate_output,
        load_2d_test_dataset,
        load_model,
        make_2d_test_loader,
        predict_batch,
        write_json,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Dataset root containing image2label_2d_test.json")
    parser.add_argument("--output-dir", required=True, help="Directory for masks, metrics, and optional visualizations")
    parser.add_argument("--model", default="McGregorW/SyRe", help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--vision-pretrained", default=None, help="Optional local SAM ViT-H checkpoint")
    parser.add_argument("--mode", choices=["2d_test"], default="2d_test")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=1536)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, choices=[1], default=1, help="Native 2d_test evaluation uses batch size 1")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples from the test index")
    parser.add_argument("--conv-type", choices=["llava_v1", "llava_llama_2"], default="llava_v1")
    parser.add_argument("--save-predictions", action="store_true", help="Save one binary PNG per class")
    parser.add_argument("--save-visualizations", action="store_true", help="Save green-GT/red-pred overlays")
    return parser


def _write_compatibility_outputs(output_dir: Path, summary: dict) -> None:
    """Write filenames used by the GLaMMedv16 2D evaluation workflow."""

    write_json(output_dir / "class_iou_avg--2d_test.json", summary["class_dice"])
    write_json(output_dir / "mod_iou_avg--2d_test.json", summary["modality_class_dice"])
    write_json(output_dir / "summary--2d_test.json", summary)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when provided")

    output_dir = Path(args.output_dir).expanduser().resolve()
    bundle = load_model(
        model_id=args.model,
        vision_pretrained=args.vision_pretrained,
        device=args.device,
        dtype=args.dtype,
        model_max_length=args.model_max_length,
        seq_length=args.seq_length,
        conv_type=args.conv_type,
    )
    dataset = load_2d_test_dataset(
        bundle.tokenizer,
        args.dataset_dir,
        image_size=args.image_size,
        precision=args.dtype if args.dtype != "auto" else "bf16",
        mode=args.mode,
    )
    end_index = len(dataset) if args.limit is None else min(len(dataset), args.start_index + args.limit)
    if args.start_index >= len(dataset):
        raise IndexError(f"--start-index {args.start_index} is outside the 2d_test dataset of length {len(dataset)}")

    import torch

    subset = torch.utils.data.Subset(dataset, range(args.start_index, end_index))
    loader = make_2d_test_loader(bundle, subset, batch_size=args.batch_size, workers=args.workers)
    accumulator = MetricAccumulator()
    metrics_path = output_dir / "metrics--2d_test.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(loader, start=args.start_index):
            model_output, moved_batch = predict_batch(bundle, batch)
            sample_result = evaluate_output(
                model_output,
                moved_batch,
                bundle,
                output_dir=output_dir,
                save_predictions=args.save_predictions,
                save_visualizations=args.save_visualizations,
            )
            accumulator.update(sample_result)
            handle.write(json.dumps(sample_result, ensure_ascii=False) + "\n")
            print(f"[{batch_index + 1}/{end_index}] {sample_result['image']}")

    summary = accumulator.summary(mode=args.mode)
    summary["start_index"] = args.start_index
    summary["limit"] = args.limit
    _write_compatibility_outputs(output_dir, summary)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
