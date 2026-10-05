#!/usr/bin/env python3
"""Run SyRe on one sample from the native ``2d_test`` split.

This follows the GLaMMedv16 medical-segmentation path instead of constructing
an independent image/prompt manifest: the sample is loaded from
``image2label_2d_test.json`` by SyRe's ``MedReferSegmDataset`` and passed
through ``custom_collate_fn`` before model inference.
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
        evaluate_output,
        load_2d_test_dataset,
        load_model,
        make_2d_test_loader,
        MetricAccumulator,
        predict_batch,
        write_json,
    )
except ImportError:  # Direct invocation: python scripts/inference_syre.py
    from syre_2d_test import (
        evaluate_output,
        load_2d_test_dataset,
        load_model,
        make_2d_test_loader,
        MetricAccumulator,
        predict_batch,
        write_json,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Dataset root containing image2label_2d_test.json")
    parser.add_argument("--index", type=int, required=True, help="Zero-based sample index in the 2d_test index")
    parser.add_argument("--output-dir", required=True, help="Directory for masks, visualization, and JSON output")
    parser.add_argument("--model", default="McGregorW/SyRe", help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--vision-pretrained", default=None, help="Optional local SAM ViT-H checkpoint")
    parser.add_argument("--mode", choices=["2d_test"], default="2d_test")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=1536)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--conv-type", choices=["llava_v1", "llava_llama_2"], default="llava_v1")
    parser.add_argument("--save-visualization", action="store_true", help="Save a green-GT/red-pred overlay")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.index < 0:
        raise ValueError("--index must be non-negative")

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
    if args.index >= len(dataset):
        raise IndexError(f"--index {args.index} is outside the 2d_test dataset of length {len(dataset)}")

    # Reuse the exact collate path used by evaluation/training.  A one-item
    # subset keeps the batch dictionary identical to the GLaMMedv16 flow.
    import torch

    sample = torch.utils.data.Subset(dataset, [args.index])
    loader = make_2d_test_loader(bundle, sample, batch_size=1, workers=0)
    batch = next(iter(loader))
    model_output, moved_batch = predict_batch(bundle, batch)

    accumulator = MetricAccumulator()
    sample_result = evaluate_output(
        model_output,
        moved_batch,
        bundle,
        output_dir=output_dir,
        save_predictions=True,
        save_visualizations=args.save_visualization,
    )
    accumulator.update(sample_result)
    summary = accumulator.summary(mode=args.mode)
    summary["index"] = args.index
    write_json(output_dir / "inference.json", {"sample": sample_result, "summary": summary})
    print(json.dumps({"sample": sample_result, "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
