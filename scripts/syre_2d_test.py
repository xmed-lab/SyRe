#!/usr/bin/env python3
"""Shared native 2D-test utilities for SyRe inference and evaluation.

The data path intentionally mirrors the GLaMMedv16 medical-segmentation
evaluation code: ``image2label_2d_test.json`` is read by
``MedReferSegmDataset`` and batches are built with ``custom_collate_fn``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    torch: object
    device: object
    dtype: object
    seq_length: int
    conv_type: str


def resolve_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def resolve_dtype(torch, requested: str, device) -> object:
    if requested == "auto":
        requested = "bf16" if device.type == "cuda" else "fp32"
    if requested == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested, but this CUDA device does not support it")
        return torch.bfloat16
    if requested == "fp16":
        if device.type == "cpu":
            raise ValueError("fp16 inference on CPU is unsupported; use --dtype fp32")
        return torch.float16
    if requested == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {requested}")


def load_model(
    model_id: str = "McGregorW/SyRe",
    vision_pretrained: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    model_max_length: int = 1536,
    seq_length: int = 1024,
    conv_type: str = "llava_v1",
) -> ModelBundle:
    """Load the public or local SyRe checkpoint for native 2D-test batches."""

    import torch
    import transformers
    from model.SyRe import SyReForCausalLM
    from model.llava import conversation as conversation_lib

    torch_device = resolve_device(torch, device)
    load_dtype = resolve_dtype(torch, dtype, torch_device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    # The released checkpoint was exported with use_mm_proj=true.  The local
    # SyRe constructor must see use_mm_proj=false first so it creates the
    # projector before the checkpoint state dict is loaded.
    config = transformers.AutoConfig.from_pretrained(model_id)
    config.use_mm_proj = False
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    model = SyReForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=load_dtype,
        train_mask_decoder=True,
        out_dim=int(getattr(config, "out_dim", 256)),
        ce_loss_weight=1.0,
        dice_loss_weight=2.0,
        bce_loss_weight=0.5,
        boundary_loss_weight=0.2,
        seg_token_idx=seg_token_idx,
        bbox_token_idx=bbox_token_idx,
        vision_pretrained=vision_pretrained,
        use_mm_start_end=True,
        mm_use_im_start_end=True,
        mm_vision_select_layer=int(getattr(config, "mm_vision_select_layer", -2)),
        pretrain_mm_mlp_adapter="",
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        with_region=True,
        use_mm_proj=False,
        seq_length=seq_length,
        num_level_reg_features=4,
    )
    model.get_model().initialize_vision_modules(model.config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device=torch_device, dtype=load_dtype)
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type]
    return ModelBundle(model, tokenizer, torch, torch_device, load_dtype, seq_length, conv_type)


def load_2d_test_dataset(
    tokenizer,
    dataset_dir: str,
    image_size: int = 1024,
    precision: str = "bf16",
    mode: str = "2d_test",
):
    """Build the same held-out dataset used by GLaMMedv16's 2D evaluator."""

    if mode != "2d_test":
        raise ValueError("SyRe's native evaluator only supports --mode 2d_test")
    manifest = Path(dataset_dir).expanduser().resolve() / "image2label_2d_test.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Expected 2D test index at {manifest}")

    from dataset.segm_datasets.Med_Segm_ds_new import MedReferSegmDataset

    return MedReferSegmDataset(
        dataset_dir=str(Path(dataset_dir).expanduser().resolve()),
        tokenizer=tokenizer,
        precision=precision,
        image_size=image_size,
        mode=mode,
        validation=True,
        inference=True,
        split="val",
        random_sampling=False,
    )


def make_2d_test_loader(bundle: ModelBundle, dataset, batch_size: int = 1, workers: int = 0):
    """Create a deterministic loader using SyRe's training collate function."""

    if batch_size != 1:
        raise ValueError("The native 2d_test path currently requires --batch-size 1")
    from dataset.dataset import custom_collate_fn
    import torch

    collate_fn = partial(
        custom_collate_fn,
        tokenizer=bundle.tokenizer,
        use_mm_start_end=True,
        inference=True,
        local_rank=-1,
        seq_length=bundle.seq_length,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=bundle.device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )


def move_batch(bundle: ModelBundle, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Move tensors in a collated GLaMM/SyRe batch to the selected device."""

    torch = bundle.torch

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(bundle.device, non_blocking=True)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    moved = move(batch)
    moved["grounding_enc_images"] = moved["grounding_enc_images"].to(dtype=bundle.dtype)
    return moved


def predict_batch(bundle: ModelBundle, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one native collated 2D-test batch and return outputs plus the batch."""

    moved = move_batch(bundle, batch)
    with bundle.torch.inference_mode():
        # ``custom_collate_fn`` already supplies ``inference=True`` in the
        # batch dictionary; only the segmentation branch is added here.
        output = bundle.model(**moved, train_seg=True)
    return output, moved


def _as_bool_tensor(value, torch):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value > 0


def mask_metrics(prediction, target, torch) -> Dict[str, float]:
    """Match the foreground metric conventions used in GLaMMedv16."""

    pred = _as_bool_tensor(prediction, torch)
    gt = _as_bool_tensor(target, torch)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    intersection = torch.logical_and(pred, gt).sum().double()
    union = torch.logical_or(pred, gt).sum().double()
    pred_sum = pred.sum().double()
    gt_sum = gt.sum().double()
    iou = 1.0 if union.item() == 0 else float((intersection / union).item())
    dice = float(((2.0 * intersection + 1e-7) / (pred_sum + gt_sum + 1e-7)).item())
    return {"iou": iou, "dice": dice, "intersection": float(intersection.item()), "union": float(union.item())}


def modality_from_path(image_path: str) -> str:
    name = Path(image_path).name
    return name.split("--", 1)[0].split("_", 1)[0]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "item"


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_output(
    output: Dict[str, Any],
    batch: Dict[str, Any],
    bundle: ModelBundle,
    output_dir: Optional[Path] = None,
    save_predictions: bool = False,
    save_visualizations: bool = False,
) -> Dict[str, Any]:
    """Collect per-mask, per-class, and per-modality metrics for one image."""

    torch = bundle.torch
    predictions = output["pred_masks"]
    targets = batch["masks_list"]
    image_paths = batch["image_paths"]
    classes_batch = batch["sampled_classes_list"]
    if len(image_paths) != 1 or len(predictions) != 1:
        raise ValueError("The native 2d_test evaluator expects one image per batch")
    image_path = str(image_paths[0])
    prediction_list = predictions[0]
    target_list = targets[0]
    class_names = list(classes_batch[0])
    if len(prediction_list) != len(target_list) or len(prediction_list) != len(class_names):
        raise ValueError(
            "The model returned a different number of masks than the dataset "
            f"(pred={len(prediction_list)}, target={len(target_list)}, classes={len(class_names)})"
        )

    modality = modality_from_path(image_path)
    records = []
    for class_name, prediction, target in zip(class_names, prediction_list, target_list):
        pred_binary = prediction > 0
        metrics = mask_metrics(pred_binary, target, torch)
        record = {
            "image": image_path,
            "modality": modality,
            "class": str(class_name),
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "intersection": metrics["intersection"],
            "union": metrics["union"],
        }
        records.append(record)
        if output_dir is not None and (save_predictions or save_visualizations):
            save_prediction_artifacts(
                image_path=image_path,
                class_name=str(class_name),
                prediction=pred_binary.detach().cpu().numpy(),
                target=_as_bool_tensor(target, torch).detach().cpu().numpy(),
                metrics=metrics,
                output_dir=output_dir,
                save_prediction=save_predictions,
                save_visualization=save_visualizations,
            )

    return {"image": image_path, "modality": modality, "records": records}


def save_prediction_artifacts(
    image_path: str,
    class_name: str,
    prediction,
    target,
    metrics: Dict[str, float],
    output_dir: Path,
    save_prediction: bool,
    save_visualization: bool,
) -> None:
    """Save masks and the green-GT/red-pred overlay used by the reference code."""

    import cv2
    import numpy as np

    stem = safe_name(Path(image_path).stem)
    class_dir = output_dir / "pred_masks" / safe_name(class_name)
    if save_prediction:
        class_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(class_dir / f"{stem}.png"), (prediction.astype(np.uint8) * 255))

    if not save_visualization:
        return
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return
    height, width = image.shape[:2]
    target = cv2.resize(target.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    prediction = cv2.resize(prediction.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    overlay[target > 0] = (0, 220, 0)
    overlay[prediction > 0] = (0, 0, 220)
    both = (target > 0) & (prediction > 0)
    overlay[both] = (0, 180, 220)
    result = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    text = f"{class_name} | Dice {metrics['dice']:.4f} | IoU {metrics['iou']:.4f}"
    cv2.putText(result, text[:180], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    vis_dir = output_dir / "visualizations" / safe_name(class_name)
    vis_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_dir / f"{stem}.jpg"), result)


class MetricAccumulator:
    """Accumulate GLaMM-style aggregate and grouped 2D metrics."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.intersection = 0.0
        self.union = 0.0
        self.per_mask_iou: List[float] = []
        self.per_mask_dice: List[float] = []
        self.class_dice = defaultdict(list)
        self.modality_dice = defaultdict(list)
        self.modality_class_dice = defaultdict(lambda: defaultdict(list))

    def update(self, sample: Dict[str, Any]) -> None:
        self.records.append(sample)
        for record in sample["records"]:
            self.intersection += record["intersection"]
            self.union += record["union"]
            self.per_mask_iou.append(record["iou"])
            self.per_mask_dice.append(record["dice"])
            self.class_dice[record["class"]].append(record["dice"])
            self.modality_dice[record["modality"]].append(record["dice"])
            self.modality_class_dice[record["modality"]][record["class"]].append(record["dice"])

    def summary(self, mode: str = "2d_test") -> Dict[str, Any]:
        class_dice = {key: _mean(value) for key, value in sorted(self.class_dice.items())}
        modality_dice = {key: _mean(value) for key, value in sorted(self.modality_dice.items())}
        modality_class_dice = {
            modality: {key: _mean(value) for key, value in sorted(classes.items())}
            for modality, classes in sorted(self.modality_class_dice.items())
        }
        return {
            "mode": mode,
            "num_images": len(self.records),
            "num_masks": len(self.per_mask_dice),
            # global_iou is the mean per-mask foreground IoU used by GLaMM.
            "global_iou": _mean(self.per_mask_iou),
            # class_iou is the foreground IoU after global intersection/union accumulation.
            "class_iou": float(self.intersection / self.union) if self.union else 0.0,
            "dice": _mean(self.per_mask_dice),
            "class_dice": class_dice,
            "modality_dice": modality_dice,
            "modality_class_dice": modality_class_dice,
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
