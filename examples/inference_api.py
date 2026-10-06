"""Python API example for one sample from SyRe's native 2D test split."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.syre_2d_test import (
    MetricAccumulator,
    evaluate_output,
    load_2d_test_dataset,
    load_model,
    make_2d_test_loader,
    predict_batch,
    write_json,
)


bundle = load_model(model_id="McGregorW/SyRe", device="cuda:0", dtype="bf16")
dataset = load_2d_test_dataset(bundle.tokenizer, "/path/to/SyReData", mode="2d_test")
sample = torch.utils.data.Subset(dataset, [0])
loader = make_2d_test_loader(bundle, sample)
batch = next(iter(loader))
model_output, moved_batch = predict_batch(bundle, batch)

result = evaluate_output(
    model_output,
    moved_batch,
    bundle,
    output_dir=Path("outputs/sample_000000"),
    save_predictions=True,
    save_visualizations=True,
)
summary = MetricAccumulator()
summary.update(result)
write_json(Path("outputs/sample_000000/summary.json"), summary.summary(mode="2d_test"))
print(result)
