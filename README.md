# SyRe: Synergistic Vision-Language Reinforcement Enables Scalable On-Demand Analysis across Diverse Clinical Tasks

SyRe is a multimodal model for medical image segmentation, built on a vision-language architecture (LLaVA + SAM). It supports text-guided segmentation with LoRA fine-tuning.

## Architecture

- **Language backbone**: LLaMA-based LLaVA model
- **Vision encoder**: SAM ViT-H (frozen, with trainable mask decoder)
- **Training strategy**: LoRA fine-tuning on medical segmentation datasets

## Project Structure

```
SyRe/
├── train.py                    # Main training script
├── model/
│   ├── SyRe.py                 # Model definition (SyReForCausalLM)
│   ├── layers.py               # Custom layers
│   ├── llava/                  # Language model components
│   └── SAM/                    # Vision backbone (SAM ViT-H)
├── dataset/
│   ├── dataset.py              # Collate function
│   └── segm_datasets/          # Medical segmentation dataset
├── utils/
│   └── utils.py                # Utilities and constants
└── scripts/
    ├── finetune_2d_syre.sh     # Multi-node training launch script
    └── merge_lora_weights.py   # Merge LoRA weights after training
```

## Training

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- DeepSpeed
- SAM ViT-H checkpoint (`sam_vit_h_4b8939.pth`)
- Base model checkpoint (LLaVA-based pretrained weights)

### Launch Training

```bash
bash scripts/finetune_2d_syre.sh
```

Key training arguments (see `python train.py --help` for all options):

| Argument | Description | Default |
|----------|-------------|---------|
| `--version` | Path to base model | `MBZUAI/GLaMM-GranD-Pretrained` |
| `--dataset_dir` | Dataset root directory | - |
| `--vision_pretrained` | SAM checkpoint path | - |
| `--lora_r` | LoRA rank | 8 |
| `--lr` | Learning rate | 3e-4 |
| `--epochs` | Training epochs | 10 |
| `--batch_size` | Per-GPU batch size | 2 |

### Merge LoRA Weights

After training, merge LoRA adapters into the base model:

```bash
python scripts/merge_lora_weights.py \
  --version /path/to/base_model \
  --weight /path/to/trained_checkpoint.bin \
  --save_path /path/to/merged_output
```

## Installation

```bash
pip install -r requirements.txt
```

Note: PyTorch and transformers should be installed separately following the official guides for your CUDA version.
