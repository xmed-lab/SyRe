"""
train.py - SyRe Training on Single Dataset Type

Trains the SyRe model on one dataset type (Segmentation) at a time, iterating thoroughly through
the chosen dataset. This targeted approach is optimal for specialized training on specific downstream task.
"""
import os
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.SyRe import SyReForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU, calculateDice)

from dataset.segm_datasets.Med_Segm_ds_new import MedReferSegmDataset
import warnings
warnings.filterwarnings('ignore')




def parse_args(args):
    parser = argparse.ArgumentParser(description="SyRe Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--seq_length", default=1024, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str, help="Choose from: CocoCap, LLaVaInstruct")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--num_classes_per_sample", default=5, type=int)
    parser.add_argument('--mode', default=None, type=str)
    parser.add_argument('--mode_val', default=None, type=str)
    parser.add_argument('--text_prompts_path', default=None, type=str)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=2.0, type=float)
    parser.add_argument("--bce_loss_weight", default=0.5, type=float)
    parser.add_argument("--boundary_loss_weight", default=0.2, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--use_mm_proj", action="store_true", default=False)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--print_freq_val", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--rank", default=0, type=int, help="global rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="RefCOCOgRegVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Mid-epoch validation & checkpoint
    parser.add_argument("--mid_val_frac", default=0.1, type=float,
                        help="Do a small validation every this fraction of an epoch. e.g. 0.25 => 4 times/epoch")
    parser.add_argument("--mid_val_steps", default=500, type=int,
                        help="How many val batches to run for mid-epoch validation (small-scale val).")
    parser.add_argument("--save_mid_ckpt", action="store_true", default=True,
                        help="Save checkpoint after each mid-epoch validation.")
    parser.add_argument("--resume_from_mid", action="store_true",
                        help="If set, resume using mid-epoch info (epoch + step_in_epoch) from deepspeed client_state/tag.")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)

    return parser.parse_args(args)



def _get_torch_dtype(args):
    if args.precision == "bf16":
        return torch.bfloat16
    elif args.precision == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Unknown precision: {args.precision}")



def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.rank == 0 and args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, 
        model_max_length=args.model_max_length, padding_side="right", use_fast=False, local_files_only=True
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    # modality_tokens = ['<m>', '</m>']
    # tokenizer.add_tokens(modality_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    print(f"\033[91m {args.seg_token_idx, args.bop_token_idx, args.eop_token_idx} \033[0m")
    
    # ===== 保存全部词表 =====
    #vocab = tokenizer.get_vocab()
    #sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # 按 id 排序
    #save_path = os.path.join("./", "tokenizer_vocab.txt")
    #with open(save_path, "w", encoding="utf-8") as f:
    #    for token, idx in sorted_vocab:
    #        token_display = token.replace("\n", "\\n")  # 避免换行干扰
    #        f.write(f"{idx}\t{token_display}\n")

    #print(f"\033[93m Tokenizer vocab saved to {save_path} ({len(sorted_vocab)} tokens) \033[0m")

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the SyRe model. """
    
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight", "boundary_loss_weight",
                   "seg_token_idx", "vision_pretrained", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx", "use_mm_proj", "seq_length"]}
    model_args["num_level_reg_features"] = 4

    dtype = _get_torch_dtype(args)


    print(f"\033[95m {model_args} \033[0m")

    model = SyReForCausalLM.from_pretrained(
        args.version, 
        torch_dtype=torch.bfloat16, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    dtype = _get_torch_dtype(args)


    # Initialize vision tower
    # print(
    #     '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
    #         args.vision_tower
    #     ) + '\033[0m'
    # )
    model.get_model().initialize_vision_modules(model.get_model().config)



    # Initialize SyRe model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_syre_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    # for p in vision_tower.parameters():
    #     p.requires_grad = False


    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    model.get_model().grounding_encoder.to(dtype=dtype, device=args.local_rank)
    model.get_model().mm_projector.to(device=args.local_rank, dtype=dtype)
    model.get_model().text_hidden_fcs.to(device=args.local_rank, dtype=dtype)

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            # print(f"\033[95m {name} \033[0m")
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["mm_projector", "text_hidden_fcs", "grounding_encoder"]
                    # x not in name for x in ["mm_projector", "text_hidden_fcs", "mask_decoder", "vision_tower"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)


    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "grounding_encoder", "text_hidden_fcs",
                         "mm_projector", "t2i_projection"]
    #trainable_modules = ["lm_head", "grounding_encoder", "text_hidden_fcs",
    #                     "mm_projector", "t2i_projection"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            # print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    # world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size, "mode": args.mode,
                      "text_prompts_path": args.text_prompts_path,
                      "num_classes_per_sample": args.num_classes_per_sample}


    train_dataset = MedReferSegmDataset(**common_ds_args, random_sampling=False, refer_segm_data=args.refer_segm_data)

    # Assert that exactly one dataset type is set

    # world_size = torch.cuda.device_count()
    # print(f"\033[91m torch world_size {world_size} \033[0m")
    # Summing lengths of all datasets
    total_length = len(train_dataset)
    print(f"Training with {total_length} examples.")
    # Calculate steps per epoch
    effective_batch_size = args.batch_size * args.grad_accumulation_steps * world_size
    steps_per_epoch = total_length // effective_batch_size
    # modify steps per epoch
    args.steps_per_epoch = steps_per_epoch

    # Validation datasets
    val_dataset = None
    if not args.no_eval:
        # val_dataset_class = MedReferSegmDataset
        common_ds_args['mode'] = args.mode_val
        val_dataset = MedReferSegmDataset(**common_ds_args, validation=True, split='val')

    return train_dataset, val_dataset


def setup_data_loaders(args, train_dataset, val_dataset, tokenizer):
    sampler_args = {"shuffle": True, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=local_rank,
        inference=False, seq_length=args.seq_length
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=local_rank,
        inference=inference_mode, seq_length=args.seq_length
    )

    # Training loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, **sampler_args
            ), collate_fn=collate_fn_args_train, **train_loader_args
        )

    # Validation loader
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, **sampler_args), )

    return train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.01,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    args.start_step = 0
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model_last_epoch")
        if os.path.exists(resume):
            args.resume = resume
        else:
            print("[WARNING] Resume ckpt dir not exists!")

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
            print(ckpt_dir)
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")




def fast_forward_train_iterator(train_loader, start_step, grad_accum_steps):
    it = iter(train_loader)
    to_skip = int(start_step) * int(grad_accum_steps)
    for _ in range(to_skip):
        try:
            next(it)
        except StopIteration:
            it = iter(train_loader)
            next(it)
    return it



def main(args):

    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    #if args.rank == 0 and args.local_rank == 0:
    #   for name, param in model.named_parameters():
    #      if param.requires_grad == False:
    #          print(f"\033[94m Frozen: {name} \033[0m")
    #      else:
    #          print(f"\033[91m Trainable: {name} \033[0m")

    writer = initialize_environment(args)

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)



    train_dataset, val_datasets = initialize_datasets_and_loaders(args, tokenizer)

    # Choose resume behavior
    if args.resume_from_mid:
        if args.rank == 0 and args.local_rank == 0:  # Log the progress
            print("-----------------------------------------------------------------")
            print("Resuming training from mid epoch")
            print("-----------------------------------------------------------------")

        resume_training_from_checkpoint_mid(model_engine, args)   # epoch + step
    else:
        resume_training_from_checkpoint(model_engine, args)       # epoch only

    train_loader, val_loader = setup_data_loaders(args, train_dataset, val_datasets, tokenizer)

    # NEW: start_step resume support
    dataset_iter = iter(train_loader)




    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args)[0]
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]

    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        dataset_iter = train(train_loader, val_loader, model_engine, epoch, scheduler, writer, dataset_iter, args)

        if args.mask_validation:
            giou, ciou, dice = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            best_ciou = ciou if is_best else best_ciou
            if args.rank == 0 and args.local_rank == 0:  # Log the progress
                print("================================================================================================")
                print(f"Epoch: {epoch},  dice: {dice}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
                print("================================================================================================")
            torch.distributed.barrier()
            save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = cur_val_loss < best_val_loss
            best_val_loss = min(cur_val_loss, best_val_loss)
            if args.rank == 0 and args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")

            torch.distributed.barrier()
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)





def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)

    # 只让 rank0 做非分布式的 torch.save（可选）
    if args.rank == 0 and args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value},
                   os.path.join(save_dir, ckpt_filename))

    # deepspeed 的 save_checkpoint 必须所有rank都调用（它自己会处理并行保存）
    model_engine.save_checkpoint(save_dir)



def save_checkpoint_mid(model_engine, args, epoch, metric_name, metric_value, is_best, step_in_epoch=None, global_step=None):
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)

    # Deepspeed checkpoint tag: make it resume-able mid-epoch
    client_state = {"epoch": int(epoch)}
    if step_in_epoch is not None:
        client_state["step_in_epoch"] = int(step_in_epoch)
    if global_step is not None:
        client_state["global_step"] = int(global_step)

    tag = None
    if global_step is not None:
        tag = f"global_step{int(global_step)}"

    if args.distributed:
        torch.distributed.barrier()

    print(f"[rank{args.rank}] about to save_checkpoint tag={tag} dir={save_dir}", flush=True)

    # 所有 rank 都必须调用
    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    
    if args.distributed:
        torch.distributed.barrier()
    # 只让 rank0 写完成标记
    if args.rank == 0 and args.local_rank == 0 and tag is not None:
        done = os.path.join(save_dir, tag, "DONE")
        with open(done, "w") as f:
            f.write("ok\n")


import traceback, sys, os

def safe_print(msg):
    print(msg, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

def resume_training_from_checkpoint_mid(model_engine, args):
    args.start_step = 0

    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model_last_epoch")
        if os.path.exists(resume):
            args.resume = resume
        else:
            safe_print("[WARNING] Resume ckpt dir not exists!")

    if not args.resume:
        return

    # --- pre-check visible for every rank ---
    safe_print(f"[rank{args.rank}] resume={args.resume} exists={os.path.exists(args.resume)}")
    if os.path.exists(args.resume):
        safe_print(f"[rank{args.rank}] resume ls={os.listdir(args.resume)[:50]}")

    # Make sure everyone reaches here
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Read tag explicitly (avoid implicit 'latest' confusion)
    ckpt_tag = None
    latest_path = os.path.join(args.resume, "latest")
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            ckpt_tag = f.read().strip()
    safe_print(f"[rank{args.rank}] latest tag = {ckpt_tag}")

    try:
        safe_print(f"[rank{args.rank}] >>> calling load_checkpoint(tag={ckpt_tag})")
        load_path, client_state = model_engine.load_checkpoint(args.resume, tag=ckpt_tag)
        safe_print(f"[rank{args.rank}] <<< load_checkpoint done. load_path={load_path}")

    except Exception as e:
        safe_print(f"[rank{args.rank}] !!! load_checkpoint EXCEPTION: {repr(e)}")
        safe_print(traceback.format_exc())

        # Make the failure obvious (avoid only seeing TCPStore reset)
        try:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception:
            pass

        os._exit(1)

    # parse client_state if success
    if client_state is not None and "epoch" in client_state:
        args.start_epoch = int(client_state.get("epoch", 0))
        args.start_step  = int(client_state.get("step_in_epoch", 0))
    else:
        args.start_epoch = 0
        args.start_step = 0


def format_seconds(seconds):
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def _amp_dtype(args):
    return torch.float16 if args.precision == "fp16" else torch.bfloat16


def train(data_loader, val_loader, model, epoch, scheduler, writer, dataset_iter, args):
    """Main training loop."""
    if getattr(args, "distributed", False) and hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)


    # print(data_loader)

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress (with ETA)."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            # -------- ETA --------
            remaining_steps = args.steps_per_epoch - (global_step + 1)
            eta_seconds = remaining_steps * batch_time.avg
            eta_str = format_seconds(eta_seconds)

            if args.rank == 0 and args.local_rank == 0:
                progress.display(global_step + 1, extra_str=f"ETA {eta_str}")

                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            for tracker in trackers.values():
                tracker.reset()


    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_boundary_loss": AverageMeter("MaskBoundaryLoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    # NEW: mid-epoch resume only for the first resumed epoch
    start_step = int(getattr(args, "start_step", 0)) if epoch == int(getattr(args, "start_epoch", 0)) else 0

    # NEW: how often to do mid-val
    frac = float(getattr(args, "mid_val_frac", 0.25))
    interval = max(1, int(round(args.steps_per_epoch * frac)))

    amp_dtype = _amp_dtype(args)


    for global_step in range(start_step, args.steps_per_epoch):

        # if global_step > 20:
        #    break


        ### train CLIP

        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice

            # freeze_SAM(model)
            data_batch, new_iter = get_next_input(dataset_iter, data_loader)

            # if global_step > 39:
            #     print(f"====== global_step {global_step, data_batch['image_paths']}")
            # if args.local_rank == 0:
                # print(data_batch)
                # print(f"\033[93m----{list(data_batch.keys())}\033[0m")

            dataset_iter = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["grounding_enc_images"]:
                data_batch[key] = data_batch[key].to(dtype=amp_dtype)

            # print(f"\033[91m ====== {args.rank, model.device} \033[0m")

            output_dict = model(**data_batch, train_seg=True)
            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    # print(f"\033[91m {key} \033[0m")
                    tracker.update(output_dict[key].item(), data_batch["grounding_enc_images"].size(0))

            # print(f"\033[95m {output_dict['mask_loss']} \033[0m")

            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()



        # ----------------------------
        # Mid-epoch small validation + checkpoint
        # ----------------------------
        is_mid_point = (((global_step + 1) % interval == 0) and ((global_step + 1) != args.steps_per_epoch)) or ((global_step + 1) == args.steps_per_epoch)
        if is_mid_point and (not args.no_eval) and (val_loader is not None):
            if args.distributed:
                torch.distributed.barrier()
            if args.rank == 0 and args.local_rank == 0:
                print("---------------------------------------------------------------------------------------------------------------------------------------------")
                print(f"\n----- Mid-epoch val @ epoch={epoch}, step={global_step+1}/{args.steps_per_epoch} (max_steps={args.mid_val_steps}) -----")
                print("---------------------------------------------------------------------------------------------------------------------------------------------")

            giou, ciou, dice = validate_model_performance(
                val_loader, model, epoch, writer, args, max_steps=int(args.mid_val_steps)
            )
            metric_name = "giou-ciou"
            metric_value = f"{giou:.4f}-{ciou:.4f}"
            is_best = False  # mid-val 不更新 best（你也可以自己改成更新）

            if args.distributed:
                torch.distributed.barrier()

            # Save mid-epoch ckpt
            if getattr(args, "save_mid_ckpt", True):
                # total global step across run (for tag/logging)
                total_gs = epoch * args.steps_per_epoch + (global_step + 1)
                torch.distributed.barrier() if args.distributed else None
                save_checkpoint_mid(
                    model, args, epoch,
                    metric_name=metric_name, metric_value=metric_value,
                    is_best=is_best,
                    step_in_epoch=(global_step + 1),
                    global_step=total_gs
                )
            if args.distributed:
                torch.distributed.barrier()
                

        

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.rank == 0 and args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

        model.train()

    
    if epoch == int(getattr(args, "start_epoch", 0)):
        args.start_step = 0
    return dataset_iter



def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args, max_steps=None):
    """
    Validation without tqdm; prints like train via ProgressMeter.
    Fixes:
      - Never put numpy arrays into AverageMeter (ProgressMeter formatting requires scalars).
      - For segm: accumulate intersection/union and dice as tensors for epoch-level metrics; DDP all_reduce.
    Returns:
      - if mask_validation: (giou_fg, ciou_fg, dice_epoch)
      - else: avg_val_ce_loss
    """
    if validation_loader is None:
        return (0.0, 0.0, 0.0) if args.mask_validation else 0.0

    def is_dist():
        return bool(getattr(args, "distributed", False)) and torch.distributed.is_available() and torch.distributed.is_initialized()

    # -------------------------
    # Segmentation/GCG validation
    # -------------------------
    if args.mask_validation:
        # ✅ Only scalar meters for printing
        meters = {
            "gIoU_fg_stepavg": AverageMeter("gIoU_fg", ":.4f"),
            "dice_stepavg": AverageMeter("dice", ":.4f"),
        }
        batch_time = AverageMeter("Time", ":.4f")
        data_time = AverageMeter("Data", ":.4f")

        progress = ProgressMeter(
            len(validation_loader),
            list(meters.values()) + [batch_time, data_time],
            prefix=f"Val(Segm): [Epoch {current_epoch}]"
        )

        device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

        # ✅ Epoch-level accumulators (tensor, DDP-reducible)
        inter_sum = torch.zeros(2, device=device, dtype=torch.float64)   # [bg, fg]
        union_sum = torch.zeros(2, device=device, dtype=torch.float64)   # [bg, fg]
        dice_sum = torch.zeros((), device=device, dtype=torch.float64)
        n_total = torch.zeros((), device=device, dtype=torch.float64)

        training_model.eval()
        end = time.time()

        for step, data_batch in enumerate(validation_loader):
            if max_steps is not None and step >= max_steps:
                break
            data_time.update(time.time() - end)

            data_batch = dict_to_cuda(data_batch)
            for key in ["grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()

            torch.cuda.empty_cache()
            with torch.no_grad():
                results = training_model(**data_batch)            
            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            pred_masks = (predictions[0] > 0).int()

            # batch-local accumulators
            giou_fg_sum = 0.0
            dice_batch_sum = 0.0
            n = int(gt_masks.shape[0])

            for target, pred in zip(gt_masks, pred_masks):
                intersect, union, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                # intersect/union are tensors shape [2] on GPU
                inter_sum += intersect.to(dtype=torch.float64)
                union_sum += union.to(dtype=torch.float64)

                # foreground IoU (idx=1), handle no-object (union==0) -> iou=1
                union_fg = float(union[1].item())
                if union_fg == 0.0:
                    iou_fg = 1.0
                else:
                    iou_fg = float((intersect[1].double() / (union[1].double() + 1e-5)).item())

                giou_fg_sum += iou_fg

                d = float(calculateDice(pred.contiguous().clone(), target.contiguous()))
                # print(d)
                dice_batch_sum += d

            giou_fg_avg = giou_fg_sum / max(n, 1)
            dice_avg = dice_batch_sum / max(n, 1)

            # ✅ meters only get scalars (NOT arrays)
            meters["gIoU_fg_stepavg"].update(float(giou_fg_avg), n=n)
            meters["dice_stepavg"].update(float(dice_avg), n=n)

            # ✅ epoch dice accumulation
            dice_sum += torch.tensor(dice_batch_sum, device=device, dtype=torch.float64)
            n_total += torch.tensor(n, device=device, dtype=torch.float64)

            batch_time.update(time.time() - end)
            end = time.time()

            # print like train
            if step % args.print_freq_val == 0:
                if is_dist():
                    for m in meters.values():
                        m.all_reduce()
                    batch_time.all_reduce()
                    data_time.all_reduce()

                # -------- ETA --------
                remaining = len(validation_loader) - (step + 1)
                eta_seconds = remaining * batch_time.avg
                eta_str = format_seconds(eta_seconds)

                if args.rank == 0 and args.local_rank == 0:
                    progress.display(step + 1, extra_str=f"ETA {eta_str}")

                for m in meters.values():
                    m.reset()
                batch_time.reset()
                data_time.reset()


        # ✅ DDP reduce epoch accumulators
        if is_dist():
            torch.distributed.all_reduce(inter_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(union_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(dice_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(n_total, op=torch.distributed.ReduceOp.SUM)

        # epoch-level metrics
        iou_per_class = inter_sum / (union_sum + 1e-10)
        ciou_fg = float(iou_per_class[1].item())                          # foreground class IoU
        giou_fg = float((inter_sum[1] / (union_sum[1] + 1e-10)).item())   # foreground global IoU
        dice_epoch = float((dice_sum / (n_total + 1e-10)).item())

        if args.rank == 0 and args.local_rank == 0:
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("val/giou_fg_epoch", giou_fg, current_epoch)
                tensorboard_writer.add_scalar("val/ciou_fg_epoch", ciou_fg, current_epoch)
                tensorboard_writer.add_scalar("val/dice_epoch", dice_epoch, current_epoch)
            print("giou_fg_epoch: {:.4f}, ciou_fg_epoch: {:.4f}, dice_epoch: {:.4f}".format(
                giou_fg, ciou_fg, dice_epoch
            ))

        return giou_fg, ciou_fg, dice_epoch





if __name__ == "__main__":
    # torchrun sets these env vars automatically:
    #   RANK, LOCAL_RANK, WORLD_SIZE (and usually MASTER_ADDR/MASTER_PORT)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(world_size, rank, local_rank)

    args = parse_args(sys.argv[1:])

    # keep your original logic: distributed iff world_size > 1
    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank

    main(args)
