import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from utils.utils import IMAGE_TOKEN_INDEX



def boundary_map(x):
    # x: [N,H,W], in [0,1]
    x = x.unsqueeze(1)
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    return (dil - ero).clamp(0, 1).squeeze(1)

def compute_boundary_loss(pred_logits, gt_mask, mask_count):
    pred = pred_logits.sigmoid()
    pred_b = boundary_map(pred)
    gt_b = boundary_map(gt_mask.float())
    loss = F.binary_cross_entropy(pred_b, gt_b, reduction="none")
    loss = loss.flatten(1).mean(1).sum() / (mask_count + 1e-8)
    return loss

def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class SyReBaseModel:
    def __init__(self, config, **kwargs):
        super(SyReBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_syre_model(self.config)

    def initialize_syre_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)
        self.config.mm_hidden_size = self.grounding_encoder.image_encoder.embed_dim

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.1), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()

        t2i_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
                                  nn.Dropout(0.1), ]
        self.t2i_projection = nn.ModuleList([nn.Sequential(*t2i_projection_layers)])
        self.t2i_projection.train()

        # self.modality_projection = nn.Linear(in_dim, self.config.mm_hidden_size)
        # self.modality_projection.train()


class SyReModel(SyReBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(SyReModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _initialize_new_class_dict(self):
        self.new_class_dict = {}

    def _initialize_new_class_dict_cls(self, cls_name):
        self.new_class_dict[cls_name] = {}
        self.new_class_dict[cls_name]["feature"] = []


    def _save_class_dict(self, save_dir):
        torch.save(self.new_class_dict, save_dir)

    def _load_class_dict(self, save_dir):
        self.new_class_dict = torch.load(save_dir)


    def register_masks(self, mask):
        self.mask = mask


    def register_roi_features(self, cls_name, image_embeddings):
        for i in range(self.mask.shape[0]):
            mask_i = self.mask[i:i+1].unsqueeze(0).float()  # (1, 1, H, W)
            mask_resize = F.interpolate(mask_i, (64, 64), mode="bilinear",
                                        align_corners=False)  # (1, 1, 64, 64)
            if mask_resize.sum() < 1.0:
                continue
            # Masked average pooling → class prototype vector (256-dim)
            masked_feats = image_embeddings * mask_resize.cuda()  # (1, 256, 64, 64)
            prototype = masked_feats.sum(dim=[2, 3]) / (mask_resize.sum() + 1e-6)  # (1, 256)
            self.new_class_dict[cls_name]["feature"].append(prototype)

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class SyReForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = SyReModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.seq_length = kwargs.pop("seq_length")
        # Adaptation hyperparameters (can be overridden at inference time)
        self.adapt_alpha = kwargs.pop("adapt_alpha", 0.3)
        self.location_scale = kwargs.pop("location_scale", 0.5)

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.boundary_loss_weight = kwargs.pop("boundary_loss_weight", None)


    def adapt_features(self, image_embeddings, feature_info):
        """Prototype-based similarity modulation.
        feature_info: (N, 256) — N prototype vectors from support images
        image_embeddings: (1, 256, 64, 64)
        Returns: sim_map (1, 1, 64, 64) in [0, 1]
        """
        # 平均所有 support prototype → class prototype
        prototype = feature_info.mean(dim=0, keepdim=True)  # (1, 256)
        prototype = F.normalize(prototype.float(), dim=1)   # L2 归一化

        B, C, H, W = image_embeddings.shape
        feats_flat = image_embeddings.view(B, C, -1)                    # (1, 256, 4096)
        feats_norm = F.normalize(feats_flat.float(), dim=1)             # (1, 256, 4096)

        # Cosine similarity: prototype vs each spatial position
        sim_map = torch.bmm(prototype.unsqueeze(0), feats_norm)         # (1, 1, 4096)
        sim_map = sim_map.view(B, 1, H, W)                              # (1, 1, 64, 64)

        # 标准化后 sigmoid → [0, 1]，高相似度区域接近 1
        sim_map = (sim_map - sim_map.mean()) / (sim_map.std() + 1e-6)
        sim_map = torch.sigmoid(sim_map)
        return sim_map




    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward(self,grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, train_seg: bool = True, new_cls_adapt: bool = False, cls_name=None, **kwargs):
        # Extract grounding encoder image embeddings
        # mod_embeds = self._create_mod_token_mask(input_ids)
        # indicate where is <SEG>
        # print(f"\033[92m mod_token_mask {mod_embeds.shape} \033[0m")

        image_embeddings, image_embeddings_last = self.model.grounding_encoder.image_encoder(grounding_enc_images, return_feat=True)
        assert image_embeddings.shape[0] == len(offset) - 1

        if new_cls_adapt:

            print(cls_name)

            location_info = self.model.new_class_dict[cls_name]['location'].unsqueeze(1).cuda()
            location_expanded = location_info.expand(1, 1280, 64, 64).bfloat16()
            image_embeddings_last = image_embeddings_last * (1.0 + self.location_scale * location_expanded)

            feature_info = torch.cat(self.model.new_class_dict[cls_name]['feature'], dim=0)  # (N, 256)

            # Prototype-based similarity map: 哪些区域像目标类
            sim_map = self.adapt_features(image_embeddings, feature_info)  # (1, 1, 64, 64)

            # 乘性调制：增强与 prototype 相似的区域
            semantic_info = sim_map.bfloat16()
            image_embeddings = image_embeddings * (1.0 + self.adapt_alpha * semantic_info)


        elif cls_name is not None:
            self.model.register_roi_features(cls_name, image_embeddings)
        else:
            pass
    


        seg_token_mask = self._create_seg_token_mask(input_ids, self.seq_length) # indicate where is <SEG>
        #print(f"\033[93m input_ids {len(input_ids)} {input_ids} \033[0m")
        #print("self.seg_token_idx", self.seg_token_idx)
        #print(f"\033[93m seg_token_mask {seg_token_mask} \033[0m")

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, image_embeddings_last, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                image_embeddings_last, bboxes, input_ids, labels, attention_masks, offset
            )

        if train_seg:
            # Process hidden states
            _, pred_embeddings, LLM_image_features = self._process_hidden_states(output_hidden_states, seg_token_mask, offset, self.seq_length, input_ids=input_ids)

            B, N, C = LLM_image_features.shape
            LLM_image_features = LLM_image_features.view(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2).contiguous()
            LLM_image_features = LLM_image_features.to(torch.float32)
            LLM_image_features = F.interpolate(LLM_image_features, scale_factor=2, mode='bilinear')
            LLM_image_features = LLM_image_features.to(torch.bfloat16)

            if new_cls_adapt:
                pred_masks, dec_features, upscaled_embeddings, dense_embeddings_list = self._generate_and_postprocess_masks_location_info(
                    pred_embeddings, image_embeddings + LLM_image_features, resize_list, label_list,
                    location_info=location_info,
                    semantic_info=semantic_info, new_cls_adapt=new_cls_adapt,
                    dense_embeddings_new=None
                )
            else:
                pred_masks  = self._generate_and_postprocess_masks(
                    pred_embeddings, image_embeddings + LLM_image_features, resize_list, label_list,
                )
                dec_features = None
                upscaled_embeddings = None
            
        else:
            pred_masks = None
            dec_features = None
            upscaled_embeddings = None
            dense_embeddings_list = None


        if inference:
            if new_cls_adapt:
                return {"pred_masks": pred_masks, "gt_masks": masks_list, "image_emb": image_embeddings_last,
                        "semantic_info": semantic_info}
            return {"pred_masks": pred_masks, "gt_masks": masks_list, }

        # Calculate losses
        return self._calculate_losses(pred_masks, masks_list, output, train_seg)

    def _create_seg_token_mask(self, input_ids, seq_length):
        mask = (input_ids[:, 1:] == self.seg_token_idx)
        # print(f"\033[91m mask {mask.shape, mask} \033[0m")
        return torch.cat(
            [torch.zeros((mask.shape[0], seq_length-1)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        )


    def _inference_path(self, input_ids, image_embeddings_last, attention_masks):
        length = input_ids.shape[0]
        image_embeddings_last_extended = image_embeddings_last.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images_feats=image_embeddings_last_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, images_feats, bboxes, input_ids, labels, attention_masks, offset):
        # print(f"\033[96m before {global_enc_images.shape} \033[0m")

        bboxes_list = bboxes

        # print(f"\033[96m after {global_enc_images.shape} \033[0m")
        # print(f"\033[96m {input_ids.shape} \033[0m")

        output = super().forward(
            images_feats=images_feats, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, bboxes=bboxes_list, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states



    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, seq_length, infer=False, input_ids=None):
        # print(f"\033[91m output_hidden_states: {output_hidden_states[-1].shape} \033[0m")
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        # print(f"\033[91m hidden_states: {len(hidden_states), hidden_states[0].shape} \033[0m")
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # print(f"\033[91m last_hidden_state: {last_hidden_state.shape} \033[0m")

        t2i_hidden_states = [self.model.t2i_projection[0](output_hidden_states[-1])]
        # print(f"\033[91m hidden_states: {len(hidden_states), hidden_states[0].shape} \033[0m")
        t2i_embeddings = torch.stack(t2i_hidden_states, dim=-1).sum(dim=-1)

        LLM_image_features = []

        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            image_token_start = image_token_indices[0]

            LLM_image_features.append(t2i_embeddings[batch_idx, image_token_start:image_token_start+seq_length])

        LLM_image_features = torch.stack(LLM_image_features)
        # print(f"\033[93m {LLM_image_features.shape} \033[0m")

        pred_embeddings = last_hidden_state[seg_token_mask]
        # print(f"\033[91m pred_embeddings: {pred_embeddings.shape} \033[0m")
        seg_token_counts = seg_token_mask.int().sum(-1)
        # print("seg_token_counts", seg_token_counts)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            # print(f"\033[96m {start_i, end_i} \033[0m")
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list, LLM_image_features

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _generate_and_postprocess_masks_location_info(self, pred_embeddings, image_embeddings, resize_list,
                                                      label_list, location_info, semantic_info, new_cls_adapt, dense_embeddings_new, infer=False):
        pred_masks = []
        dec_features = []
        upscaled_embeddings = []
        dense_embeddings_list = []
        for i, pred_embedding in enumerate(pred_embeddings):
            # print(f"\033[91m ------ pred_embedding {pred_embedding.shape} \033[0m")

            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )

            # print(f"\033[92m ------ dense_embeddings_new {dense_embeddings_new.shape} \033[0m")
            # print(f"\033[91m ------ dense_embeddings {dense_embeddings.shape} \033[0m")
            # print(f"\033[92m ------ sparse_embeddings {sparse_embeddings.shape} \033[0m")

            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _, feature_dict = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, location_info=location_info, semantic_info=semantic_info, new_cls_adapt=new_cls_adapt)

            # print("low_res_masks", low_res_masks.shape)
            # print("image_embeddings[i].unsqueeze(0)", image_embeddings[i].unsqueeze(0).shape)

            dec_feature = feature_dict['dec_features']
            upscaled_embedding = feature_dict['upscaled_embedding'].unsqueeze(1)

            # print(dec_feature.shape)

            dec_feature = dec_feature.mean(-1).view(1, 1, 64, 64).float()

            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size)

            dec_feature = self.model.grounding_encoder.postprocess_masks(
                dec_feature, input_size=resize_list[i], original_size=orig_size)

            upscaled_embedding = self.model.grounding_encoder.postprocess_masks(
                upscaled_embedding, input_size=resize_list[i], original_size=orig_size)

            # print(f"\033[92m +++ {dec_feature.shape} \033[0m")
            pred_masks.append(pred_mask[:, 0])
            dec_features.append(dec_feature[:, 0])
            upscaled_embeddings.append(upscaled_embedding[:, 0])
            dense_embeddings_list.append(dense_embeddings[0])
        return pred_masks, dec_features, upscaled_embeddings, dense_embeddings_list

    def _calculate_losses(self, pred_masks, masks_list, output, train_seg):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output, train_seg)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output, train_seg):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_boundary_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0
    
        if train_seg:
            # ===== 新增：统计是否整个 batch 都为空 =====
            all_empty = True
    
            for batch_idx, pred_mask in enumerate(pred_masks):
    
                if pred_mask.numel() > 0:
                    all_empty = False  # 只要有一个非空，就不是全空
    
                    gt_mask = masks_list[batch_idx]
    
                    # Resize gt_mask to match pred_mask if needed
                    if gt_mask.shape[0] != pred_mask.shape[0]:
                        gt_mask = gt_mask[:pred_mask.shape[0]]
    
                    assert gt_mask.shape[0] == pred_mask.shape[0], \
                        f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"
    
                    # BCE
                    mask_bce_loss += (
                        compute_sigmoid_cross_entropy(
                            pred_mask, gt_mask, mask_count=gt_mask.shape[0]
                        ) * gt_mask.shape[0]
                    )
    
                    # Dice
                    mask_dice_loss += (
                        calculate_dice_loss(
                            pred_mask, gt_mask, mask_count=gt_mask.shape[0]
                        ) * gt_mask.shape[0]
                    )

                    mask_boundary_loss += compute_boundary_loss(pred_mask, gt_mask, gt_mask.shape[0]) * gt_mask.shape[0]

    
                    num_masks += gt_mask.shape[0]
    
            # ===== 新增：如果整个 batch 都没有预测 mask =====
            if all_empty:
                warnings.warn(
                    "⚠️  All pred_mask in this batch are empty. "
                    "No segmentation loss will be computed.",
                    RuntimeWarning
                )
    
            # Normalize
            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_boundary_loss = self.boundary_loss_weight * mask_boundary_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss
            total_loss = ce_loss + mask_bce_loss + mask_dice_loss + mask_boundary_loss
    
        else:
            mask_loss = mask_bce_loss + mask_dice_loss
            total_loss = ce_loss
    
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_boundary_loss": mask_boundary_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(self, grounding_enc_images, input_ids, max_tokens_new=512):

        image_embeddings, image_embeddings_last = self.model.grounding_encoder.image_encoder(grounding_enc_images, return_feat=True)

        with torch.no_grad():
            generation_outputs = self.generate(
                images_feats=image_embeddings_last, input_ids=input_ids, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True)

            generated_output_ids = generation_outputs.sequences

            # seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
            # seg_token_mask = torch.cat(
            #     [torch.zeros((seg_token_mask.shape[0], 575), dtype=torch.bool).cuda(), seg_token_mask], dim=1, )
            # Process hidden states
            # hidden_states, predicted_embeddings = self._process_hidden_states(
            #     output_hidden_states, seg_token_mask, None, infer=True, input_ids=input_ids
            # )
            # image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            # pred_masks = self._generate_and_postprocess_masks(
            #     predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            # )
        return generated_output_ids
