import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import torch.nn.functional as F


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # print(f"\033[91m {config} \033[0m")

        if not config.use_mm_proj:
            modules = [nn.Linear(1280, self.config.hidden_size),
                        nn.GELU(),
                        nn.Linear(self.config.hidden_size, self.config.hidden_size)]
            self.mm_projector = nn.Sequential(*modules)



    def initialize_vision_modules(self, model_args, fsdp=None):
        # vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_hidden_size = model_args.mm_hidden_size

        # self.config.mm_vision_tower = vision_tower

        # vision_tower = build_vision_tower(model_args)

        # if fsdp is not None and len(fsdp) > 0:
        #     self.vision_tower = [vision_tower]
        # else:
        #     self.vision_tower = vision_tower


        self.config.use_mm_proj = True
        self.config.mm_hidden_size = mm_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            modules = [nn.Linear(self.config.mm_hidden_size, self.config.hidden_size),
                       nn.GELU(),
                       nn.Linear(self.config.hidden_size, self.config.hidden_size)]
            self.mm_projector = nn.Sequential(*modules)









class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass


    def v_l_projection(self, image_features):

        # print(f"\033[94m image_features before {image_features.shape} \033[0m")
        # image_features = image_features.to(torch.float32)
        # print(f"\033[91m image_features {image_features.shape} \033[0m")
        # image_feats_down = F.interpolate(image_features, scale_factor=3/8, mode='bilinear')
        if len(image_features.shape) == 5:
            orig_dtype = image_features.dtype
            image_feats_down = F.interpolate(image_features.float(), scale_factor=(1, 1/2, 1/2), mode='trilinear', align_corners=False).to(orig_dtype)
            # print(f"\033[92m image_features after {image_feats_down.shape} \033[0m")
            B, C, D, H, W = image_feats_down.shape
            image_feats_down = image_feats_down.reshape(B, C, D * H * W).permute(0, 2, 1)

        else:
            orig_dtype = image_features.dtype
            image_feats_down = F.interpolate(image_features.float(), scale_factor=1/2, mode='bilinear', align_corners=False).to(orig_dtype)
            B, C, H, W = image_feats_down.shape
            image_feats_down = image_feats_down.reshape(B, C, H * W).permute(0, 2, 1)
        # image_feats_down = image_feats_down.to(torch.bfloat16)



        image_features = self.get_model().mm_projector(image_feats_down)
        # print(f"\033[93m after mm projector {image_features.shape} \033[0m")
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images_feats, bboxes
    ):
        # Process for region
        image_features = self.v_l_projection(images_feats)
        # print(f"\033[94m image_features: {image_features.shape} \033[0m")

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids): # Adjusted the loop to include reg_feat
            # print(f"\033[92m cur_input_ids {cur_input_ids.shape} \033[0m")

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )

                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    # preparing input embedding
                    # mm_use_im_start_end: True
                    # tune_mm_mlp_adapter: False

                    # print(f"\033[93m cur_input_ids {cur_input_ids.shape} \033[0m")
                    # print(f"\033[93m image_token_start {image_token_start} \033[0m")

                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    # print(f"\033[94m cur_new_input_embeds[0] {cur_new_input_embeds[0].shape} \033[0m")
                    cur_new_input_embeds.append(cur_image_features)
                    # print(f"\033[94m cur_new_input_embeds[1] {cur_new_input_embeds[1].shape} \033[0m")

                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # print(f"\033[94m cur_new_input_embeds[2] {cur_new_input_embeds[2].shape} \033[0m")

                    # preparing input_ids
                    # print(f"\033[95m curr_full_input_ids[0] {curr_full_input_ids[0].shape} \033[0m")




                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)

                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]

                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            # print(f"\033[92m after while loop {cur_input_ids.shape} \033[0m")


            # pure text input_ids
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            # print(f"\033[94m -- cur_new_input_embeds {cur_new_input_embeds.shape} \033[0m")
            # print(f"\033[94m -- curr_full_input_ids {curr_full_input_ids.shape} \033[0m")

            # current new_input_embeds computation complete (Lx4096)
            # Replace embeds of <bbox> with region feats (num_box x 4096)


            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)



        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):

            # print(f"\033[91m different len \033[0m")
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            # print(f"\033[92m same len \033[0m")
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

