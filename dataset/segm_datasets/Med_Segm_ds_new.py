import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, MED_SEG_QUESTIONS, GCG_QUESTIONS, ANSWER_LIST_CLS, MED_SEG_QUESTIONS_DES
import json
import albumentations as A
from pathlib import Path
from collections import Counter


def ct_png_window_bgr(
        img_bgr: np.ndarray,
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
        win_low: float = -75.0,
        win_high: float = 265.0,
) -> np.ndarray:
    """
    Apply HU windowing for CT images stored as 8-bit PNG (0..255).

    Assumption:
      png = (clip(HU, hu_min, hu_max) - hu_min) / (hu_max - hu_min) * 255

    Input:
      img_bgr: uint8 BGR from cv2.imread (H,W,3) or grayscale (H,W)

    Output:
      uint8 BGR (H,W,3) windowed to 0..255
    """
    if img_bgr is None:
        return img_bgr

    # Ensure float32
    if img_bgr.ndim == 3:
        # If it's a grayscale saved as 3-channel, channels are usually identical.
        # Use first channel to compute windowed grayscale.
        g = img_bgr[..., 0].astype(np.float32)
    else:
        g = img_bgr.astype(np.float32)

    # PNG -> HU
    hu = (g / 255.0) * (hu_max - hu_min) + hu_min

    # HU window
    hu_w = np.clip(hu, win_low, win_high)

    # HU -> PNG [0,255]
    out = (hu_w - win_low) / (win_high - win_low) * 255.0
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    # Back to BGR for downstream pipeline
    if img_bgr.ndim == 3:
        out_bgr = np.stack([out, out, out], axis=-1)
        return out_bgr
    else:
        return out


def train_transforms(max_size):
    transforms = []
    transforms.append(A.LongestMaxSize(int(max_size), interpolation=cv2.INTER_LINEAR))
    return A.Compose(transforms, p=1., is_check_shapes=False)


def convert_true_modality(modality):
    if 'mr' in modality:
        true_modality = 'magnetic resonance'
    elif 'ct' in modality:
        true_modality = 'computed tomography'
    elif modality == 'x':
        true_modality = 'X-Ray'
    else:
        true_modality = modality
    return true_modality


def _convert_dataset_paths(dataset_dict, root_dir):
    abs_image_paths = []
    abs_label_paths_text = []

    root_dir = os.path.abspath(root_dir)

    for img_path, label_list in dataset_dict.items():
        abs_img_path = os.path.abspath(os.path.join(root_dir, img_path))
        abs_image_paths.append(abs_img_path)

        cur_abs_label_list = []
        for label_item in label_list:
            abs_label_item = {}
            for mask_path, label_name in label_item.items():
                abs_mask_path = os.path.abspath(os.path.join(root_dir, mask_path))
                abs_label_item[abs_mask_path] = label_name
            cur_abs_label_list.append(abs_label_item)

        abs_label_paths_text.append(cur_abs_label_list)

    return abs_image_paths, abs_label_paths_text


class MedReferSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 8,
                 refer_segm_data="refcoco", validation=False, split='train', mode='', text_prompts_path='',
                 random_sampling=True, inference=False,
                 badcases_jsonl: str = "",
                 badcases_reason_allow: set = None):

        self.dataset_dir = os.path.abspath(dataset_dir)
        print(f"\033[93m --dataset_dir: {self.dataset_dir}\033[0m")

        self.image_size = image_size
        self.question_templates = MED_SEG_QUESTIONS
        self.question_templates_des = MED_SEG_QUESTIONS_DES
        self.question_templates_gcg = GCG_QUESTIONS
        self.answer_list_cls = ANSWER_LIST_CLS
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation
        self.random_sampling = random_sampling

        self.mask_num = num_classes_per_sample
        self.mode = mode

        if text_prompts_path is None:
            self.modality_class_text_prompts = None
        else:
            print(f"\033[91m {text_prompts_path} \033[0m")
            with open(text_prompts_path, 'r') as file:
                modality_class_text_prompts = json.load(file)
            self.modality_class_text_prompts = modality_class_text_prompts

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.extra_dataset_dir = "/mnt/ali/fmodimg/BiomedParseData"

        dataset = json.load(open(os.path.join(self.dataset_dir, f'image2label_{mode}.json'), "r"))

        main_image_paths, main_label_paths_text = _convert_dataset_paths(dataset, self.dataset_dir)

        self.image_paths = list(main_image_paths)
        self.label_paths_text = list(main_label_paths_text)

        if mode == "2d_train":
            extra_dataset = json.load(open('/mnt/ali/fmodimg/BiomedParseData/image2label_biomed_test.json', "r"))
            extra_image_paths, extra_label_paths_text = _convert_dataset_paths(
                extra_dataset, self.extra_dataset_dir
            )
            self.image_paths.extend(extra_image_paths)
            self.label_paths_text.extend(extra_label_paths_text)

        # Filter bad cases
        if badcases_jsonl is not None and os.path.isfile(badcases_jsonl):
            bad_set = set()
            bad_reason_cnt = {}

            with open(badcases_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    img_rel = rec.get("image_rel", None)
                    reason = rec.get("reason", "unknown")

                    if img_rel is None:
                        continue

                    if badcases_reason_allow is not None and reason not in badcases_reason_allow:
                        continue

                    img_abs = os.path.abspath(os.path.join(self.dataset_dir, img_rel))
                    bad_set.add(img_abs)
                    bad_reason_cnt[reason] = bad_reason_cnt.get(reason, 0) + 1

            if len(bad_set) > 0:
                new_image_paths = []
                new_label_paths_text = []
                for p, e in zip(self.image_paths, self.label_paths_text):
                    if p not in bad_set:
                        new_image_paths.append(p)
                        new_label_paths_text.append(e)

                removed = len(self.image_paths) - len(new_image_paths)
                self.image_paths = new_image_paths
                self.label_paths_text = new_label_paths_text

                print(f"\033[91m[MedReferSegmDataset] Filter badcases from: {badcases_jsonl}\033[0m")
                print(
                    f"\033[91m[MedReferSegmDataset] Removed {removed} / {removed + len(self.image_paths)} samples.\033[0m")
                top = sorted(bad_reason_cnt.items(), key=lambda x: -x[1])[:20]
                print(f"\033[91m[MedReferSegmDataset] Top bad reasons: {top}\033[0m")
        else:
            if badcases_jsonl is not None:
                print(f"\033[93m[MedReferSegmDataset] badcases_jsonl not found: {badcases_jsonl}\033[0m")

        # Dataset statistics (after optional merge/filter)
        total_images = len(self.image_paths)
        total_masks = 0
        empty_label_images = 0
        unique_mask_paths = set()
        all_mask_paths = []

        for labels in self.label_paths_text:
            if (not isinstance(labels, list)) or (len(labels) == 0):
                empty_label_images += 1
                continue

            total_masks += len(labels)
            for item in labels:
                if isinstance(item, dict) and len(item) > 0:
                    mask_path = list(item.keys())[0]
                    unique_mask_paths.add(mask_path)
                    all_mask_paths.append(mask_path)

        print(
            f"\033[96m[MedReferSegmDataset][mode={self.mode}] "
            f"images={total_images}, masks={total_masks}, "
            f"unique_masks={len(unique_mask_paths)}, "
            f"empty_label_images={empty_label_images}\033[0m"
        )

        image_counter = Counter(self.image_paths)
        mask_counter = Counter(all_mask_paths)
        duplicate_images = [(p, c) for p, c in image_counter.items() if c > 1]
        duplicate_masks = [(p, c) for p, c in mask_counter.items() if c > 1]
        duplicate_images.sort(key=lambda x: (-x[1], x[0]))
        duplicate_masks.sort(key=lambda x: (-x[1], x[0]))

        print(
            f"\033[96m[MedReferSegmDataset][mode={self.mode}] "
            f"duplicate_images={len(duplicate_images)}, duplicate_masks={len(duplicate_masks)}\033[0m"
        )

        max_print = 5
        if len(duplicate_images) > 0:
            print(f"\033[95m[MedReferSegmDataset] Top-{max_print} duplicate image examples:\033[0m")
            for path, cnt in duplicate_images[:max_print]:
                print(f"\033[95m  - count={cnt}, image={path}\033[0m")

        if len(duplicate_masks) > 0:
            print(f"\033[95m[MedReferSegmDataset] Top-{max_print} duplicate mask examples:\033[0m")
            for path, cnt in duplicate_masks[:max_print]:
                print(f"\033[95m  - count={cnt}, mask={path}\033[0m")

        self.transform = train_transforms(image_size)

    def __len__(self):
        return len(self.image_paths)

    def extract_foreground_regions(self, image, masks):
        mask, _ = torch.max(masks, dim=0, keepdim=True)

        masked_image = image * mask
        return masked_image

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def mask_processor(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations_cls(self, labels, modality, descriptions, is_val):
        questions = []
        # descriptions_list = descriptions.split('\n')
        # print(f"\033[91m {descriptions_list} \033[0m")
        link_str = ', '
        labels_str = [label.strip().replace("_", " ") for label in labels]
        labels_str = link_str.join(labels_str)
        # print(f"\033[92m {labels_str} \033[0m")

        question_template = random.choice(self.question_templates)
        # description = random.choice(description_aspects)
        # print(f"\033[91m {len(description)} \033[0m")
        questions.append(
            question_template.format(class_name=labels_str.lower(), modality=modality, description=descriptions))

        processed_labels = [f"<p> {l.strip().replace('_', ' ')} </p> [SEG]" for l in labels]

        if len(processed_labels) > 1:
            # 处理 "A, B, and C" 的逻辑
            main_part = ", ".join(processed_labels[:-1])
            answer = f"This is a <p> {modality} </p> image. The image contains {main_part} and {processed_labels[-1]}."
        elif len(processed_labels) == 1:
            # 只有一个标签的情况
            answer = f"This is a <p> {modality} </p> image. The image contains {processed_labels[0]}."
        else:
            # 没有标签的情况
            answer = f"This is a <p> {modality} </p> image."

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + questions[0])
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        # print(f"\033[94m {caption} \033[0m")
        # print(f"\033[92m {tokens_positive} \033[0m")

        return questions, conversations

    def create_conversations_internvl_des(self, labels, modality, descriptions, is_val):
        questions = []
        # descriptions_list = descriptions.split('\n')
        # print(f"\033[91m {descriptions_list} \033[0m")
        link_str = ', '
        labels_str = [label.strip().replace("_", " ") for label in labels]
        labels_str = link_str.join(labels_str)
        # print(f"\033[92m {labels_str} \033[0m")

        question_template = random.choice(self.question_templates_des)
        # description = random.choice(description_aspects)
        # print(f"\033[91m {len(description)} \033[0m")
        questions.append(
            question_template.format(class_name=labels_str.lower(), modality=modality, description=descriptions))

        # 预处理标签：去除空格并替换下划线
        processed_labels = [f"<p> {l.strip().replace('_', ' ')} </p> [SEG]" for l in labels]

        if len(processed_labels) > 1:
            # 处理 "A, B, and C" 的逻辑
            main_part = ", ".join(processed_labels[:-1])
            answer = f"This is a <p> {modality} </p> image. The image contains {main_part} and {processed_labels[-1]}."
        elif len(processed_labels) == 1:
            # 只有一个标签的情况
            answer = f"This is a <p> {modality} </p> image. The image contains {processed_labels[0]}."
        else:
            # 没有标签的情况
            answer = f"This is a <p> {modality} </p> image."

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + questions[0])
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        # print(f"\033[94m {caption} \033[0m")
        # print(f"\033[92m {tokens_positive} \033[0m")

        return questions, conversations

    def create_conversations_gcg(self, labels, modality, caption, is_val):
        """
        GCG: ask for detailed description, answer contains:
          1) modality header + interleaved [SEG] for each label
          2) appended caption text from modality_class_text_prompts (if provided)
        """
        # ---- Question ----
        question_template = random.choice(self.question_templates_gcg)  # GCG_QUESTIONS
        question = question_template

        # ---- Answer (your required format) ----
        # 预处理标签：去除空格并替换下划线
        processed_labels = [f"<p> {l.strip().replace('_', ' ')} </p> [SEG]" for l in labels]

        if len(processed_labels) > 1:
            # 处理 "A, B, and C" 的逻辑
            main_part = ", ".join(processed_labels[:-1])
            answer = f"This is a <p> {modality} </p> image. The image contains {main_part} and {processed_labels[-1]}."
        elif len(processed_labels) == 1:
            # 只有一个标签的情况
            answer = f"This is a <p> {modality} </p> image. The image contains {processed_labels[0]}."
        else:
            # 没有标签的情况
            answer = f"This is a <p> {modality} </p> image."

        # ---- Append caption (from modality_class_text_prompts) ----
        if caption is None:
            caption = ""
        if isinstance(caption, (list, tuple)):
            # 有些 json 可能是 list of sentences
            caption = " ".join([str(x) for x in caption if x is not None])
        else:
            caption = str(caption)

        caption = caption.strip()
        if len(caption) > 0:
            # 你可以把前缀改成更符合你的训练风格
            answer = answer + " " + caption

        # ---- Build conversation ----
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())

        return [question], conversations

    def __getitem__(self, idx):
        # dataset_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
        # print(f"\033[91m---dataset_idx: {dataset_idx}")
        # print(f"\033[91m---dataset_name: {dataset_name}")
        # refer_seg_ds = self.refer_segm_data
        # print(f"\033[91m---refer_seg_ds: {refer_seg_ds['annotations'][list(refer_seg_ds['annotations'].keys())[0]]}\033[0m")
        # print(f"\033[91m---refer_seg_ds: {self.label_paths_text[idx]}")

        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(self.image_paths) - 1)

        # print(f"\033[91m {idx} \033[0m")
        if len(self.label_paths_text[idx]) == 0:
            print("\033[93m[FALLBACK] idx={} reason=empty_label image={}\033[0m".format(idx, self.image_paths[idx]))
            return self.__getitem__(idx + 1)

        modality = self.image_paths[idx].split('/')[-1].split('--')[0]

        # modality = "ct"

        # print(f"\033[92m modality {modality} \033[0m")
        # print(f"\033[94m {self.label_paths_text[idx]} \033[0m")

        # Set paths
        mask_paths = [list(label_path.keys())[0] for label_path in self.label_paths_text[idx]]
        text_labels_all = [list(label_path.values())[0] for label_path in self.label_paths_text[idx]]
        # combined_list = list(zip(label_paths, text_labels_all))
        # print('combined_list', combined_list)

        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if "BiomedParseData" in image_path:
            fname_stem = image_path.split("/")[-1].rsplit(".", 1)[0]
            parts = fname_stem.split("_")
            modality = parts[-2].lower() if len(parts) >= 2 else ""
            if modality in ("ct", "mri"):
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # print(f"\033[91m ---- {image.shape} \033[0m")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ---------- NEW: CT windowing for PNG ----------
        # modality extracted earlier:
        # modality = self.image_paths[idx].split('/')[-1].split('--')[0]
        if image is None:
            print("\033[93m[FALLBACK] idx={} reason=image_none image={}\033[0m".format(idx, self.image_paths[idx]))
            return self.__getitem__(idx + 1)

        # if ('ct' in modality.lower()):
        #    image = ct_png_window_bgr(image, hu_min=-1000.0, hu_max=400.0, win_low=-75.0, win_high=265.0)

        merged_dict = {}
        merged_mask_paths = {}
        for label, mask_path in zip(text_labels_all, mask_paths):
            # print(f"\033[91m {label} \033[0m")

            label = label.replace('gall_bladder', 'gallbladder')
            if self.mode == "22cancers" or self.mode == "oneshot":
                if "Lung" not in label:
                    label = label.replace('cancer', 'tumor')
            mask = cv2.imread(os.path.join(mask_path), 0)
            # print(f"\033[91m +++ {mask.shape} \033[0m")
            if mask is None:
                print("\033[93m[FALLBACK] idx={} reason=mask_none mask={}\033[0m".format(idx, mask_path))
                return self.__getitem__(idx + 1)

            if mask.max() == 0:
                print("\033[93m[FALLBACK] idx={} reason=mask_zero mask={}\033[0m".format(idx, mask_path))
                return self.__getitem__(idx + 1)
            if mask.max() > 1:
                mask = mask / mask.max()
            # if mask.max() != 1 and  mask.max() > 0:
            #    print("error", mask.max())
            mask = np.clip(mask, 0, 1)

            if "BiomedParseData" in image_path:
                if modality in ("ct", "mri"):
                    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

            if label not in merged_dict:
                merged_dict[label] = mask
                merged_mask_paths[label] = [mask_path]
            else:
                try:
                    merged_dict[label] = np.clip(merged_dict[label] + mask, 0, 1)
                    merged_mask_paths[label].append(mask_path)
                except:
                    print("\033[93m[FALLBACK] idx={} reason=merge_except image={}\033[0m".format(idx, self.image_paths[idx]))
                    return self.__getitem__(idx + 1)

        selected_labels = list(merged_dict.keys())
        masks_list = list(merged_dict.values())
        selected_mask_paths = [merged_mask_paths[l][0] if len(merged_mask_paths.get(l, [])) > 0 else None
                               for l in selected_labels]

        combined_list = list(zip(masks_list, selected_labels, selected_mask_paths))
        if len(combined_list) > self.mask_num and not self.validation:
            skip_des = True
            combined_list = random.choices(combined_list, k=self.mask_num)
            masks_list, selected_labels, selected_mask_paths = zip(*combined_list)
            masks_list = list(masks_list)
            selected_labels = list(selected_labels)
            selected_mask_paths = list(selected_mask_paths)
        else:
            skip_des = False

        masks_np = np.stack(masks_list, axis=0)
        masks = torch.from_numpy(masks_np)
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL

        # print(f"\033[92m === {masks.shape, label.shape} \033[0m")

        # for SAM input
        try:
            image = self.transform(image=image)['image']
        except:
            print("\033[93m[FALLBACK] idx={} reason=transform_except image={}\033[0m".format(idx, self.image_paths[idx]))
            return self.__getitem__(idx + 1)
        # print(f"\033[94m transformed image {image.shape} \033[0m")
        # print(f"\033[95m transformed masks_np {aug_masks_list.shape} \033[0m")
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # print(f"\033[91m grounding_enc_img {grounding_enc_img.max(), grounding_enc_img.min()} \033[0m")

        if torch.isnan(grounding_enc_img).any():
            print("\033[93m[FALLBACK] idx={} reason=nan_enc image={}\033[0m".format(idx, self.image_paths[idx]))
            return self.__getitem__(idx + 1)

        # prompts = None
        # print(f"\033[92m {prompts} \033[0m")
        # for text_label in selected_labels:
        # print("text_label",text_label)
        # prompts.append(list(self.modality_class_text_prompts[modality][text_label].values()))

        true_modality = convert_true_modality(modality)
        # print(f"\033[92m {true_modality} \033[0m")
        # Generate questions and answers
        # if not self.validation:

        # if true_modality == "pathology":
        #    dataset = self.image_paths[idx].split('/')[-1].split('--')[1]
        #    #if dataset == "hubmap_organ" or dataset == "cpm15" or dataset == "cpm17" or dataset == "bchi" or dataset == "crag":
        #    if dataset == "pannuke" or dataset == "hubmap_ext":
        #        return self.__getitem__(idx+1)

        if self.modality_class_text_prompts is not None and not skip_des and self.image_paths[
            idx] in self.modality_class_text_prompts:
            prompts = self.modality_class_text_prompts[self.image_paths[idx]]

            use_gcg = (random.random() < 0.25) and not self.validation
            if use_gcg:
                questions, conversations = self.create_conversations_gcg(
                    selected_labels, true_modality, prompts, self.validation
                )
            else:
                questions, conversations = self.create_conversations_internvl_des(
                    selected_labels, true_modality, prompts, self.validation
                )
        else:
            prompts = ''
            questions, conversations = self.create_conversations_cls(
                selected_labels, true_modality, prompts, self.validation
            )

        # else:
        #     prompts = ''
        #     questions, conversations = self.create_conversations_cls(selected_labels, true_modality, prompts,
        #                                                              self.validation)
        # set bboxes to None for segmentation datasets
        bboxes = None

        # print(f"\033[93m---image_path: {image_path}")
        # print(f"\033[93m---grounding_enc_img: {grounding_enc_img.shape}\033[0m")
        # print(f"\033[93m---image_resize: {image_resize}\033[0m")
        # print(f"\033[94m---selected_labels: {selected_labels}\033[0m")
        # print(f"\033[96m---prompts: {len(prompts), prompts}\033[0m")
        # print(f"\033[93m---questions: {len(questions), questions}\033[0m")
        # print(f"\033[94m---masks: {len(masks)}\033[0m")
        # print(f"\033[92m---masks: {masks.shape}\033[0m")
        # print(f"\033[96m---conversations: {conversations}\033[0m")
        # print(f"\033[96m---conversations: {len(conversations[0])}\033[0m")

        return (image_path, grounding_enc_img, bboxes, conversations, masks, label,
                image_resize, questions, selected_labels, selected_mask_paths)
