import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import io
import zipfile

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor


system_message = "You are an autonomous driving agent. Answer the question concisely without uncesessary details. Question: "

import os
import torch
import pandas as pd
from torch.utils.data import Dataset

def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    if needle.numel() == 0 or haystack.numel() < needle.numel():
        return -1
    # naive search is fine
    for i in range(haystack.numel() - needle.numel() + 1):
        if torch.equal(haystack[i:i+needle.numel()], needle):
            return i
    return -1


class QwenVLBatchCollator:
    def __init__(self, processor, system_message, max_length=None, verbose=False):
        self.processor = processor
        self.tok = processor.tokenizer
        self.system_message = system_message
        self.max_length = max_length
        self.verbose = verbose

    def __call__(self, batch):
        # Build messages for each sample
        msgs = []
        images_batch = []  # list-of-list, one entry per sample
        answers = []

        for ex in batch:
            imgs = ex["images"]
            q = ex["question"]
            a = ex["answer"]
            answers.append(a)

            m = [
                {
                    "role": "system",
                    "content": (
                        [{"type": "image", "image": im} for im in imgs] +
                        [{"type": "text", "text": self.system_message + q}]
                    ),
                },
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ]
            msgs.append(m)

            # IMPORTANT: keep per-sample image list (don’t flatten)
            # process_vision_info expects messages, returns image inputs for that sample
            img_inputs, _ = process_vision_info(m)
            images_batch.append(img_inputs)

        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in msgs
        ]

        # One processor call for the whole batch -> correct padding/stacking
        inputs = self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,               # let processor choose correct padding (left padding ok)
            truncation=False,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"]          # [B, T]
        attn = inputs["attention_mask"]          # [B, T]
        labels = torch.full_like(input_ids, -100)

        # Answer-only labels by matching answer token sequence inside full input_ids
        for i, ans in enumerate(answers):
            ans_ids = self.tok(ans, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

            real_pos = torch.nonzero(attn[i], as_tuple=False).squeeze(-1)
            real_ids = input_ids[i][real_pos]

            start_in_real = _find_subsequence(real_ids, ans_ids)

            # common template inserts a leading newline before answer
            if start_in_real == -1:
                for prefix in ["\n", " ", "\n\n"]:
                    alt = self.tok(prefix + ans, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    start_in_real = _find_subsequence(real_ids, alt)
                    if start_in_real != -1:
                        ans_ids = alt
                        break

            if start_in_real == -1:
                # Fail-safe: supervise nothing (better than supervising prompt)
                continue

            start = int(real_pos[start_in_real].item())
            end = min(start + ans_ids.numel(), input_ids.size(1))
            labels[i, start:end] = input_ids[i, start:end]

        labels[attn == 0] = -100
        inputs["labels"] = labels

        # Report supervised token counts for debugging/truncation checks
        sup_counts = [(labels[i] != -100).sum().item() for i in range(labels.size(0))]
        if self.verbose:
            for i, c in enumerate(sup_counts):
                print(f"[Collator] sample={i} supervised_tokens={c}")
                # Print detokenized supervised tokens for each sample
            for i in range(labels.size(0)):
                sup_mask = labels[i] != -100
                sup_ids = input_ids[i][sup_mask].tolist()
                detok = ""
                if len(sup_ids) > 0:
                    detok = self.tok.decode(sup_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    print(f"[Collator] sample={i} supervised_text: {detok}, input shape: {input_ids.shape}")
        return inputs


class LingoQADataset(Dataset):
    def __init__(self, pq_file, image_dir, limit = -1):
        self.df = pd.read_parquet(pq_file)
        self._image_dir = image_dir
        self._limit = limit

    def __len__(self):
        if self._limit == -1:
            return len(self.df)
        else:
            return min(len(self.df), self._limit)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        for p in row["images"]:
            # construct full path; p may already be absolute or include zip
            candidate = os.path.join(self._image_dir, p) if not os.path.isabs(p) and self._image_dir else p

            # If the path references a zip archive (e.g. /path/to/archive.zip/inner/path.jpg
            # or archive.zip::inner/path.jpg), read from the zip into a PIL.Image
            if ".zip" in candidate:
                # normalize patterns like 'archive.zip/inner/..' or 'archive.zip::inner/..'
                if "::" in candidate:
                    zip_path, inner = candidate.split("::", 1)
                else:
                    idx = candidate.find('.zip')
                    zip_path = candidate[: idx + 4]
                    inner = candidate[idx + 5 :].lstrip(os.sep)

                if not os.path.exists(zip_path):
                    raise FileNotFoundError(f"Zip archive not found: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    data = z.read(inner)
                img = Image.open(io.BytesIO(data)).convert('RGB')
                images.append(img)
            else:
                # regular file on disk
                full = candidate
                if not os.path.exists(full):
                    raise FileNotFoundError(f"Image file not found: {full}")
                img = Image.open(full).convert('RGB')
                images.append(img)
        return {
            "images": images,
            "question": row["question"],
            "answer": row["answer"],
        }


def test_l():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = Qwen2VLProcessor.from_pretrained(model_id,  max_pixels=322*511)
    dataset = LingoQADataset("./data/val/val.parquet", image_dir="./data/val/images.zip")
    print()

    # b = dataset[0]
    # print("supervised tokens:", (b["labels"] != -100).sum().item())

    # sup_ids = b["input_ids"][b["labels"] != -100].tolist()
    # print(processor.tokenizer.decode(sup_ids))

    collator = QwenVLBatchCollator(processor=processor, system_message="You are a helpful agent.")

    batch0 = collator([dataset[0], dataset[1]])
    for k, v in batch0.items():
        if hasattr(v, "shape"):
            print(k, v.shape)


test_l()