import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor


system_message = "You are an autonomous driving agent. Answer the question concisely without uncesessary details. Question: "


class LingoQADataset(Dataset):
    def __init__(self, pq_file, image_dir, processor):
        self.df = pd.read_parquet(pq_file)
        self._n = self.df.shape[0]
        self._image_dir = image_dir
        self._processor = processor

    def __len__(self):
        return self._n

    # def preprocess(self, question, images):
    #     sample = {
    #         "images": images,
    #         "messages": [
    #             {
    #                 "role": "system",
    #                 "content": [
    #                     {"type": "text", "text": system_message + question}
    #                 ] + [
    #                     {"type": "image", "image": os.path.join(self._image_dir, img)}
    #                     for img in images
    #                 ],
    #             },            
    #         ]
    #     }

    #     text_input = self._processor.apply_chat_template(
    #         sample["messages"],  # Use the sample without the system message
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )

    #     # Process the visual input from the sample
    #     image_inputs, _ = process_vision_info(sample["messages"])

    #     model_inputs = self._processor(
    #             text=[text_input],
    #             images=image_inputs,
    #             return_tensors="pt",
    #     )
    #     return {key: model_inputs[key] for key in model_inputs.keys()}
    
    # def __getitem__(self, idx):
    #     images = self.df.iloc[idx]["images"]
    #     question = self.df.iloc[idx]["question"]
    #     answer = self.df.iloc[idx]["answer"]

    #     input_dict = self.preprocess(question, images)
    #     input_dict["labels"] = self._processor.tokenizer.encode(
    #         answer,
    #         return_tensors="pt")
    #     return input_dict
    def __getitem__(self, idx):
        images = self.df.iloc[idx]["images"]
        question = self.df.iloc[idx]["question"]
        answer = self.df.iloc[idx]["answer"]
        images = [os.path.join(self._image_dir, img) for img in images[:1]]
        sample = {
            "images": images,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_message + question}
                    ] + [
                        {"type": "image", "image": img}
                        for img in images
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ],
                },   
            ]
        }
        return sample


def test_l():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    dataset = LingoQADataset("./data/val/val.parquet", image_dir="./data/val/", processor=processor)
    print(dataset[0])


test_l()