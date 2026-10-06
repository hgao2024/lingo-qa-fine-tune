
import gc
import time
import torch
from peft import LoraConfig
import yaml



class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_section(self, section):
        return self.config.get(section, {})


def clear_memory():
    """
    Clean up GPU memory by deleting common variables and running garbage collection.
    Prints allocated and reserved GPU memory after cleanup.
    """
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def get_lora_config(r=None, lora_alpha=None, lora_dropout=None, target_modules=None, config: 'Config'=None):
    """
    Create LoRA configuration for Qwen2-VL model fine-tuning.
    If a Config object is provided, use its values as defaults.
    """
    if config is not None:
        r = r if r is not None else config.get("lora.r", 16)
        lora_alpha = lora_alpha if lora_alpha is not None else config.get("lora.lora_alpha", 16)
        lora_dropout = lora_dropout if lora_dropout is not None else config.get("lora.lora_dropout", 0.05)
        target_modules = target_modules if target_modules is not None else config.get("lora.target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mlp.0", "mlp.2"
        ])
    else:
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mlp.0", "mlp.2"
            ]
        r = 16 if r is None else r
        lora_alpha = 16 if lora_alpha is None else lora_alpha
        lora_dropout = 0.05 if lora_dropout is None else lora_dropout
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


def run_example(model, processor, sample_row, device, system_message="", max_new_tokens=100, print_result=True):
    """Run inference for a single (question + images) example and return the generated text.

    Args:
        model: vision-language model
        processor: model processor
        question: question string
        images: list of PIL.Image or image paths (processor/process_vision_info handles images)
        device: torch device
        system_message: optional system message prefix
        max_new_tokens: generation length

    Returns:
        Decoded generated string
    """
    from qwen_vl_utils import process_vision_info

    model.eval()

    # Expect `sample_row` to be a dict-like with keys: 'images', 'question'
    question = sample_row.get("question") if isinstance(sample_row, dict) else getattr(sample_row, "question", None)
    images = sample_row.get("images") if isinstance(sample_row, dict) else getattr(sample_row, "images", None)

    # Build messages expected by processor and process_vision_info
    messages = [
        {
            "role": "system",
            "content": (
                [{"type": "image", "image": im} for im in images] +
                [{"type": "text", "text": system_message + question}]
            ),
        }
    ]

    # Apply chat template and process vision inputs
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    # Prepare model inputs
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # Generate
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim input ids from output (assumes batch size 1)
    in_ids = model_inputs.input_ids[0]
    out_ids = generated_ids[0]
    if out_ids.numel() <= in_ids.numel():
        trimmed = out_ids.new_empty((0,))
    else:
        trimmed = out_ids[in_ids.numel():]

    # Decode
    output_text = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pred = output_text[0] if isinstance(output_text, (list, tuple)) else str(output_text)

    # Optionally print question / prediction / ground-truth
    if print_result:
        gt = None
        if isinstance(sample_row, dict):
            gt = sample_row.get("answer")
        else:
            gt = getattr(sample_row, "answer", None)
        print("---")
        print(f"Question: {question}")
        print(f"Prediction: {pred}")
        if gt is not None:
            print(f"Ground truth: {gt}")

    return pred
