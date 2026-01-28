import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Trainer, TrainingArguments, TrainerCallback

from peft import get_peft_model
from dataset import LingoQADataset, QwenVLBatchCollator
from utils import clear_memory, get_lora_config, Config, run_example
import argparse



def load_model_and_processor(config):
    model_cfg = config.get_section("model")
    model_id = model_cfg.get("id")
    device = torch.device(model_cfg.get("device", "cuda"))
    dtype = torch.bfloat16 if model_cfg.get("dtype") == 'bfloat16' else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=dtype
    )
    max_pixels = model_cfg.get("max_pixels", 322*511)
    processor = Qwen2VLProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    return model, processor, device


def load_datasets(config, processor):
    ds_cfg = config.get_section("datasets")
    system_message = ds_cfg["system_message"]
    train_dataset = LingoQADataset(ds_cfg.get("train_file", "./data/val/val.parquet"), image_dir=ds_cfg.get("train_image_dir", "./data/val/"))
    val_dataset = LingoQADataset(
        ds_cfg.get("val_file", "./data/val/val.parquet"),
        image_dir=ds_cfg.get("val_image_dir", "./data/val/"),
        limit=ds_cfg.get("val_limit", -1)
    )
    collator = QwenVLBatchCollator(processor, system_message)
    return train_dataset, val_dataset, collator


def setup_peft(model, config):
    peft_config = get_lora_config(config=config)
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def get_training_args(config):
    training_cfg = config.get_section("training")
    # Ensure evaluation settings exist: default to evaluating every save_steps if not explicitly set
    args_dict = dict(training_cfg)
    # if "evaluation_strategy" not in args_dict:
    #     if args_dict.get("save_steps"):
    #         args_dict["evaluation_strategy"] = "steps"
    #         args_dict["eval_steps"] = args_dict.get("save_steps")
    return TrainingArguments(**args_dict)



class ShowExamplesCallback(TrainerCallback):
    """Trainer callback that prints a few validation examples during evaluation."""
    def __init__(self, processor, val_dataset, device, n_examples=3, max_new_tokens=100, system_message=""):
        self.processor = processor
        self.val_dataset = val_dataset
        self.device = device
        self.n_examples = n_examples
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.system_message = system_message

    def on_train_begin(self, args, state, control, **kwargs):
        self.model = kwargs.get("model", None)

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model", self.model)
        if model is None:
            return
        try:
            model.eval()
            ds = self.val_dataset
            print("[ShowExamplesCallback] Showing examples from validation set:")
            for i in range(min(self.n_examples, len(ds))):
                item = ds[i]
                with torch.no_grad():
                    _ = run_example(
                        model, self.processor, item, self.device, self.system_message,
                        max_new_tokens=self.max_new_tokens, print_result=True)
        except Exception as e:
            print(f"[ShowExamplesCallback] example printing failed: {e}")


def main(args):
    # Load config (from provided path if given)
    config = Config(args.config)

    # Allow overriding training.output_dir via CLI
    if args.output_dir:
        config.config.setdefault("training", {})["output_dir"] = args.output_dir

    model, processor, device = load_model_and_processor(config)
    train_dataset, val_dataset, collator = load_datasets(config, processor)
    peft_model = setup_peft(model, config)
    training_args = get_training_args(config)
    # Optionally add callback to show example predictions during evaluation
    callbacks = None
    if getattr(args, "show_examples", False):
        cb = ShowExamplesCallback(
            processor, val_dataset, device, n_examples=getattr(args, "show_examples_n", 3),
            system_message=config.get_section("datasets")["system_message"]
        )
        callbacks = [cb]

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    clear_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen2-VL LoRA on LingoQA")
    parser.add_argument("--config", type=str, default="config_debug.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override training.output_dir")
    parser.add_argument("--show-examples", action="store_true", help="Print example predictions during evaluation")
    parser.add_argument("--show-examples-n", type=int, default=10, help="Number of examples to print during evaluation")
    args = parser.parse_args()
    main(args)
