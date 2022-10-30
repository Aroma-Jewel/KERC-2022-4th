from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    PLM: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="checkpoints", metadata={"help": "Path to save checkpoint from fine tune model"},
    )

    head_class: str = field(
      default = "Org",
      metadata = {
        "help": "Concat_Hidden_States, ConvSDSHead, cls_sep, CLS_Weight"
      }
    )