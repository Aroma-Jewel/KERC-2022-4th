from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class MyTrainingArguments(TrainingArguments):
    max_answer_length: Optional[int] = field(
        default=30, metadata={"help": "Maximum length of answer after post processing"}
    )

    output_dir: str = field(
        default='./exps', metadata={"help": "checkpoint save directory"}
    )

    use_rdrop: bool = field(
      default=False, metadata={"help":"use r-drop"}
    )

    use_Smart_loss: bool = field(
      default=True, metadata={"help":"use SMART Loss"}
    )

    reg_alpha: float = field(
        default=0.7,
        metadata={
            "help": "alpha value for regularized dropout(default: 0.7)"
        },
    )

    multiple_weight: str = field(
      default=None, metadata={"help":"several model predict (soft voting)"}
    )

    use_special_tokens: bool = field(
      default=False, metadata={"help":"use special tokens"}
    )

    use_RBERT: bool = field(
      default=False, metadata={"help":"use RBERT"}
    )

    use_kfold: bool = field(
      default=True, metadata = {"help":"want to train kfold"}
    )
    
    loss_name: str = field(
      default='CrossEntropy', metadata = {"help":"want to loss function"}
    )

    
