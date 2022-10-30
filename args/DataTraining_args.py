from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
  max_length: int = field(
    default=512, metadata={"help": "Max length of input sequence"},
  )

  data_dir: str = field(
    default='./Datasets', metadata={"help": "Datasets directory"},
  )

  data_type: str = field(
    default='train', metadata={"help": "Datasets type"},
  )

  use_rtt_data: bool = field(
    default=False, metadata={"help":"use rtt dataset"}
  )

  past_sentence : int = field(
    default=6, metadata={"help" : "number of contain past length"}
  )

  preprocess_version : str = field(
    default='v5', metadata = {"help" : "v1,v2,v3,v4 ... reference : notion"}
  )

  use_substitute_preprocess : bool = field(
    default=False, metadata = {"help" : "v0,v1,v2 .. / v2 : whole test(hanspell,[UNK]) / reference : notion"}
  )

  use_pykospacing : bool = field(
    default=False, metadata = {"help" : 'Applying pykospacing to sentence & context'}
  )