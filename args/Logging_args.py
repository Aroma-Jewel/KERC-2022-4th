from typing import Optional
from dataclasses import dataclass, field

@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="./wandb.env", metadata={"help": "input your dotenv path"},
    )
    project_name: Optional[str] = field(
        default="KERC", metadata={"help": "project name"},
    )
    group_name: Optional[str] = field(
        default="reproduction", metadata={"help": "group name"},
    )