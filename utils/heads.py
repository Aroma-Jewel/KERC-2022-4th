import torch
import string
import collections
import numpy as np
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if self.config.head_class == "CLS_Weight":
          x = features
        else:
          x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Hidden_States_Outputs(nn.Module):
  def __init__(self, config, num_labels):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size*4, config.hidden_size*4).cuda()
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )

    self.dropout = nn.Dropout(classifier_dropout)
    self.out_proj = nn.Linear(config.hidden_size*4, num_labels).cuda()
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    x = self.dropout(x)
    x = self.dense(x)
    x = self.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x


class Concat_Hidden_States(nn.Module):
  def __init__(self, config, num_labels: int = 3):
    super().__init__()

    self.hidden_concat_outputs = Hidden_States_Outputs(config, config.num_labels)
    self.out_proj = nn.Linear(config.hidden_size*4, config.num_labels).cuda()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    all_hidden_states = torch.stack(x)
    
    stacked_output = torch.cat((all_hidden_states[-4],all_hidden_states[-3], all_hidden_states[-2], all_hidden_states[-1]),-1)
    sequence_stacked_output = stacked_output[:,0]
    
    logits = self.out_proj(sequence_stacked_output)
    return logits


# SDS Conv
class ConvSDSLayer(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=input_size * 2, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(in_channels=input_size * 2, out_channels=input_size, kernel_size=1,)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + self.activation(out)
        out = self.layer_norm(out)
        return out


class ConvSDSHead(nn.Module):
    def __init__(
        self, config, num_labels: int = 3
    ):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, num_labels).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convs = []
        for n in range(5):
            convs.append(ConvSDSLayer(len(x[0]), self.config.hidden_size).cuda())
        self.convs = nn.Sequential(*convs)
        out = self.convs(x)
        return self.classifier(out)[:,0,:]


class cls_sep(nn.Module):
    def __init__(
        self, config, num_labels: int = 3
    ):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        ).cuda()

        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels).cuda()

    def forward(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_size = input_ids.shape
        last_hidden_states = x

        cls_flag = input_ids == 0 # tokenizer cls token
        sep_flag = input_ids == 2 # tokenizer sep toen

        sep_token_states = last_hidden_states[cls_flag + sep_flag]
        sep_token_states = sep_token_states.view(batch_size, -1, self.config.hidden_size)
        sep_hidden_states = self.net(sep_token_states)

        pooled_output = sep_hidden_states.view(batch_size, -1)
        logits = self.classifier(pooled_output)
        
        return logits

class CLS_Weight(nn.Module):
  def __init__(
        self, config, num_labels: int = 3
    ):
    super().__init__()
    self.config = config
    self.classifier = RobertaClassificationHead(self.config).cuda()
        

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    hidden_states = x

    cls_output = hidden_states[-1][:,0] * 0.6
    midterm_output1 = hidden_states[-2][:,0] * 0.3
    midterm_output2 = hidden_states[-3][:,0] * 0.1
    
    pooled_output = cls_output + midterm_output1 + midterm_output2

    logits = self.classifier(pooled_output)
    return logits

class Org(nn.Module):
  def __init__(
        self, config, num_labels: int = 3
    ):
    super().__init__()
    self.config = config
    self.classifier = RobertaClassificationHead(self.config).cuda()
        

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    sequence_output = x
    logits = self.classifier(sequence_output)
    return logits
