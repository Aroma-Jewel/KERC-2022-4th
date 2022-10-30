import math
import torch
import importlib
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.heads import ConvSDSHead

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # if explainable model
        x = features
        # not explainable model
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class RobertaForSequenceClassification(RobertaPreTrainedModel):
  _keys_to_ignore_on_load_missing = [r"position_ids"]

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.config = config

    self.roberta = RobertaModel(config, add_pooling_layer=False)

    self.head_file = importlib.import_module("utils.heads")
    self.head_class = getattr(self.head_file, config.head_class)
    self.head_class_forward = self.head_class(self.config, self.config.num_labels)


    # Initialize weights and apply final processing
    self.post_init()
  
  def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
      r"""
      labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
          Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
          config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
          `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
      """
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict
      output_hidden_states = (
            True
            if "hidden" in self.config.head_class.lower() or "CLS_Weight" in self.config.head_class
            else output_hidden_states
        )
      outputs = self.roberta(
          input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids,
          position_ids=position_ids,
          head_mask=head_mask,
          inputs_embeds=inputs_embeds,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )

      if "cls" in self.config.head_class:
        logits = self.head_class_forward(
          input_ids, outputs[0]
        )
      
      else:
        logits = self.head_class_forward(
          outputs.hidden_states
          if "hidden" in self.config.head_class.lower() or "CLS_Weight" in self.config.head_class
          else outputs[0]
        )

      outputs.hidden_states = None

      loss = None
      if labels is not None:
          if self.config.problem_type is None:
              if self.num_labels == 1:
                  self.config.problem_type = "regression"
              elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                  self.config.problem_type = "single_label_classification"
              else:
                  self.config.problem_type = "multi_label_classification"

          if self.config.problem_type == "regression":
              loss_fct = MSELoss()
              if self.num_labels == 1:
                  loss = loss_fct(logits.squeeze(), labels.squeeze())
              else:
                  loss = loss_fct(logits, labels)
          elif self.config.problem_type == "single_label_classification":
              loss_fct = CrossEntropyLoss()
              loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          elif self.config.problem_type == "multi_label_classification":
              loss_fct = BCEWithLogitsLoss()
              loss = loss_fct(logits, labels)

      if not return_dict:
          output = (logits,) + outputs[2:]
          return ((loss,) + output) if loss is not None else output

      return SequenceClassifierOutput(
          loss=loss,
          logits=logits,
          hidden_states=outputs.hidden_states,
          attentions=outputs.attentions,
      )
