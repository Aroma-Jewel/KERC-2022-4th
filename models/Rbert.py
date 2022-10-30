import torch
from torch import nn
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


class FCLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.dropout_rate = 0.1


        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

        self.post_init()

    def entity_average(self, hidden_output, e_mask):

        e_mask_unsqueeze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
        
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector
  
    def forward(
          self,
          input_ids: Optional[torch.LongTensor] = None,
          attention_mask: Optional[torch.FloatTensor] = None,
          subject_mask=None,
          object_mask=None,
          subject_special_token_index = None,
          object_special_token_index = None,
          token_type_ids: Optional[torch.LongTensor] = None,
          position_ids: Optional[torch.LongTensor] = None,
          head_mask: Optional[torch.FloatTensor] = None,
          inputs_embeds: Optional[torch.FloatTensor] = None,
          labels: Optional[torch.LongTensor] = None,
          output_attentions: Optional[bool] = None,
          output_hidden_states: Optional[bool] = None,
          return_dict: Optional[bool] = None,
      ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        sequence_output = outputs["last_hidden_state"]
        pooled_output = sequence_output[:, 0, :]

        
        subject_h = self.entity_average(
            sequence_output, subject_mask
        )
        object_h = self.entity_average(
            sequence_output, object_mask
        )
        
        pooled_output = self.cls_fc_layer(
            pooled_output
        ) 
        
        subject_h = self.entity_fc_layer(
            subject_h
        )
        object_h = self.entity_fc_layer(
            object_h
        )
        
        
        concat_hidden = torch.cat([pooled_output, subject_h, object_h], dim=-1)
        logits = self.label_classifier(concat_hidden)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
          loss=loss,
          logits=logits,
          hidden_states=outputs.hidden_states,
          attentions=outputs.attentions,
        )
