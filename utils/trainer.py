import pickle as pickle
import os
import pandas as pd
import sklearn
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers.trainer_pt_utils import nested_detach
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers import Trainer

from typing import Callable
from torch import Tensor
from itertools import count 
from torch.utils.checkpoint import checkpoint


def kl_loss(inputs, target, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(inputs, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )

def sym_kl_loss(input, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed: Tensor, state: Tensor) -> Tensor:
        noise = torch.randn_like(embed, requires_grad=True) * self.noise_var

        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.eval_fn(embed_perturbed)
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)

            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise

            noise_gradient, = torch.autograd.grad(outputs = loss, inputs = noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()


class Smart_Trainer(Trainer):
  
  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

      model.train()
      inputs = self._prepare_inputs(inputs)
      
      loss = self.compute_loss(model, inputs)

      if self.args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training

      if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
          # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
          loss = loss / self.args.gradient_accumulation_steps

      if self.deepspeed:
          # loss gets scaled under gradient_accumulation_steps in deepspeed
          loss = self.deepspeed.backward(loss)
      else:
          loss.backward()

      return loss.detach()
  
  def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
      """
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
      """

      # if self.label_smoother is not None and "labels" in inputs:
      if "labels" in inputs:
          labels = inputs['labels'] # inputs.pop("labels")
          pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
      else:
          labels = None
      
      outputs = model(**inputs)

      embed = self.model.roberta.embeddings(inputs.input_ids)
      def eval_fn(embed):
        outputs = self.model.roberta(inputs_embeds=embed, attention_mask=inputs.attention_mask)
        if self.args.RBERT:
          def entity_average(hidden_output, e_mask):
            """
            Average the entity hidden state vectors (H_i ~ H_j)
            :param hidden_output: [batch_size, j-i+1, dim]
            :param e_mask: [batch_size, max_seq_len]
                    e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
            :return: [batch_size, dim]
            """

            e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
            length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
            
            # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
            sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
            avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
            return avg_vector
          sequence_output = outputs["last_hidden_state"]
          pooled_output = sequence_output[:, 0, :]  # [CLS] token's hidden featrues(hidden state)

          # hidden state's average in between entities
          e1_h = entity_average(
              sequence_output, inputs['subject_mask']
          )  # token in between subject entities ->
          e2_h = entity_average(
              sequence_output, inputs['object_mask']
          )  # token in between object entities

          # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
          pooled_output = self.model.cls_fc_layer(
              pooled_output
          )  # [CLS] token -> hidden state | green on diagram
          e1_h = self.model.entity_fc_layer(
              e1_h
          )  # subject entity's fully connected layer | yellow on diagram
          e2_h = self.model.entity_fc_layer(
              e2_h
          )  # object entity's fully connected layer | red on diagram

          # Concat -> fc_layer / [CLS], subject_average, object_average
          concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
          logits = self.model.label_classifier(concat_h)

        else:
          pooled = outputs[0]
          logits = self.model.head_class_forward(pooled) 
        return logits
      
      smart_loss_fn = SMARTLoss(eval_fn = eval_fn, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)

      # Save past state if it exists
      # TODO: this needs to be fixed and made cleaner later.
      if self.args.past_index >= 0:
          self._past = outputs[self.args.past_index]

      if labels is not None:
          loss_fct = CrossEntropyLoss()
          loss = loss_fct(outputs['logits'], labels)
          if return_outputs==False:
            loss += 0.5 * smart_loss_fn(embed, outputs['logits'])

      else:
          # We don't use .loss here since the model may return tuples instead of ModelOutput.
          loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

      return (loss, outputs) if return_outputs else loss


class R_drop_Trainer(Trainer):
    
  def get_normalized_probs(self, net_output: Dict[str, Union[torch.Tensor, Any]], log_probs=True) -> torch.Tensor:
      logits = net_output["logits"] if isinstance(net_output, dict) else net_output[0]
      if log_probs:
          return F.log_softmax(logits, dim=-1)
      else:
          return F.softmax(logits, dim=-1)
      
  
  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
      if not self.args.use_rdrop:
          return super().training_step(model, inputs)
          
      model.train()
      inputs = self._prepare_inputs(inputs)
      concat_inputs = {
          'input_ids': torch.cat([inputs['input_ids'], inputs['input_ids'].clone()], 0),
          'attention_mask': torch.cat([inputs['attention_mask'], inputs['attention_mask'].clone()], 0),
          'labels': torch.cat([inputs['labels'], inputs['labels'].clone()], 0),
      }
      
      loss = self.compute_loss(model, concat_inputs)

      if self.args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training

      if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
          # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
          loss = loss / self.args.gradient_accumulation_steps

      if self.deepspeed:
          # loss gets scaled under gradient_accumulation_steps in deepspeed
          loss = self.deepspeed.backward(loss)
      else:
          loss.backward()

      return loss.detach()
  
  
  def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
      """
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
      """
      if not self.args.use_rdrop and self.args.label_smoothing_factor == 0:
          return super().compute_loss(model, inputs)

      elif not self.args.use_rdrop and self.args.label_smoothing_factor != 0:
          assert "labels" in inputs
          labels = inputs["labels"]
          outputs = model(**inputs)
          # Save past state if it exists
          # TODO: this needs to be fixed and made cleaner later.
          if self.args.past_index >= 0:
              self._past = outputs[self.args.past_index]

          if labels is not None:
              loss = self.label_smoother(outputs, labels)
          else:
              # We don't use .loss here since the model may return tuples instead of ModelOutput.
              loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

          return (loss, outputs) if return_outputs else loss

      else:
          # if self.label_smoother is not None and "labels" in inputs:
          if "labels" in inputs:
              labels = inputs['labels'] # inputs.pop("labels")
              pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
          else:
              labels = None
          
          outputs = model(**inputs)
          
          # Save past state if it exists
          # TODO: this needs to be fixed and made cleaner later.
          if self.args.past_index >= 0:
              self._past = outputs[self.args.past_index]

          if labels is not None:
              # loss = self.label_smoother(outputs, labels)
              
              # nll loss original version
              loss = self.label_smoothed_nll_loss(outputs, labels,
                                                  epsilon=0.1 if self.label_smoother else 0) 
              
              kl_loss = self.compute_kl_loss(outputs, pad_mask=pad_mask)
              loss += self.args.reg_alpha * kl_loss

          else:
              # We don't use .loss here since the model may return tuples instead of ModelOutput.
              loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

          return (loss, outputs) if return_outputs else loss

  def compute_kl_loss(self, net_output: Dict[str, Union[torch.Tensor, Any]], pad_mask=None, reduce=True) -> torch.Tensor:
      net_prob = self.get_normalized_probs(net_output, log_probs=True)
      net_prob_tec = self.get_normalized_probs(net_output, log_probs=False)
      if net_prob.size(0) == 3 or net_prob.size(0) == 5:
        return 0
      p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
      p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
      
      p_loss = F.kl_div(p, q_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean') v2 / none(v0)
      q_loss = F.kl_div(q, p_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean')
      
      if pad_mask is not None:
          pad_mask, _ = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
          p_loss.masked_fill_(pad_mask, 0.)
          q_loss.masked_fill_(pad_mask, 0.)

      if reduce:
          p_loss = p_loss.mean()
          q_loss = q_loss.mean()

      loss = (p_loss + q_loss) / 2
      return loss
  
  def label_smoothed_nll_loss(self, model_output: Dict[str, Union[torch.Tensor, Any]], labels: torch.Tensor, epsilon: float) -> torch.Tensor:
      logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
      log_probs = -F.log_softmax(logits, dim=-1)
      if labels.dim() == log_probs.dim() - 1:
          labels = labels.unsqueeze(-1)

      padding_mask = labels.eq(-100)
      # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
      # will ignore them in any case.
      labels = torch.clamp(labels, min=0)
      nll_loss = log_probs.gather(dim=-1, index=labels)
      # works for fp16 input tensor too, by internally upcasting it to fp32
      smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

      nll_loss.masked_fill_(padding_mask, 0.0)
      smoothed_loss.masked_fill_(padding_mask, 0.0)
      
      nll_loss = nll_loss.sum()
      smoothed_loss = smoothed_loss.sum()
      eps_i = epsilon / log_probs.size(-1)
      return (1. - epsilon) * nll_loss + eps_i * smoothed_loss



class MyTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name 
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # custom loss
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        model.cuda()
        # model.to(self.device)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_name == 'CrossEntropy':
            custom_loss = torch.nn.CrossEntropyLoss()
        elif self.loss_name == 'focal':
            custom_loss = FocalLoss()
        elif self.loss_name == 'labelsmoother':
            custom_loss = LabelSmoother(epsilon=self.args.label_smoothing_factor)
            loss = custom_loss(outputs, labels)
            return (loss, outputs) if return_outputs else loss 

        loss = custom_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
