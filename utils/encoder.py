import re
import torch
import numpy as np

class Encoder :
    def __init__(self, tokenizer, max_input_length: int) :
      self.tokenizer = tokenizer
      self.max_input_length = max_input_length
    
    def __call__(self, examples):
      model_inputs = self.tokenizer(examples['inputs'],
          max_length=self.max_input_length, 
          return_token_type_ids=False,
          truncation=True,
      )

      if 'labels' in examples :
          model_inputs['labels'] = examples['labels']
      return model_inputs


class RBERT_Encoder:
    def __init__(self, tokenizer, max_input_length: int) :
      self.tokenizer = tokenizer
      self.max_input_length = max_input_length
    
    def __call__(self, examples):
      # padding = "max_length"

      model_inputs = self.tokenizer(examples['inputs'],
          examples['target_inputs'],
          max_length=self.max_input_length,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
      )

      subject_entity_mask = []
      object_entity_mask = []
      object_start_index = []
      for input_id, inputs, target_inputs in zip(model_inputs['input_ids'], examples['inputs'], examples['target_inputs']):
        subject_entity_mask_one, object_entity_mask_one, object_start_index_one = self.add_entity_mask(
          input_id, inputs, target_inputs
        )
        subject_entity_mask.append(subject_entity_mask_one)
        object_entity_mask.append(object_entity_mask_one)
        object_start_index.append(object_start_index_one)

      model_inputs["subject_mask"] = subject_entity_mask
      model_inputs["object_mask"] = object_entity_mask
      model_inputs['subject_special_token_index'] = [1] * len(subject_entity_mask)
      model_inputs['object_special_token_index'] = object_start_index

      if 'labels' in examples :
          model_inputs['labels'] = examples['labels']
      return model_inputs
    
    def add_entity_mask(self, input_id, subject_entity, object_entity):

        # initialize entity masks
        subject_entity_mask = np.zeros(self.max_input_length, dtype=int)
        object_entity_mask = np.zeros(self.max_input_length, dtype=int)

        subject_len = len(self.tokenizer.encode(subject_entity, add_special_tokens=False))
        object_len = len(self.tokenizer.encode(object_entity, add_special_tokens=False))

        subject_entity_mask[
          1 : 1 + subject_len
        ] = 1
        
        object_entity_mask[
          2 + subject_len : 2 + subject_len + object_len
        ] = 1

        return subject_entity_mask, object_entity_mask, 2 + subject_len