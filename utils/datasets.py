import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict

import math

from transformers import HfArgumentParser
from args import (MyTrainingArguments, ModelArguments, LoggingArguments, DataTrainingArguments)

parser = HfArgumentParser(
  (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
)
model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()

class KERCDataset(Dataset):
  def __init__(self, tokenizer, data_dir, data_type, use_pykospacing = False):

    self.tokenizer = tokenizer
    self.data_type = data_type
    self.use_pykospacing = use_pykospacing
    assert data_type in ['train', 'public_test', 'private_test'], f'Unknown data_type {data_type}'
    
    if data_type in 'public_test':
      if self.use_pykospacing:
        data_df_test0 = pd.read_csv(f'{data_dir}/public_test_data_pykospacing.tsv', delimiter='\t')
        data_df_test1 = pd.read_csv(f'{data_dir}/private_test_data_pykospacing.tsv', delimiter='\t')
      else:
        data_df_test0 = pd.read_csv(f'{data_dir}/public_test_data.tsv', delimiter='\t')
        data_df_test1 = pd.read_csv(f'{data_dir}/private_test_data.tsv', delimiter='\t')
      self.data_df = data_df_test0.append(data_df_test1, ignore_index = True)
      self.data_df = self.data_df.sort_values("sentence_id")
      self.data_df = self.data_df.reset_index(drop=True)
    else:
      if self.use_pykospacing:
        self.data_df = pd.read_csv(f'{data_dir}/{data_type}_data_pykospacing.tsv', delimiter='\t')  
      else:
        self.data_df = pd.read_csv(f'{data_dir}/{data_type}_data.tsv', delimiter='\t')
    self.labels = ['dysphoria', 'neutral', 'euphoria']

    if data_type not in ['public_test', 'private_test']:
      self.label_df = pd.read_csv(f'{data_dir}/{data_type}_labels.csv')
      self.label_dict = {'dysphoria':0, 'neutral':1, 'euphoria':2}

    self.len = self.data_df.shape[0]

  def __get_scene_context(self, data_df, scene_groups, sample_idx):
    sample_row =  data_df.loc[sample_idx]
    final_sentences = self.concat_sentence(data_df, scene_groups, sample_row['scene'], sample_idx)
    return final_sentences
  
  def preprocess(self, text):
    # v0
    before_text_v0 = ['아부진', '아부지', '지껄여', '멋지드라', '식 끝났어요?', '느이', '느그', \
    '지구댑니다', '그렸시유', '혼수아빤', '스돕']

    after_text_v0 = ['아버지', '아버지', '말해', '멋있더라', '식사 끝났어요?', '너희', '너희', \
    '지구대 입니다', '그랬어요', '혼수 아빠는', '스톱']

    assert len(before_text_v0)==len(after_text_v0), f'Not matching length v0'

    for idx, before_one_text in enumerate(before_text_v0):
      text = text.replace(before_one_text, after_text_v0[idx])
    
    # add v1
    before_text_v1 = ['여그', '되얏어', '만만찮여', '이거유', '하디', '워칙혀', '쪼께', '숙젤', \
    '엇다대고', '뉘집', '뭐드러', '비위꺼정', '본겨', '숭볼라면', '델꾸', '델꼬',\
    '그러유', '연앨', '어때서유', '끝내지간디유', '쪼께', '형님 여?어요', '하는기유',\
    '안녕하십니,','이여대','좋으시디']
    
    after_text_v1 = ['여기', '되었어', '만만치 않어', '이거에요', '해', '어떻게 해', '쫌만', '숙제를', \
    '어디다 대고', '누구 집', '뭐하러', '비위까지', '본거야', '흉 볼려면', '데리고 와서', '데리고',\
    '그래요', '연애를', '어때서요', '끝내야 간대요', '조금', '형님 여기요', '하는거에요',\
    '안녕하십니까','이화여자대학교','좋대']

    assert len(before_text_v1)==len(after_text_v1), f'Not matching length v1'

    for idx, before_one_text in enumerate(before_text_v1):
      text = text.replace(before_one_text, after_text_v1[idx])

    return text
    

  def concat_sentence(self, data_df, scene_groups, sample_row_scene, sample_idx):
    start_idx = scene_groups[sample_row_scene].min() #first sample in a scene
    scene_sentences  = []
    sep_token = self.tokenizer.sep_token
    special_token = self.tokenizer.additional_special_tokens
    for idxx in range(start_idx, sample_idx+1):
      if idxx==sample_idx:
        if data_args.preprocess_version in ["v1","v5","v7"]:
          if data_args.use_substitute_preprocess:
            scene_sentences.append(sep_token+" "+'"'+data_df.loc[idxx]['person']+'"' + " " + self.preprocess(data_df.loc[idxx]['sentence']))
          else:
            scene_sentences.append(sep_token+" "+'"'+data_df.loc[idxx]['person']+'"' + " " + data_df.loc[idxx]['sentence'])
        elif data_args.preprocess_version in ["v2","v3","v4"]:
          if data_args.use_substitute_preprocess:
            scene_sentences.append(sep_token+" "+'"'+data_df.loc[idxx]['person']+'"' + " " + self.preprocess(data_df.loc[idxx]['sentence'])+sep_token)
          else:
            scene_sentences.append(sep_token+" "+'"'+data_df.loc[idxx]['person']+'"' + " " + data_df.loc[idxx]['sentence']+sep_token)
        elif data_args.preprocess_version == "v6":
          if data_args.use_substitute_preprocess:
            scene_sentences.append(special_token[1] + sep_token+" "+'"'+data_df.loc[idxx]['person']+'"' + " " + self.preprocess(data_df.loc[idxx]['sentence']))
          else:
            scene_sentences.append(special_token[1] + sep_token+" "+'"'+data_df.loc[idxx]['person']+'"' + " " + data_df.loc[idxx]['sentence'])
      else:
        if data_args.use_substitute_preprocess:
          scene_sentences.append('"'+data_df.loc[idxx]['person']+'"' + " " + self.preprocess(data_df.loc[idxx]['sentence']) + " ")
        else:
          scene_sentences.append('"'+data_df.loc[idxx]['person']+'"' + " " + data_df.loc[idxx]['sentence'] + " ")

    tmp_context = data_df.loc[sample_idx]['context']
    final_sentences = ''

    try:
      if math.isnan(tmp_context):
        if data_args.preprocess_version in ["v6","v7"]:
          final_sentences += special_token[1]

    except:
      if data_args.preprocess_version == "v3":
        final_sentences += sep_token + tmp_context + sep_token
      elif data_args.preprocess_version =='v4':
        if data_args.use_substitute_preprocess:
          final_sentences += sep_token + tmp_context + sep_token + " " + '"' + data_df.loc[sample_idx]['person'] + '"' + " " + self.preprocess(data_df.loc[sample_idx]['sentence']) + sep_token
        else:
          final_sentences += sep_token + tmp_context + sep_token + " " + '"' + data_df.loc[sample_idx]['person'] + '"' + " " + data_df.loc[sample_idx]['sentence'] + sep_token

        return final_sentences
      elif data_args.preprocess_version == "v5":
          final_sentences += '[context]' + tmp_context + '[context]' + ' ' 
      elif data_args.preprocess_version == "v6":
        final_sentences += special_token[0] + tmp_context + special_token[0] + special_token[1]
      elif data_args.preprocess_version == "v7":
        final_sentences += special_token[0] + tmp_context + special_token[0]
    
  
    if len(scene_sentences)>data_args.past_sentence:
      for idx, s_sentence in enumerate(scene_sentences):
        if len(scene_sentences)-idx<=data_args.past_sentence:
          final_sentences+=s_sentence
    else:
      for s_sentence in scene_sentences:
        final_sentences+=s_sentence
    
    return final_sentences

  def __rbert_get_scene_context(self, data_df, scene_groups, sample_idx):
    sample_row =  data_df.loc[sample_idx]
    start_idx = scene_groups[sample_row['scene']].min() #first sample in a scene
    scene_sentences  = []
    target_sentences = ''
    sep_token = self.tokenizer.sep_token
    for idxx in range(start_idx, sample_idx+1):
      if idxx==sample_idx:
        target_sentences = '"'+data_df.loc[idxx]['person']+'"' + " " + data_df.loc[idxx]['sentence']
      else:
        scene_sentences.append('"'+data_df.loc[idxx]['person']+'"' + " " + data_df.loc[idxx]['sentence'] + " ")

    final_sentences = ""
    tmp_context = data_df.loc[sample_idx]['context']
    try:
      if math.isnan(tmp_context):
        final_sentences+=""
    except:
      final_sentences += ' [context] ' + tmp_context + ' [context] '
    
    if len(scene_sentences)>4:
      for idx, s_sentence in enumerate(scene_sentences):
        if len(scene_sentences)-idx<=4:
          final_sentences+=s_sentence
    else:
      for s_sentence in scene_sentences:
        final_sentences+=s_sentence
    
    if len(final_sentences)==0:
      final_sentences = target_sentences
    return final_sentences, target_sentences

  # 일단은 sentence id, scene, sentence만 사용하기로 생각
  def load_datasets(self):
    sentence_id = self.data_df['sentence_id']
    sentence = self.data_df['sentence']
    scene = self.data_df['scene']
    person = self.data_df['person']

    scene_sents = []
    for idx in tqdm(range(len(self.data_df))):
      scene_groups = self.data_df.groupby(by='scene').indices
      scene_ = self.__get_scene_context(self.data_df, scene_groups, idx)
      scene_sents.append(scene_)

    if self.data_type not in ['public_test', 'private_test']:
      label = self.label_df['label']
      label_tag = [self.label_dict[i] for i in label]

      dataset = Dataset.from_dict({
        "sentence_id": sentence_id,
        "sentence": sentence,
        "concat_sentence":scene_sents,
        "scene":scene,
        "person":person,
        "label":label_tag
      })

      dataset_dict = DatasetDict({"train": dataset})

    else:
      dataset = Dataset.from_dict({
        "sentence_id": sentence_id,
        "sentence": sentence,
        "concat_sentence":scene_sents,
        "scene":scene,
        "person":person
      })

      dataset_dict = DatasetDict({"test": dataset})
    
    return dataset_dict
  
  def rbert_load_datasets(self):
    sentence_id = self.data_df['sentence_id']
    sentence = self.data_df['sentence']
    scene = self.data_df['scene']
    person = self.data_df['person']

    final_scene_sents = []
    target_scene_sents = []
    for idx in tqdm(range(len(self.data_df))):
      scene_groups = self.data_df.groupby(by='scene').indices
      final_sentences, target_sentences = self.__rbert_get_scene_context(self.data_df, scene_groups, idx)
      final_scene_sents.append(final_sentences)
      target_scene_sents.append(target_sentences)

    if self.data_type not in ['public_test', 'private_test']:
      label = self.label_df['label']
      label_tag = [self.label_dict[i] for i in label]

      dataset = Dataset.from_dict({
        "sentence_id": sentence_id,
        "sentence": sentence,
        "final_sentence":final_scene_sents,
        "target_sentence":target_scene_sents,
        "scene":scene,
        "person":person,
        "label":label_tag
      })

      dataset_dict = DatasetDict({"train": dataset})

    else:
      dataset = Dataset.from_dict({
        "sentence_id": sentence_id,
        "sentence": sentence,
        "final_sentence":final_scene_sents,
        "target_sentence":target_scene_sents,
        "scene":scene,
        "person":person
      })

      dataset_dict = DatasetDict({"test": dataset})
    
    return dataset_dict