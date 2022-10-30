import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from utils.datasets import KERCDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer
)
from args import (MyTrainingArguments, ModelArguments, LoggingArguments, DataTrainingArguments)
from models.roberta import RobertaForSequenceClassification

from models.Rbert import RBERT
from utils.preprocessor import Rbert_Preprocessor
from utils.encoder import RBERT_Encoder

from datetime import datetime, timedelta


def inference_ensemble(model_dir, training_args, data_args, device):
  dirs = os.listdir(model_dir)
  dirs = sorted(dirs)

  final_output_pred = []
  for i in range(len(dirs)):
    model_d = os.path.abspath(os.path.join(model_dir, dirs[i]))

    config = AutoConfig.from_pretrained(model_d)

    # head_class가 없다면
    if "head_class" not in str(config):
      config.head_class = "Org"

    if training_args.use_RBERT:
      model = RBERT.from_pretrained(model_d, config=config)

    elif config.head_class == "Org" and training_args.use_Smart_loss==False:
      model = AutoModelForSequenceClassification.from_pretrained(model_d, config=config)
    else:
      model = RobertaForSequenceClassification.from_pretrained(model_d, config=config)

    
    tokenizer = AutoTokenizer.from_pretrained(model_d)

    data_args.data_type = "public_test"
    loader = KERCDataset(tokenizer, data_args.data_dir, data_args.data_type, data_args.use_pykospacing)
    if training_args.use_RBERT:
      dset = loader.rbert_load_datasets()
    else:
      dset = loader.load_datasets()
    print(dset)

    final_prediction_id = dset['test']['sentence_id']
    dset = dset['test']

    if training_args.use_RBERT:
      preprocessor = Rbert_Preprocessor(tokenizer, train_flag=False)
    else:
      preprocessor = Preprocessor(tokenizer, train_flag=False)
    dset = dset.map(preprocessor, batched=True, num_proc=4, remove_columns=dset.column_names)

    if training_args.use_RBERT:
      encoder = RBERT_Encoder(tokenizer, data_args.max_length)
    else:
      encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    trainer = Trainer(                       
      model=model,                  
      args=training_args,
      data_collator=data_collator,
    )

    outputs = trainer.predict(dset)
    prob = F.softmax(torch.Tensor(outputs[0]), dim=-1)

    if i==0:
      final_output_pred.append(prob)
    else:
      final_output_pred[0]+=prob
  
  return final_output_pred, len(dirs), final_prediction_id


def main():
  parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
  )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

  if training_args.multiple_weight != None:
    pred_answer, checkpoint_length, final_prediction_id = inference_ensemble(
        training_args.multiple_weight, training_args, data_args, device
    )  # model에서 class 추론

    prediction = np.argmax(pred_answer[0], axis=1)
    prob = [[k/checkpoint_length for k in l] for l in pred_answer]
    prob = [aa.tolist() for aa in prob[0]]
    
  else:
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)

    data_args.data_type = "public_test"
    config = AutoConfig.from_pretrained(model_args.PLM)
    loader = KERCDataset(tokenizer, data_args.data_dir, data_args.data_type, data_args.use_pykospacing)

    if training_args.use_RBERT:
      dset = loader.rbert_load_datasets()
    else:
      dset = loader.load_datasets()
    print(dset)

    final_prediction_id = dset['test']['sentence_id']
    dset = dset['test']

    if training_args.use_RBERT:
      preprocessor = Rbert_Preprocessor(tokenizer, train_flag=False)
    else:
      preprocessor = Preprocessor(tokenizer, train_flag=False)
    dset = dset.map(preprocessor, batched=True, num_proc=4, remove_columns=dset.column_names)

    if training_args.use_RBERT:
      encoder = RBERT_Encoder(tokenizer, data_args.max_length)
    else:
      encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    if "head_class" not in str(config):
      config.head_class = "Org"

    if training_args.use_RBERT:
      model = RBERT.from_pretrained(model_args.PLM, config=config)

    elif config.head_class == "Org" and training_args.use_Smart_loss==False:
      model = AutoModelForSequenceClassification.from_pretrained(model_args.PLM, config=config)
    else:
      model = RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)

    trainer = Trainer(                       
      model=model,                         
      args=training_args,                  
      data_collator=data_collator,
    )

    outputs = trainer.predict(dset)
    prob = F.softmax(torch.Tensor(outputs[0]), dim=-1).tolist()
    prediction = outputs[0].argmax(axis=1)

  index_to_label = {0:'dysphoria', 1:'neutral', 2:'euphoria'}
  final_prediction = []
  for p in prediction:
    final_prediction.append(index_to_label[int(p)])
  
  final_submission = pd.DataFrame(columns = ['Id','Predicted'])
  final_submission_prob = pd.DataFrame(columns = ['Id','Predicted', 'probs'])
  final_submission['Id'] = final_prediction_id
  final_submission['Predicted'] = final_prediction

  final_submission_prob['Id'] = final_prediction_id
  final_submission_prob['Predicted'] = final_prediction

  final_prob = []
  for one_prob in prob:
    d = list(map(str,one_prob))
    strr = ' '.join(d)
    final_prob.append(strr)
  final_submission_prob['probs'] = final_prob

  now_date = datetime.now()
  diff_hours = timedelta(hours=9)
  now_date += diff_hours
  print_now = str(now_date.month) + '_' + str(now_date.day) + '_' + str(now_date.hour) + '_' + str(now_date.minute)


  if not os.path.exists('./results'):
    os.makedirs('./results')
  
  if not os.path.exists('./results_probs'):
    os.makedirs('./results_probs')

  submission_save_path = os.path.join('./results',f'submission_{print_now}.csv')
  probs_submission_save_path = os.path.join('./results_probs',f'submission_{print_now}.csv')
  final_submission.to_csv(submission_save_path, index=False)
  final_submission_prob.to_csv(probs_submission_save_path, index=False)


if __name__ == "__main__" :
    main()