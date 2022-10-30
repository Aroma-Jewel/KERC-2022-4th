import os
import math
import wandb
import torch
import random
import numpy as np
from dotenv import load_dotenv
from transformers import (
  Trainer, 
  HfArgumentParser, 
  AutoTokenizer, 
  AutoConfig, 
  DataCollatorWithPadding, 
  AutoModelForSequenceClassification,
  T5Tokenizer,  
)
from functools import partial
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold
from args import (MyTrainingArguments, ModelArguments, LoggingArguments, DataTrainingArguments)

from utils.datasets import KERCDataset
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder
from utils.trainer import R_drop_Trainer, Smart_Trainer, MyTrainer


from models.roberta import RobertaForSequenceClassification

from models.Rbert import RBERT
from utils.preprocessor import Rbert_Preprocessor
from utils.encoder import RBERT_Encoder



def seed_everything(seed):
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  np.random.default_rng(seed)
  random.seed(seed)

def compute_metrics(EvalPrediction):
  preds, labels = EvalPrediction
  preds = np.argmax(preds, axis=1)

  f1_metric = load_metric('f1')    
  f1 = f1_metric.compute(predictions = preds, references = labels, average="micro")

  acc_metric = load_metric('accuracy')
  acc = acc_metric.compute(predictions = preds, references = labels)
  acc.update(f1)
  return f1

def main():
  print(f"# of CPU : {os.cpu_count()}")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
  )
  model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
  seed_everything(training_args.seed)

  tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
  if training_args.use_special_tokens:
    if data_args.preprocess_version in ['v6', 'v7']:
      special_tokens_dict = {'additional_special_tokens': ['[context]','[past]']}
      num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  
  loader = KERCDataset(tokenizer, data_args.data_dir, data_args.data_type, data_args.use_pykospacing)

  if training_args.use_RBERT:
    dset = loader.rbert_load_datasets()
  else:
    dset = loader.load_datasets()
  dset = dset['train'].shuffle(training_args.seed)
  print(dset)

  if training_args.use_RBERT:
    preprocessor = Rbert_Preprocessor(tokenizer, train_flag=True)
  else:
    preprocessor = Preprocessor(tokenizer, train_flag=True)
  dset = dset.map(preprocessor, batched=True, num_proc=4,remove_columns=dset.column_names)
  print(dset)

  config = AutoConfig.from_pretrained(model_args.PLM)
  config.num_labels = 3
  config.head_class = model_args.head_class

  if training_args.use_RBERT:
    encoder = RBERT_Encoder(tokenizer, data_args.max_length)
  else:
    encoder = Encoder(tokenizer, data_args.max_length)
  dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
  print(dset)
  
  data_collator =DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
  skf = StratifiedKFold(n_splits=5, shuffle=True)

  for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):

    train_dataset = dset.select(train_idx.tolist())
    valid_dataset = dset.select(valid_idx.tolist())

    load_dotenv(dotenv_path=logging_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("DATASETS_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    if training_args.max_steps == -1:
      name = f"EP_Fold{i}:{training_args.num_train_epochs}_"
    else:
      name = f"MS_Fold{i}:{training_args.max_steps}_"
    name += f"LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_{model_args.head_class}"
    
    training_args.RBERT = False
    if training_args.use_RBERT:
      training_args.RBERT = True
      model = RBERT.from_pretrained(model_args.PLM, config=config)

    elif config.head_class != "Org" or training_args.use_Smart_loss:
      model = RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)
    else:
      model = AutoModelForSequenceClassification.from_pretrained(model_args.PLM, config=config)

    
    if training_args.use_special_tokens:
      model.resize_token_embeddings(len(tokenizer))
      tokenizer.save_pretrained("./checkpoints/"+name)
    

    wandb.init(
      entity="aroma-jewel",
      project="KERC",
      name=name
    )
    wandb.config.update(training_args)
    
    if training_args.use_Smart_loss:
      trainer = Smart_Trainer(      
        model=model,
        args=training_args,
        train_dataset=train_dataset,            
        eval_dataset=valid_dataset,             
        data_collator=data_collator,            
        tokenizer=tokenizer,                    
        compute_metrics=compute_metrics,
      )

    
    elif training_args.use_rdrop:
      trainer = R_drop_Trainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,            
        eval_dataset=valid_dataset,             
        data_collator=data_collator,            
        tokenizer=tokenizer,                    
        compute_metrics=compute_metrics,
      )
    
    else:
      trainer = MyTrainer(     
        model=model,
        args=training_args,
        train_dataset=train_dataset,            
        eval_dataset=valid_dataset,             
        data_collator=data_collator,            
        tokenizer=tokenizer,                    
        compute_metrics=compute_metrics,
        loss_name = training_args.loss_name
      )

    trainer.train()
    trainer.evaluate()
    prev_path = model_args.save_path
    model_args.save_path = os.path.join(model_args.save_path, name)
    trainer.save_model(model_args.save_path)
    model_args.save_path = prev_path
    wandb.finish()
    if training_args.use_kfold==False:
      break


if __name__ == "__main__":
  main()
