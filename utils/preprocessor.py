class Preprocessor:
  def __init__(self, tokenizer, train_flag):
    self.tokenizer = tokenizer
    self.train_flag=train_flag

  def __call__(self, dataset):
    inputs = []
    labels = []

    for i in range(len(dataset['sentence_id'])):
      inputs.append(dataset['concat_sentence'][i])
      if self.train_flag == True:
        labels.append(dataset['label'][i])
    
    dataset['inputs'] = inputs
    if self.train_flag == True:
      dataset['labels'] = labels
    return dataset


class Rbert_Preprocessor:
  def __init__(self, tokenizer, train_flag):
    self.tokenizer = tokenizer
    self.train_flag=train_flag

  def __call__(self, dataset):
    inputs = []
    target_inputs = []
    labels = []

    for i in range(len(dataset['sentence_id'])):
      inputs.append(dataset['final_sentence'][i])
      target_inputs.append(dataset['target_sentence'][i])
      if self.train_flag == True:
        labels.append(dataset['label'][i])
    
    dataset['inputs'] = inputs
    dataset['target_inputs'] = target_inputs
    if self.train_flag == True:
      dataset['labels'] = labels
    return dataset