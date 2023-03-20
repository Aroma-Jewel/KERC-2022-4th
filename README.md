
# [[KERC2022] 제4회 한국인 감정인식 국제경진대회](https://sites.google.com/view/kerc2022) - 팀: 아로마쥬얼

## 팀 소개 & 성적

### Score  
![public 2nd](https://img.shields.io/badge/PUBLIC-2nd-red?style=plastic) ![private 3rd](https://img.shields.io/badge/PRIVATE-3rd-red?style=plastic)

### Public & Private Leaderboard
▶ Public Score  
<img width="889" alt="image" src="https://user-images.githubusercontent.com/46811558/197766357-d9ff20dc-8312-4894-b24d-99ebde55a844.png"> 

▶ Private Score  
<img width="885" alt="image" src="https://user-images.githubusercontent.com/46811558/197766361-f281646d-9d80-4345-bf59-99b11c12e0bc.png">


### Members
이재학|정유석|
:-:|:-:
<img src='https://user-images.githubusercontent.com/46811558/197347033-8e0b3742-8450-43b6-aa71-12e3f1745031.jpg' height=100 width=100px></img>|<img src='https://user-images.githubusercontent.com/46811558/197347034-af633a11-d50d-42d5-b403-69b3870e0ff7.jpg' height=100 width=100px></img>
|[wogkr810](https://github.com/wogkr810)|[j961224](https://github.com/j961224)|
|jaehahk810@naver.com|j961224@naver.com

---

## 대회 개요


>대화체 텍스트 데이터를 분석하여 한국인의 감정을 예측하는 한국어 자연어처리 인공지능 모델 개발  


## 대회 기간 
> **2022.09.01 ~ 2022.10.23**

## 평가 방법 : Micro-F1 Score
![micro](https://user-images.githubusercontent.com/46811558/197593517-b07ec66d-de4e-402e-8bce-3195155aaf1e.png)

## 대회 데이터셋
> 수상한 삼형제(한국 드라마)의 1513개 scene에서 추출한 총 12289개의 대화 텍스트 데이터  
> 각 데이터에는 *sentence_id*, *person(speaker)*, *sentence*, *scene_ID*, *context(Scene description)*, *label* 존재
- train : 7339개
- public : 2566개
- private : 2384개

> Labels(Train Only)  

![labels](https://user-images.githubusercontent.com/46811558/197594218-5638d268-1b0b-44ca-9caa-74221ca15e64.png)

> 데이터셋 예시
- Train  

![train](https://user-images.githubusercontent.com/46811558/197594590-b16a5e18-27c5-4982-92d6-a564654a066d.png)

- Public & Private  

![test](https://user-images.githubusercontent.com/46811558/197594597-4ac0dc8a-5825-42b1-bdc6-83f941607095.png)

## 사용방법론 및 재현 명령어

**훈련 & 추론 명령어**  

`sh run.sh`

**모델**

[klue/roberta-large](http://huggingface.co/klue/roberta-large)

## 하드웨어

**Colab Pro Plus :** `CPU 12C, Nvidia A100 GPU x 2, 40MEM, 130GB`

## 디렉토리 구조

```
USER/
├── run.sh
├── train.py
├── inference.py
├── pykospacing.ipynb
├── soft_ensemble.ipynb
├── hard_ensemble.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets
│   └── 아로마쥬얼_발표자료.pdf
│
├── Datasets
│   ├── example_submission.csv
│   ├── private_test_data.tsv
│   ├── public_test_data.tsv
│   ├── train_data.tsv
│   ├── private_test_data_pykospacing.tsv
│   ├── public_test_data_pykospacing.tsv
│   ├── train_data_pykospacing.tsv
│   └── train_labels.csv
│
├── utils
│   ├── datasets.py
│   ├── encoder.py
│   ├── heads.py
│   ├── preprocessor.py
│   └── trainer.py
│
├── args
│   ├── DataTraining_args.py
│   ├── Logging_args.py
│   ├── Model_args.py
│   ├── MyTraining_args.py
│   └── _init_.py
│  
├── models
│   ├── Rbert.py
│   └── roberta.py
│
├── final_submission(75)_sh
│   └── run_(55,57,69,70,72,73).sh
│ 
└──────────────────────────
```
- `run.sh`

  - 모델 학습&추론을 위한 shell script 파일입니다.
  - 실행에 필요한 argument는 아래를 참조하시기 바랍니다.  
  

- `train.py`

  - 모델 학습을 실행하는 코드입니다.
  - Eval steps마다 가중치 파일은 './exp' 폴더에 저장되며, 최종 추론에 쓰이는 모델 가중치 파일은 './checkpoints' 폴더에 생성됩니다.

- `inference.py`

  - 학습된 model 가중치를 통해 prediction하고, 예측한 결과와 각 확률을 csv파일로 저장하는 코드입니다.
  - 최종 csv 파일은 './results'폴더에 생성되며, 확률이 결합된 csv 파일은 './results_probs'폴더에 생성됩니다.

- `pykospacing.ipynb`

  - 데이터 전처리에 사용된 [Pykospacing](https://github.com/haven-jeon/PyKoSpacing)을 데이터에 적용하고 생성할 수 있는 노트북 파일입니다.

- `soft_ensemble.ipynb`

  - `inferece.py`를 통해 생성된, 확률이 결합된 최종 csv 파일을 './results_probs' 폴더에서 가져와, 확률을 통한 soft voting 이후 './final_submission/soft_ensemble_final_submission.csv'에 제출 파일을 생성합니다.

- `hard_ensemble.ipynb`

  - `inferece.py`를 통해 생성된, 최종 csv 파일을 './results' 폴더에서 가져와, 최다빈도 hard voting 이후 './final_submission/hard_ensemble_final_submission.csv'에 제출 파일을 생성합니다.

- `Datasets/`

  - 원시데이터 및 Pykospacing을 적용한 데이터들이 있는 디렉토리입니다.

- `models/`

  - 모델 class를 구현한 파일들이 있는 디렉토리입니다.  
  - `roberta.py`
  	- Roberta model을 기반으로 classification을 할 수 있도록 한 class가 구현된 파일입니다.
  - `Rbert.py`
  	- 기존 R-bert를 변형하여 previous sentence와 target sentence를 entitiy라고 생각하고 구현한 모델에 대한 class를 정의한 파일입니다.

- `utils/`

    - 데이터셋 전처리, 모델 입력 데이터 전처리, 모델에 사용된 loss 및 head 함수들이 있는 디렉토리입니다. 
	- `datasets.py`  
	    - 원시 데이터들을 type에 따라 나누고 전처리 함수를 적용 한 후 Dataset dict를 load하는 Dataset class를 정의한 파일입니다.
	- `preprocessor.py` 
      - `datasets.py`를 적용한 뒤, load한 데이터들을 encoder에 넣기 위한 형태로 변환시키는 Preprocessor class를 정의한 파일입니다.
	- `encoder.py`
      - 데이터를 tokenize하며 데이터의 type에 따라 `label`을 붙이고, 모델의 input형태로 변환하는 Encoder class를 정의한 파일입니다.
	- `heads.py`
      - 모델에 사용된 layer 및 head들을 customize 할 수 있는 class들을 정의한 파일입니다.
      - CLS token, SEP token, hidden states 등을 custom하게 사용합니다.
	- `trainer.py`
      - 모델에 사용된 loss 및 Huggingface기반의 Custom Trainer class를 정의한 파일입니다.
      - R-drop, R3F와 Smart loss를 이용할 수 있는 class입니다.

- `args/`
    - 학습 및 추론에 필요한 Arguments들을 정의한 파일들이 있는 디렉토리입니다.
	- `DataTraining_args.py`  
      - 데이터에 관련된 argument들을 정의한 파일입니다.
	- `Logging_args.py`   
      - WandB Logging에 관련된 argument들을 정의한 파일입니다.
	- `Model_args.py`   
      - 모델과 관련된 argument들을 정의한 파일입니다.
	- `MyTrainings_args.py`   
      - 모델 학습에 관련된 argument들을 정의한 파일입니다.

- `final_submission(75)_sh/`
  - 최종 제출에 관련된 shell script파일들이 있는 디렉토리입니다.
  - 각 파일들로 학습 및 추론을 진행하여, 생성된 'results' & 'results_probs' 폴더 내의 결과물로 'hard & soft ensemble.ipynb'를 실행하면 최종 제출과 관련된 파일을 생성할 수 있습니다.
  - [최종 제출 관련 실험 Wandb 링크](http://wandb.ai/aroma-jewel/KERC_master?workspace=)


## Arguments

### DataTraining_args.py Argument 설명

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          max_length         | 모델 input의 최대 길이를 설정합니다.                        |
|          data_dir           | dataset의 경로를 설정합니다.                         |
|          data_type          | dataset의 type을 설정합니다.                               |
|          use_rtt_data       | Round-trip translation의 수행여부를 결정합니다.            |
|          past_sentence      | 각 Scene에서 사용할 target sentence 이전의 sentence 개수를 설정합니다.|
|          preprocess_version | 모델에 사용할 데이터의 version을 결정합니다.                |
|          use_substitute_preprocess | 정의한 전처리 함수의 수행여부를 결정합니다.          |
|          use_pykospacing | Pykospacing을 적용한 데이터셋의 사용여부를 결정합니다.       |

### Logging_args.py Argument 설명

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          dotenv_path        | wandb token이 존재하는 wandb.env의 경로를 설정합니다.       |
|          project_name       | Wandb에 Logging할 project 이름을 설정합니다.               |
|          group_name         | Wandb에 Logging할 group 이름을 설정합니다.                 |

### Model_args.py Argument 설명

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          PLM                | 모델의 PLM을 결정합니다.                                   |
|          save_path          | 최종 checkpoint가 저장될 경로를 설정합니다.                |
|          head_class         | `./utils/head.py`에서 사용할 custom layer & head를 설정합니다. |


### MyTraining_args.py Argument 설명

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          output_dir         | Eval시에 생성된 weight파일들이 저장되는 경로를 설정합니다.   |
|          use_rdrop          | R-drop의 수행여부를 결정합니다.                             |
|          reg_alpha          | R-drop에 사용되는 Regularized dropout의 비율을 설정합니다.  |
|          use_Smart_loss     | Smart loss의 수행여부를 결정합니다.                         |
|          multiple_weight    | soft voting을 실행할 weight들의 저장경로를 설정합니다.       | 
|          use_special_tokens | special token의 사용여부를 결정합니다.                       | 
|          use_RBERT          | RBERT의 수행여부를 결정합니다.                               | 
|          use_kfold          | 모델 학습시에 K-fold의 수행여부를 결정합니다.                 | 
|          loss_name          | 모델 학습시에 사용할 loss의 종류를 결정합니다.                | 

### run.sh Argument 설명

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          PLM                | 모델의 PLM을 결정합니다.                                   |
|          overwrite_output_dir | output_dir 폴더의 weight들을 덮어씌울지를 결정합니다.     |
|      save_total_limit       | 최대 checkpoint 저장 갯수를 지정합니다.                   |
|          output_dir         | Eval시에 생성된 weight파일들이 저장되는 경로를 설정합니다.   |
|          do_train           | 모델을 훈련할지를 결정합니다.                              |
|          do_eval           | 모델을 평가할지를 결정합니다.                               |
|        learning_rate        | 훈련 learning rate를 지정합니다.                          |
|        num_train_epochs     | 훈련 epoch 수를 지정합니다.                               |
| per_device_train_batch_size | train batch size를 지정합니다.                            |
| gradient_accumulation_steps | gradient accumulation 수를 정합니다.                      |
|          logging_dir         | logging파일들이 저장되는 경로를 설정합니다.               |
|        save_strategy        | step or epoch 기준 등으로 최종 checkpoint가 저장되는 방식을 정합니다. |
|        evaluation_strategy  | step or epoch 기준 등으로 eval step의 weight가 저장되는 방식을 정합니다. |
|        logging_steps        | logging이 진행되는 step의 간격을 지정합니다.               |
|        save_steps           | weight가 저장되는 step의 간격을 지정합니다.               |
|        eval_steps           | 훈련시에 평가를 수행할 step의 간격을 지정합니다.            |
|        weight_decay         | 옵티마이저에 적용할 weight_decay hyper parameter를 지정합니다. |
|        warmup_ratio         | 훈련초기에, 지정한 learning_rate까지 도달할 step의 비율을 지정합니다. |
|        load_best_model_at_end | 가장 성능이 좋은 weight checkpoint를 훈련 마지막 step에서 load할지 여부를 결정합니다.  |
|        metric_for_best_model | 성능의 기준이 되는 metric을 설정합니다.                    |
|        multiple_weights | soft voting을 실행할 weight들의 저장경로를 설정합니다.                   |

## Reference

- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/pdf/1905.08284.pdf)
	- [Reference Github link 1](https://github.com/monologg/R-BERT)
	- [Reference Github link 2](https://github.com/snoop2head/KLUE-RBERT)

- [R-Drop: Regularized Dropout for Neural Networks (NeurlPS 2021)](https://arxiv.org/pdf/2106.14448v2.pdf)
	- [Reference Github link](https://github.com/dropreg/R-Drop)

- [BETTER FINE-TUNING BY REDUCING REPRESENTATIONAL COLLAPSE (ICLR 2021)](https://arxiv.org/pdf/2008.03156v1.pdf)

- [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization (ACL 2020)](https://aclanthology.org/2020.acl-main.197.pdf)


