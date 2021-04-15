# Few-shot Semantic Parsing for New Predicates

This is the code for the EACL2021 paper, [[Few-shot Semantic Parsing for New Predicates]](https://arxiv.org/abs/2101.10708).

## Setup:

### Install dependency
* Cuda 10.2
* ```conda env create -f environment.yml```
* ```conda activate few-shot-semantic-parsing```

### Download Glove
```./pull_dependency.sh```

### Download the pretrained models
Download the pre-trained models from Google Drive, and copy them to the corresponding folders.
* Copy [[ATIS pre-trained model for one-shot learning]](https://drive.google.com/file/d/1ffuyVIx1-M71-CqXc8W-5d0JJLq74Sg4/view?usp=sharing) to ```saved_models/atis/freq_0/``` and unzip it.
*
*
*
*
*

## Train the semantic parsers
### Pre-process the data
This is to generate the sequences of actions that could construct the logical forms.
* *Atis* ```python preprocess_data/atis/generate_examples.py```
* *Geo* ```python preprocess_data/geo/generate_examples.py```
* *Jobs* ```python preprocess_data/jobs/generate_examples.py```

### Pre-training
You could either download the pre-trained models from the corresponding links or pre-train the parser yourself.
* *ATIS one-shot:* ```./scripts/atis/one_shot/train.sh 0 1 pretrain```
* *ATIS two-shot:* ```./scripts/atis/two_shot/train.sh 0 1 pretrain```
* *Geo one-shot:* ```./scripts/geo/one_shot/train.sh 0 1 pretrain```
* *Geo two-shot:* ```./scripts/geo/two_shot/train.sh 0 1 pretrain```
* *Jobs one-shot:* ```./scripts/jobs/one_shot/train.sh 0 1 pretrain```
* *Jobs two-shot:* ```./scripts/jobs/two_shot/train.sh 0 1 pretrain```
### Fine-tuning and Testing
* *ATIS one-shot:* ```./scripts/atis/one_shot/fine_tune.sh saved_models/atis/freq_0/pretrained_model_name.bin [0..4] [1..2]``` 
* *ATIS two-shot:* ```./scripts/atis/two_shot/fine_tune.sh saved_models/atis/freq_50/pretrained_model_name.bin [0..4] [1..2]```
* *Geo one-shot:* ```./scripts/geo/one_shot/fine_tune.sh saved_models/geo/freq_0/pretrained_model_name.bin [0..4] [1..2]```
* *Geo two-shot:* ```./scripts/geo/two_shot/fine_tune.sh saved_models/geo/freq_50/pretrained_model_name.bin [0..4] [1..2]```
* *Jobs one-shot:* ```./scripts/jobs/one_shot/fine_tune.sh saved_models/jobs/freq_0/pretrained_model_name.bin [0..4] [1..2]```
* *Jobs two-shot:* ```./scripts/jobs/two_shot/fine_tune.sh saved_models/jobs/freq_0/pretrained_model_name.bin [0..4] [1..2]```

If you find this code useful, please cite:
```angular2html
@article{li2021few,
  title={Few-Shot Semantic Parsing for New Predicates},
  author={Li, Zhuang and Qu, Lizhen and Huang, Shuo and Haffari, Gholamreza},
  journal={arXiv preprint arXiv:2101.10708},
  year={2021}
}
```
