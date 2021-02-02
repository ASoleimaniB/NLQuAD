# NLQuAD: A Non-Factoid Long Question Answering Data Set
_Paper will be published at EACL2021_ \
NLQuAD is a non-factoid long question answering dataset from BBC news articles. NLQuAD’s question types and the long length of its context documents as well as answers, make it a challenging real-world task.
NLQuAD consists of news articles as context documents, interrogative sub-headings in the articles as questions, and body paragraphs corresponding to the sub-headings as contiguous answers to the
questions. NLQuAD contains 31k non-factoid questions and long answers collected from 13k BBC news articles. See example articles in BBC [1](https://www.bbc.com/news/world-asia-china-51230011), [2](https://www.bbc.com/news/world-55709428). 
We automatically extract target answers because annotating for non-factoid long QA is extremely challenging and costly. 
 
## Dataset
Download a copy of the dataset, distributed under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/3.0/) licence providing free access for non-commercial and academic usage. The format of the dataset is like SQuAD v1.1.\
[Training Set](http://bit.ly/nlquad_train) \
[Validation Set](http://bit.ly/nlquad_valid) \
[Evaluation Set](http://bit.ly/nlquad_eval)

## Leaderboard
NLQuAD revaluation set results (as of 2nd Feb 2021).

| Method | EM | Precision | Recall | F1 | IoU
| :--- | :---: | :---: | :---: | :---: | :---: |
| BERT base | 25.03 | 60.60 | 82.48 | 63.96 | 53.75 |
| BERT large | 30.29 | 64.87 | 84.62 | 67.91 | 58.39 |
| RoBERTa base | 29.07 | 64.02 | 84.79 | 67.19 | 57.65 |
| RoBERTa large | 33.40 | 67.79 | **87.56** | 71.10 | 62.39 |
| Longformer | **50.30** | **83.92** | 85.17 | **81.38** | **73.57** |
_Please send the link of your paper to a.soleimani.b@gmail.com to include your results._

## BERT 
### Requirements
First, create a conda environment

        conda create -n nlquad python=3.6.10
1- Install `Pytorch 1.1.0` according to your OS and CUDA.

        <OSX>
        # conda
        conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
        
        <Linux and Windows>
        # CUDA 9.0
        conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
        # CUDA 10.0
        conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
        # CPU Only
        conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
2- Install `Transformers 2.5.1`

        pip install transformers==2.5.1
    
3- Install other requirements
        
        pip install numpy==1.18.1
        pip install nltk==3.5
        pip install tensorboardx==2 
    
### Train & Evaluate

We trained the BERT base model on 2 GPUs with bellow hyperparameters. Please note that by `do_sliding` and `do_answer_accumulation` you enable BERT to encode inputs longer than 512, which is the case in NLQuAD. E.g., here it slides sequences to windows of size 512 with 128 overlappings, and finally accumulates answers from each window. 
You can simply use 1 GPU, but your batch size would be 6 instead of 12 (in our paper). Check the output for **Intersection over Union (IoU)** or **Jaccard Index**. 

        python -u run_nlquad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --train_file /Path to Data/NLQuAD_train.json \
        --predict_file /Path to Data/NLQuAD_valid.json \
        --per_gpu_train_batch_size 6 \
        --learning_rate 3e-5 \
        --num_train_epochs 2.0 \
        --max_seq_length 512 \
        --doc_stride 128 \
        --output_dir /Where to save model \
        --overwrite_cache \
        --overwrite_output_dir \
        --do_sliding \ 
        --do_answer_accumulation \
        --isnt_caching_data \
        --max_answer_length=512 \
        --logging_steps=30000 \
        --save_steps=30000 \
        --warmup_steps 1000 \
## Longformer

### Requirements

Install environment and code (Allen AI Longformer V0.1)

        conda create --name longformer python=3.7
        conda activate longformer
        pip install nltk
        conda install cudatoolkit=10.0
        pip install git+https://github.com/allenai/longformer.git@v0.1
    
Download pre-trained `Longformer-base` model

        wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz
    
### Data preparation 
Simply do a small conversion using `convert_data_format.py`. Do this for the train, valid, and eval sets.

        python -u convert_data_format.py \
        --input_dir /Path do Data/NLQuAD_train.json
        --output_dir /Path to Data/NLQuAD_train_longformer.json 
        
### Train & Evaluate

We trained Longformer on one GPU using the old version of Allen AI Longformer code (V0.1) using below hyperparameters. 
There are newer versions, particularly in the [HuggingFace Transformers](https://github.com/huggingface/transformers) package.

        python -u run_nlquad_longformer.py  \
        --train_dataset /Path to Data/NLQuAD_train_longformer.json \
        --dev_dataset /Path to Data/NLQuAD_valid_longformer.json \
        --gpu 0 \
        --batch_size 12 \
        --num_workers 4 \
        --lr 0.00003 \
        --warmup 1000 \
        --epochs 5 \
        --max_seq_len 4096 \
        --doc_stride -1 \
        --save_prefix /Path for saving checkpoints \
        --model_path /pretrained model from AllenAI/longformer-base-4096 \
        --seed 4321 \
        --fp32 \
        --val_every 0.2 \
        --max_answer_length 1000 \
        --max_question_len 100 \
        

After training, you get `predictions.json`, and you need to run `eval_predictions_longformer.py` to get Precision, Recall, F1, EM, and IoU.
        
        python -u eval_predictions_longformer.py \
        --prediction_dir /Path to Prediction file /predictions.json 
        --data_dir /Path to Data/NLQuAD_valid_longformer.json


## Evaluation Metrics

The evaluation metrics have been already included in the BERT and Longformer codes. However, if you need to access direcrly you can use `Evaluation_Metrics/evaluation_metrics.py`. This evaluates `predictions` against the evaluation set. 

        python -u evaluation_metrics.py \
        --prediction_dir predictions_BERT.p
        --data_dir NLQuAD_eval_longformer.json