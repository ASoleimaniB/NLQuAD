# NLQuAD: A Non-Factoid Long Question Answering Data Set
NLQuAD is a non-factoid long question answering dataset from BBC news articles. NLQuAD’s question types and the long length of its context documents as well as answers, make it a challenging real-world task.
NLQuAD consists of news articles as context documents, interrogative sub-headings in the articles as questions, and body paragraphs corresponding to the sub-headings as contiguous answers to the
questions. NLQuAD contains 31k non-factoid questions and long answers collected from 13k BBC news articles. See example articles in BBC [1](https://www.bbc.com/news/world-asia-china-51230011), [2](https://www.bbc.com/news/world-55709428). 
We automatically extract target answers because annotating for non-factoid long QA is extremely challenging and costly. 
 
## Dataset
Download a copy of dataset, distributed under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/3.0/) licence providing free access for non-commercial and academic usage. \
[Training Set](https://drive.google.com/file/d/1Yviu4C8kJYh8EpfJGzLjAR5-N_I39Y4D/view?usp=sharing) \
[Validation Set](https://drive.google.com/file/d/17rXbzbOL71baX5ArBN3wFzC8NAZFPf8G/view?usp=sharing)

## BERT 
### Requirements
First create a conda environment

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
You can simply use 1 GPU but your batch size would be 6 instead of 12 (in our paper). Check output for **Intersection over Union (IoU)** or **Jaccard Index**. 

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