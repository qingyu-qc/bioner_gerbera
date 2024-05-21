# GERBERA
We present Gerbera (Transfer Learning for **Ge**ne**r**al-to-**B**iomedical **E**ntity **R**ecognition **A**ugmentation), a multi-task learning method that utilizes knowledge from general-domain NER datasets to improve performance on BioNER datasets, specially on limited-sized dataset.

## Install GERBERA Environment
To set up the GERBERA environment, please follow these steps. Ensure you have the correct version of [conda](https://pytorch.org/) version.
```
# Install torch
conda create -n GERBERA python=3.7
conda activate GERBERA
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
conda install faiss-gpu libfaiss-avx2 -c conda-forge

# Check if cuda is available
python -c "import torch;print(torch.cuda.is_available())"

# Install GERBERA
git clone https://github.com/qingyu-qc/bioner_gerbera.git
cd bioner_gerbera
pip install -r requirements.txt

```

## Dataset
Please download the necessary BioNER and general-domain NER datasets from [here](https://drive.google.com/drive/folders/1InU7REZ-fXBVglUAuyH4U8v5X6x98OD7?usp=drive_link). Ensure the datasets are placed in the correct directory structure as expected by the training scripts.

## Models
You can download our GERBERA model from [here](https://huggingface.co/Euanyu/GERBERA-NCBI) for BioNER tasks, including disease, Gene, Chemical, Species, DNA, RNA, Cell type and Cell line.

## Download the baseline model
Download the pre-trained baseline model for initialization.
```
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -zxvf RoBERTa-large-PM-M3-Voc-hf.tar.gz
```

## Training
### Multi-task training:
GERBERA training with the BioNER dataset and the general-domain NER dataset.
```
python run_ner.py 
--model_name_or_path ./RoBERTa-large-PM-M3-Voc-hf 
--data_dir NERdata/ 
--labels NERdata/NCBI-disease/labels.txt 
--output_dir ./gerbera_model 
--data_list NCBI-disease+CoNLL2003 
--eval_data_list NCBI-disease 
--num_train_epochs 20 
--max_seq_length 128 
--warmup_steps 0 
--learning_rate 3e-5 
--per_device_train_batch_size 16 
--per_device_eval_batch_size 16 
--seed 1 
--logging_steps 5000 
--evaluate_during_training 
--save_steps 10000 
--do_train 
--do_eval 
--do_predict 
--overwrite_output_dir 
```

### Biomedical finetuning
After intial multi-task training, further finetuning the saved model with specific BioNER dataset.
```
python run_ner.py 
--model_name_or_path ./gerberal_model/RoBERTa-ncbi # saved multi-task model
--data_dir NERdata/ 
--labels NERdata/NCBI-disease/labels.txt 
--output_dir ./gerbera_model 
--data_list NCBI-disease
--eval_data_list NCBI-disease 
--num_train_epochs 10 
--max_seq_length 128 
--warmup_steps 0 
--learning_rate 3e-5 
--per_device_train_batch_size 16 
--per_device_eval_batch_size 16 
--seed 1 
--logging_steps 5000 
--evaluate_during_training 
--save_steps 10000 
--do_train 
--do_eval 
--do_predict 
--overwrite_output_dir 
```

## Evaluation
Evaluate the fine-tuned model on various BioNER datasets to measure its performance.
```
python run_eval.py 
--model_name_or_path ./gerberal_model/RoBERTa-ncbi
--data_dir NERdata/ 
--labels NERdata/NCBI-disease/labels.txt 
--output_dir ./gerbera_model 
--eval_data_type linnaeus 
--eval_data_list linnaeus 
--max_seq_length 128 
--per_device_eval_batch_size 32 
--seed 1 
--do_eval 
--do_predict 
--overwrite_output_dir
```


## Contact Information
For help or issues using GERBERA, please submit a GitHub issue. Please contact with Yu Yin(`yinyu201906 (at) gmail (dot) com`) for communication related to GERBERA.