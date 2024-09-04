# Multi-task model training and finetuning

Code to 1) pretrain a model on 3 of the 4 existing dataframes and 2) finetune on the remaining one

Simply call: 

python multi_task_model.py --finetune_on house-prices

If no pretrained model exists, this will first run the pretraining. If a model has already been trained, it will start
a finetuning on the house-prices dataframe.