#this file initializess and fine-tunes the model based on the tokenized training data
#uses evaluation data to find a stopping point

#import necessary libraries
from transformers import TrainingArguments
from transformers import RobertaTokenizer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers import Trainer
from datasets import load_dataset

#initialize training parameters
training_args = TrainingArguments(output_dir = "Model", 
                                  eval_strategy = "epoch", 
                                  learning_rate = 0.05, 
                                  save_strategy = "epoch",
                                  load_best_model_at_end = True,
                                  remove_unused_columns = False)
#output_dir is the directory where model predictions and checkpoints will be written

#eval_strategy is the type of evaluation strategy to adopt during training
# - epoch because I am not implementing logs
#learning_rate is the initial learning fro the optimizer
# - increased from generic due to the relatively small size of the fine-tuning data set
#load_best_model_at_end is the whether or not to load the best model found during training
# - true so that selection metrics can be run for early stopping

#initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

#initialize pretrained model
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

#initialize data collator used padding due to others being built to include labels
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

#initalize callback object
callbacks = EarlyStoppingCallback()
#used early stopping due to it being asked for within the project specifications

#initialize tokenized training and validation data
training_data = load_dataset("csv", data_files = "Tokenized_Data/training.csv")
print(training_data['train'])
validation_data = load_dataset("csv", data_files = "Tokenized_Data/validating.csv")

#initialize trainer object
trainer = Trainer(model,
                  training_args,
                  train_dataset = training_data['train'],
                  eval_dataset = validation_data,
                  data_collator = data_collator,
                  tokenizer = tokenizer,
                  callbacks = [callbacks])
#model is the model that is being trained, evaluated and used for productions
#args is the training arguments that are initialized above
#data_collator is the collator object initialized above
#train_dataset is the dataset to use for training
#eval_dataset is the dataset to use for evaluation
#callbacks is the callback object initialized above

#start training the model
trainer.train()