#this file initializess and fine-tunes the model based on the tokenized training data
#uses evaluation data to find a stopping point

#import necessary libraries

#initialize training parameters

#output_dir is the directory where model predictions and checkpoints will be written
#do_eval is the whether or not to run evaluation on the validation set
#eval_strategy is the type of evaluation strategy to adopt during training
#per_device_train_batch_size is the size per device accelerator core/CPU for training
#per_device_eval_batch_size is the size per device accelerator core/CPU for training
#gradient_accumulation_steps is the number of updates steps to accumulate the gradients for before back propogation
#eval_dely is the number of epochs/steps to wait for before the first evaluation can be performed
#learning_rate is the initial learning fro the optimizer
#weight decay is the decay to apply to all layers except bias and normalization weights in the optimizer
#load_best_model_at_end is the whether or not to load the best model found during training
#metric_for_best_model is used in conjunction with load_bes_model_at_end to specify the model comparison metric
# - used loss metric due to it being asked for within project specifications
#greater_is_better is used in conjunction with model evaluation specifying which model is better

#initialize pretrained model

#initialize tokenizer

#initialize data collator used padding due to others being built to include labels

#initalize callback object

#used early stopping due to it being asked for within the project specifications

#initialize tokenized training and validation data

#initialize trainer object

#model is the model that is being trained, evaluated and used for productions
#args is the training arguments that are initialized above
#data_collator is the collator object initialized above
#train_dataset is the dataset to use for training
#eval_dataset is the dataset to use for evaluation
#callbacks is the callback object initialized above