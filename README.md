# CodeTF_Fine_Tuning_For_If_Statement_Prediction_In_Python

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)  
* [3 Report](#3-report)

---

# **1. Introduction**
This project explores **Python if-statmenet prediction**, using fine tuning of the small CodeT5 modell offered by Hugging Face. The model predicts the masked if statement of a inputed method labeled using the "<MASK>" token. The fine-tuning explores the possibility for leveraging sophisticated models for more specialized tasks.

---

# **2. Getting Started**

This project is implement in **Python 3.12.9** and is compatible with **macOS, Linux, and Windows**.

## **2.1 Preparations**

(1) Clone the repository to your workspace:
'''shell
~ $ git clone https://github.com/ViolettGee/CodeT5_Fine_Tuning_For_If_Statement_Prediction_In_Python.git
'''

## **2.2 Install Packages**

Install the required dependencies:

(venv) $ pip install pandas==2.2.3
       $ pip install numpy==1.18.5
       $ pip install pygments
       $ pip install alive_progress
       $ pip install transformers
       $ pip install transformers[torch]
       $ pip install tf-keras
       $ pip install evaluate
       $ pip install sacrebleu
       $ pip install codebleu
       $ pip install tree-sitter-python version 0.23.6

## **2.3 Import Fine-Tuned Model**

The model can be found at https://drive.google.com/drive/folders/1HJ04K5MgQ4qAzWtYtUGTCOzTaBLHDUp7?usp=sharing download the files and move the "Model" folder to the same directory as the Python files. The folder is too large for github to handle without paying for more storage space.

## **2.4 Run Code-T5 Fine Tuning**

The scripts should be run in the order as described below if you are completely re-initializing the model. However, if you download the model as descirbed above, all files should be runable regardless of order. Without the model loaded in, "Data_Preprocessing.py", "Data_Tokenization.py", "Model.py", and "Evaluation.py" should have all have the necessary dependent files in the repository and be runnable.

1. "Data_Preprocessing.py"
   The file flattens and masks the datasets within the "Archive" folder labeled as "ft_test.csv", "ft_train.csv", and "ft_valid.csv". These files are in the format: cleaned_method, target_block, and tokens_in_method. Once flattened and masked, the ouputs are exported to the "Processed_Data" folder labeled "training.csv", "testing.csv" and "validating.csv" respectively. These outputs are in the format: flattend/masked method and target_block. File run-time is about 5 minutes.
2. "Data_Tokenization.py"
   The file tokenizes and decode the datasets within the "Processed_Data" folder labeled as "testing.csv", "training.csv", and "validating.csv". These files are in the format: flattend/masked method and target_block. Once tokenized, the outputs are exported to the "Tokenized_Data" folder labeled "training.csv", "testing.csv" and "validating.csv" respectively. These outputs are in the format: tokenized_method and tokenized_target. File run-time is about 10 minutes.
3. "Model.py"
   The file fine-tunes the small CodeT5 model using the datasets in the "Tokenized_Data" folder labeled as "training.csv" and "validating.csv". Thes files are in the format: tokenized_method and tokenized_target. During training, the file exports the model data to a folder labeled "Model" where the checkpoints and run data can be found. Run time is about 48 hours without a GPU and ranges depending on GPU specs with one.
4. "Testing.py"
   The file uses the fine-tuned model to predict if-statements for the testing dataset, computing exact match, CodeBLEU score and BLEU-4 score. The dataset is found at "Tokenized_Data/testing.csv". The dataset is in the format: tokenized_method and tokenized_target. The model is found at "Model/checkpoint-18750" when loaded. After computing the predictions, the output is exported to "output.csv" in the format: input_method, exact_match, expected_if_condition, predicted_if_condition, CodeBLEU_score, and BLEU-4_score. Run time is about 40 minutes.
5. "Evaluation.py"
   The file computes the average CodeBLEU score, average BLEU-4 score, precision, recall and F1 score using the data from "output.csv". The file is in the format: input_method, exact_match, expected_if_condition, predicted_if_condition, CodeBLEU_score, and BLEU-4_score. Run time is negligible.

## 3. Report

The assignment report is available in the file "GenAI for Software Development Assignment 2.docx".
