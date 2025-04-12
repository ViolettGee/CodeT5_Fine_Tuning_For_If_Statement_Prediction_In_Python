#purpose of the file is to compute averages of inidividual measures and F1 score and perplexity

#inputs: "input_method", "exact_match", "expected_if_condition", "predicted_if_condition", "CodeBLEU_score", "BLEU-4_score"
#outputs: average CodeBLEU, average BLEU-4, precision, recall, F1 score

#import necessary libraries
import pandas as pd
import re

#function for computing F1 score
def F1_score(TP, FP, TN, FN):
    
    #computations
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1, precision, recall

#add the row to the relevant data frame
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    df = df.sort_index()
    return df

#initialize the evaluated test cases for computation
outputs = pd.read_csv("output.csv")

#generate data frames to contain the positives and negatives
positives = pd.DataFrame(columns = ["input_method", "exact_match", "expected_if_condition", "predicted_if_condition", "CodeBLEU_score", "BLEU-4_score"])
negatives = pd.DataFrame(columns = ["input_method", "exact_match", "expected_if_condition", "predicted_if_condition", "CodeBLEU_score", "BLEU-4_score"])

#count the positives and negatives
for index, row in outputs.iterrows():
    #positives are all outputs that generated if statements
    if re.search("if", row["predicted_if_condition"]):
        positives = add_row(positives, row)
    #negatives are all outputs that didn't generate if statements
    else:
        negatives = add_row(negatives, row)

#count the number of true positives
#true positives are correct if-statements
tp_df = positives[positives["CodeBLEU_score"] >= 0.45]
TP = tp_df.shape[0]

#count the number of false positives
#false positives are incorrect if-statements
fp_df = positives[positives["CodeBLEU_score"] < 0.45]
FP = fp_df.shape[0]

#count the number of true negatives
#true negatives are correctly generated non-if statements
tn_df = negatives[negatives["CodeBLEU_score"] >= 0.45]
TN = tn_df.shape[0]

#count the number of false negatives
#false negatives are incorrectly generated non-if statements
fn_df = negatives[negatives["CodeBLEU_score"] < 0.45]
FN = fn_df.shape[0]

#function call for F1 score
F1, precision, recall = F1_score(TP, FP, TN, FN)

#output average CodeBLEU
print(f"Average CodeBLEU score: {outputs["CodeBLEU_score"].mean()}")
#output average BLEU-4
print(f"Average BLEU-4 score: {outputs["BLEU-4_score"].mean()}")
#output precision
print(f"Precision: {precision}")
#output recall
print(f"Recall: {recall}")
#output F1 score
print(f"F1-score: {F1}")

