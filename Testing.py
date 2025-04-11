#Using the fine tuned model on the testing data and evaluating its performance

#input: model, tesing.csv: "flattened/masked_method", "target_block"
#output: output.csv: "input_method", "exact_match", "expected_if_condition", "predicted_if_condition", "CodeBLEU_score", "BLEU-4_score"
# - compute average BLEU-4
# - compute average CodeBLEU

#import necessary libraries
from transformers import T5ForConditionalGeneration
from transformers import RobertaTokenizer
from alive_progress import alive_bar
import evaluate
import codebleu
import pandas as pd

#function for predicting the if condition of a method
def generate_if_condition(code, tokenizer, model):
    
    #tokenize inputs to get ready for model
    input_ids = tokenizer(code, return_tensors = "pt").input_ids

    #generate sequence
    generated_ids = model.generate(input_ids)
    
    #decode the generated sequence
    predicted = tokenizer.decode(generated_ids[0])

    return predicted

#function for exact match check of an output
def exact_match(expected, predicted):

    #check for exact match
    if expected == predicted:
        return("Yes")
    else:
        return("No")

#function for computing CodeBLEU_score of an output
def CodeBLEU_score_method(expected, predicted):
    
    #compute codeBLEU scores
    results = codebleu.calc_codebleu(predictions = [predicted], references = [expected], lang = 'python')
    
    return results

#function for computing BLEU-4_score of an output
def BLEU4_score_method(expected, predicted):

    #load sacrebleu metric
    sacrebleu = evaluate.load("sacrebleu")

    #comput sacrebleu metric
    results = sacrebleu.compute(predictions = [predicted], references = [expected])
    
    return results

#function for adding a row to the output file
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    df = df.sort_index()
    return df

#import tokenizer object
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
#import fine tuned model
model = T5ForConditionalGeneration.from_pretrained("Model/checkpoint-18750")

#import testing data file
testing = pd.read_csv("Tokenized_Data/testing.csv")

#initialize output file
output = pd.DataFrame(columns = ["input_method", "exact_match", "expected_if_condition", "predicted_if_condition", "CodeBLEU_score", "BLEU-4_score"])

#initialize progress bar
with alive_bar(testing.shape[0]) as bar:
    
    #iterate through testing data
    for index, row in testing.iterrows():
        
        #function call to compute if condition generation
        predicted = generate_if_condition(row["tokenized_method"], tokenizer, model)
        
        #function call to compute exact match
        match = exact_match(row["tokenized_target"], predicted)

        #function call to compute CodeBLEU score
        CodeBLEU = CodeBLEU_score_method(row["tokenized_target"], predicted)

        #function call to compute BLEU-4 score
        BLEU4 = BLEU4_score_method(row["tokenized_target"], predicted)

        #initialize output row
        new_row = [row["tokenized_method"], match, row["tokenized_target"], predicted, CodeBLEU['codebleu'], BLEU4['score']]
        
        #function call to add row to output file
        output = add_row(output, new_row)

        #increment progress bar
        bar()