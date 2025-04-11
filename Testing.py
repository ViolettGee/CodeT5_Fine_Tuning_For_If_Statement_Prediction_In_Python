#Using the fine tuned model on the testing data and evaluating its performance

#input: model, tesing.csv: "flattened/masked_method", "target_block"
#output: output.csv: "input_method", "exact_match", "expected_if_condition", "predicted_if_condition", "CodeBLEU_score", "BLEU-4_score"
# - compute overall F1
#    - compute precision
#    - compute recall
# - compute average BLEU-4
# - compute average CodeBLEU
# - compute perplexity

#import necessary libraries

#function for predicting the if condition of a method

#function for exact match check of an output

#function for computing CodeBLEU_score of an output

#function for computing BLEU-4_score of an output

#function for adding a row to the output file

#function for computing precision

#function for computing recall

#function for computing F1

#function for computing perplexity

#import fine tuned model

#import testing data file

#initialize progress bar

    #iterate through testing data

        #input method into the model

        #catch output

        #function call to compute exact match

        #function call to compute CodeBLEU score

        #functin call to compute BLEU-4 score

        #initialize output row

        #function call to add row to output file

#function call to compute precision
#function call to compute recall
#function call to compute F1 score

#function call to compute perplexity

    

