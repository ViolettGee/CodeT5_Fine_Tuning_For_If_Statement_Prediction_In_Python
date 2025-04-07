#purpose of this file is to preprocess the data:
# - mask each of the target if statements
# - flatten each function

#input: columns: cleaned_method, target_block, tokens_in_method
#output: columns: flattend/masked method, target_block, tokens_in_method

#import necessary libraries
import pandas as pd
from pygments import lex
from pygments.lexers import PythonLexer

#function for masking the target if statement in a method
def target_masking(code_block, target):
    print(target)
    masked_code = code_block.replace(target, "<MASK>")
    print(masked_code)
    return masked_code

#function for tokenizing the current method
def tokenize_method(code_block):
    #tokenize data
    tokens = lex(code_block.strip(), PythonLexer())
    return list(tokens)

#main section of code iterating through the files

#initialize each data frame
training = pd.read_csv("Archive/ft_train.csv", encoding = "utf-8", dtype = "str")
testing = pd.read_csv("Archive/ft_test.csv", encoding = "utf-8", dtype = "str")
validating = pd.read_csv("Archive/ft_valid.csv", encoding = "utf-8", dtype = "str")
#initialize file iteration object
archive = [training, testing, validating]

#initialize output data frames
training_out = pd.DataFrame(columns = ["flattened/masked method", "target_block", "tokens_in_method"])
testing_out = pd.DataFrame(columns = ["flattened/masked method", "target_block", "tokens_in_method"])
validating_out = pd.DataFrame(columns = ["flattened/masked method", "target_block", "tokens_in_method"])
#initialize output file iterator object
updated = [training_out, testing_out, validating_out]

#iterate through each data frame
for frame in range(len(archive)):
    
    #iterate through each row
    for index, row in archive[frame].iterrows():

        #call the function for tokenizing the data
        tokenized_code = tokenize_method(row["cleaned_method"])
        #save tokens
        print(tokenized_code)
        row["tokens_in_method"] = tokenized_code

        #initalize method container
        cleaned_method = []
        #flatten the tokenized data
        for token in tokenized_code:
            if not(str(token[0]) == "Token.Text.Whitespace") and not(token[1].strip() == ""):
                cleaned_method.append(token[1].strip())
        cleaned_method = " ".join(cleaned_method)
        
        #call the function for masking on current row
        masked_method = target_masking(cleaned_method, row["target_block"])
            
        #update the tokens to correspond to the data
        row["cleaned_method"] = masked_method
    
        #update the row of the data frame
        updated[frame].loc[-1] = row
        updated[frame].index = updated[frame].index + 1
        updated[frame].index = updated[frame].sort_index()

#write each data frame to a new csv
training_out.to_csv("Processed_Data/training.csv")
testing_out.to_csv("Processed_Data/testing.csv")
validating_out.to_csv("Processed_Data/validating.csv")
