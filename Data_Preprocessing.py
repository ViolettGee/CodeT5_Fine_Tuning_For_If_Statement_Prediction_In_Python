#purpose of this file is to preprocess the data:
# - mask each of the target if statements
# - flatten each function

#input: columns: cleaned_method, target_block, tokens_in_method
#output: columns: flattend/masked method, target_block

#import necessary libraries
import pandas as pd
from pygments import lex
from pygments.lexers import PythonLexer
from transformers import RobertaTokenizer
from alive_progress import alive_bar

#function for masking the target if statement in a method
def target_masking(code_block, target):
    masked_code = code_block.replace(target, "<MASK>")
    return masked_code

#function for tokenizing the current method
def flatten_method(code_block):
    #tokenize data
    tokens = lex(code_block.strip(), PythonLexer())

    #initalize method container
    cleaned_method = []
    #flatten the tokenized data
    for token in list(tokens):
        if not(str(token[0]) == "Token.Text.Whitespace") and not(token[1].strip() == ""):
            cleaned_method.append(token[1].strip())
    cleaned_method = " ".join(cleaned_method)
    
    return cleaned_method

#function for tokenizing and embedding code
def tokenize_method(code_block, tokenizer):
    #tokenize data
    tokens = tokenizer.tokenize(code_block)

    #locate the masking in the tokens
    index = tokens.index("MASK")

    #change the tokenized mask to be a token "<MASK>"
    tokens.pop(index+1)
    tokens.pop(index-1)
    tokens[index-1] = "<MASK>"

    #encode tokens
    embedding = tokenizer.encode(tokens)
    return tokens, embedding

#add the row to the relevant data frame
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    df = df.sort_index()
    return df

#main section of code iterating through the files

#initialize pre-trained tokenizer and embedding
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

#initialize each data frame
training = pd.read_csv("Archive/ft_train.csv", encoding = "utf-8", dtype = "str")
testing = pd.read_csv("Archive/ft_test.csv", encoding = "utf-8", dtype = "str")
validating = pd.read_csv("Archive/ft_valid.csv", encoding = "utf-8", dtype = "str")
#initialize file iteration object
archive = [training, testing, validating]

#initialize output data frames
training_out = pd.DataFrame(columns = ["flattened/masked method", "target_block")
testing_out = pd.DataFrame(columns = ["flattened/masked method", "target_block")
validating_out = pd.DataFrame(columns = ["flattened/masked method", "target_block")

#create progress bar
with alive_bar(training.shape[0]+testing.shape[0]+validating.shape[0]) as bar:
    #iterate through each data frame
    for frame in range(len(archive)):
        
        #iterate through each row
        for index, row in archive[frame].iterrows():
    
            #call the function for tokenizing the data
            flattened_code = flatten_method(row["cleaned_method"])
            flattened_target = flatten_method(row["target_block"])
            
            #call the function for masking on current row
            masked_method = target_masking(flattened_code, flattened_target)
                
            #update the tokens to correspond to the data
            new_row = [masked_method, flattened_target)
            
            #update training data
            if frame == 0:
                training_out = add_row(training_out, new_row)
                
            #update test data
            elif frame == 1:
                testing_out = add_row(testing_out, new_row)
    
            #update validation data
            else:
                validating_out = add_row(validating_out, new_row)

            #increment progress bar
            bar()

#write each data frame to a new csv
training_out.to_csv("Processed_Data/training.csv")
testing_out.to_csv("Processed_Data/testing.csv")
validating_out.to_csv("Processed_Data/validating.csv")
