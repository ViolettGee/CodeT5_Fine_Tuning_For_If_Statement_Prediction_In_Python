#purpose to tokenize the files

#input: flattened/masked method, target_block
#output: tokenized_method, tokenized_target

#import necessary libraries
import pandas as pd
from alive_progress import alive_bar
from transformers import RobertaTokenizer

#function to tokenize code section
def tokenizing_code_block(code, tokenizer):

    #encode the the inputs
    encode = tokenizer(code, return_tensors = "pt").input_ids
    #decode the inputs
    decode = tokenizer.decode(encode[0])
    
    return decode

#function to add row to output
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    df = df.sort_index()
    return df

#initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

#initialize data sets
training = pd.read_csv("Processed_Data/training.csv")
testing = pd.read_csv("Processed_Data/testing.csv")
validating = pd.read_csv("Processed_Data/validating.csv")

#initialize data frame container
archive = [training, testing, validating]

#initialize output data frames
training_out = pd.DataFrame(columns = ["tokenized_method", "tokenized_target"])
testing_out = pd.DataFrame(columns = ["tokenized_method", "tokenized_target"])
validating_out = pd.DataFrame(columns = ["tokenized_method", "tokenized_target"])

#create progress bar
with alive_bar(training.shape[0]+testing.shape[0]+validating.shape[0]) as bar:
    
    #iterate through each data frame
    for frame in range(len(archive)):
        
        #iterate through each row of current data frame
        for index, row in archive[frame].iterrows():

            #call the function for tokenizing the method
            tokenized_method = tokenizing_code_block(row["flattened/masked method"], tokenizer)
            #call the function for tokenizing the target
            tokenized_target = tokenizing_code_block(row["target_block"], tokenizer)

            #initialize the output row
            new_row = [tokenized_method, tokenized_target]

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
training_out.to_csv("Tokenized_Data/training.csv")
testing_out.to_csv("Tokenized_Data/testing.csv")
validating_out.to_csv("Tokenized_Data/validating.csv")