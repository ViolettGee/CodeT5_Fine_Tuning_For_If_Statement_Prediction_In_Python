#file tokenizes the preprocessed data

#input columns: flattend/masked method, target_block
#output columns: tokenized_method, embeded_method, tokenized_target, embeded_target

#import necessary libraries
import pandas as pd
from transformers import RobertaTokenizer
from alive_progress import alive_bar

#function tokenizing a code section
def tokenize_code(code, tokenizer):
    #tokenize data
    tokens = tokenizer.tokenize(code)
    #encode tokens
    embedding = tokenizer.encode(tokens)
    return tokens, embedding

#function creating a row in the data frame
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    df = df.sort_index()
    return df

#main section of code iterating through data

#initialize dataframes for data set
training = pd.read_csv("Processed_Data/training.csv")
testing = pd.read_csv("Processed_Data/testing.csv")
validating = pd.read_csv("Processed_Data/validating.csv")
#initialize file input iterator object
processed_data = [training, testing, validating]

#initialize output dataframes for data set
training_out = pd.DataFrame(columns=["tokenized_method","embeded_method","tokenized_target","embedded_target"])
testing_out = pd.DataFrame(columns=["tokenized_method","embeded_method","tokenized_target","embedded_target"])
validating_out = pd.DataFrame(columns=["tokenized_method","embeded_method","tokenized_target","embedded_target"])

#initialize pre-trained tokenizer and embedding
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

#create progress bar
with alive_bar(training.shape[0]+testing.shape[0]+validating.shape[0]) as bar:
    #iterate through input files
    for frame in range(len(processed_data)):
        #iterate through rows in input files
        for index, row in processed_data[frame].iterrows():
            
            #tokenize the method data
            tokenized_method, embedded_method = tokenize_code(row["flattened/masked method"], tokenizer)

            #tokenize the target data
            tokenized_target, embedded_target = tokenize_code(row["target_block"], tokenizer)

            #initialize output row 
            new_row = [tokenized_method, embedded_method, tokenized_target, embedded_target]

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

#write each output dataframe to a new csv
training_out.to_csv("Tokenized_Data/training.csv")
testing_out.to_csv("Tokenized_Data/testing.csv")
validating_out.to_csv("Tokenized_Data/validating.csv")