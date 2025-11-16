import re

class SimpleTokenizerV1: 
    def __init__(self, vocab):
        self.str_to_int = vocab # converts string to unique integer (a dictionary method)
        self.int_to_str = {i:s for s,i in vocab.items()} # converts unique integer back to string

    def encoder(self, text):
        preprocessed_text = re.split(r'([,.:;!?_"()\']|--|\s)', text) # splits texts into tokens along with punctuations 

        preprocessed_text = [item.split() for item in preprocessed_text if item.strip()] # strips the white spaces
        token_ids = [self.str_to_int[s] for s in preprocessed_text] # converts the tookens into their respective unique integers
        return token_ids
    
    def decoder(self, token_ids):
        text = " ".join([self.int_to_str[s] for s in token_ids]) # converts the unique integers back to their respective tokens using the dictionary function along with spaces
        text = re.sub(r'([,.:;!?_"()\']|--|\s)', r'\1', text) # replaces spaces with proper punctuations in the text
        return text

class SimpleTokenizerV2: 
    def __init__(self, vocab):
        self.str_to_int = vocab # converts string to unique integer (a dictionary method)
        self.int_to_str = {i:s for s,i in vocab.items()} # converts unique integer back to string

    def encoder(self, text):
        preprocessed_text = re.split(r'([,.:;!?_"()\']|--|\s)', text) # splits texts into tokens along with punctuations 
        preprocessed_text = [item.split() for item in preprocessed_text if item.strip()] # strips the white spaces

        preprocessed_text = [
            item if item in self.str_to_int
            else "|<unk>|" for item in preprocessed_text
        ] # replaces out of vocabulary tokens with |<unk>| token

        token_ids = [self.str_to_int[s] for s in preprocessed_text] # converts the tookens into their respective unique integers
        return token_ids
    
    def decoder(self, token_ids):
        text = " ".join([self.int_to_str[s] for s in token_ids]) # converts the unique integers back to their respective tokens using the dictionary function along with spaces
        text = re.sub(r'([,.:;!?_"()\']|--|\s)', r'\1', text) # replaces spaces with proper punctuations in the text
        return text