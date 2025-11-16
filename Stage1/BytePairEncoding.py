import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, world! This is a test. |<endoftext>|"

integers = tokenizer.encode(text)
string = tokenizer.decode(integers)