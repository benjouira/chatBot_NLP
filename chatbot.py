#Step 1 : import required libraries
import nltk
import numpy as np
import random
import string

#Step 2 : Data reading
f=open('chatbot.txt','r',errors='ignore')
raw=f.read()

#Step 3.1 : Preprocessing raw text (tokenization)
raw=raw.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens=nltk.sent_tokenize(raw)
word_tokens=nltk.word_tokenize(raw)
