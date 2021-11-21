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

#Step 3.2 : Preprocessing raw text (lemmatization)
lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Step 4 : Programming a greet response
