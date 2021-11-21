#Step 1 : import required libraries
import nltk
import numpy as np
import random
import string

#Step 2 : Data reading
f=open('chatbot.txt','r',errors='ignore')
raw=f.read()
