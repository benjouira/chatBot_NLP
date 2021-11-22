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
Greeting_Inputs=['hello','hi','hey','bjr']
Greeting_Responses=['hi','hello','hi there']
def greeting(sentence):
  for word in sentence.split():
    if word.lower() in Greeting_Inputs:
      return random.choice(Greeting_Responses)
    
#Step 5.1 : Generate responses (first : import scikit learn libraries)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Step 5.2 : Generate responses
def response(user_response):
  robo_response=''
  sent_tokens.append(user_response)
  TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
  tfidf=TfidfVec.fit_transform(sent_tokens)
  vals=cosine_similarity(tfidf[-1],tfidf)
  idx=vals.argsort()[0][-2]
  flat=vals.flatten() #change el matrix to list
  flat.sort()
  req_tfidf=flat[-2]
  if (req_tfidf==0):
    robo_response=robo_response+" I'm sorry, I don't understand you "
    return robo_response
  else:
    robo_response=robo_response+sent_tokens[idx]
    return robo_response
  
#Step 6 : Programming Start and End points for conversation
flag=True
print("Can I help you ?")
while flag==True:
  user_response=input()
  user_response=user_response.lower()
  if (user_response!='bye'):
    if (user_response=='thanks'):
      flag=False
      print('You are welcome')
    else:
      if (greeting(user_response)!=None):
        print(greeting(user_response))
      else:
        print(response(user_response))
        sent_tokens.remove(user_response)
  else:
    flag=False
    print('bye')

#
