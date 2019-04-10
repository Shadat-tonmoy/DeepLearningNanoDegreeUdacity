from nltk.tokenize import word_tokenize,sent_tokenize

'''
Stuff can be done with NLTK

01. Tokenizing - Word Tokenizer : Separates by works, Sentence Tokenizer : Separates by Sentences

02. Corpora - Body of text like Medical Journals, Presidential Speeches 

03. Lexicons - Words and their meaning

Meaning can be different based on situations. For example 

Investors speak 'bull' = Someone who is positive about market
English speak 'bull = Scary animal

'''

text = "Hello there Mr. Smith, how are you doing today? The weather is grat and python is awsome. The sky is pinkish-blue and please don't eat card board. Card boards are not  that awsome!!"

sentences = sent_tokenize(text)
words = word_tokenize(text)
print(sentences)