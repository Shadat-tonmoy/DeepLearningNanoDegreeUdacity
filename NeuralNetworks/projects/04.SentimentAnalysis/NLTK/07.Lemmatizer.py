from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
'''

WordLemmatizer is similar to stemmer. The only difference is that in lemmatizer the root word is return as a synonyms of the actual 
word. For example, in Stemmer better will be returned to better but word lemmatizer return it as good

Lemmatizer return the original root word when the parts of speech is defined


'''

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("Cats"))
pos = pos_tag(["Better"])
pos = pos[0][1]

print(lemmatizer.lemmatize("better",pos="a"))