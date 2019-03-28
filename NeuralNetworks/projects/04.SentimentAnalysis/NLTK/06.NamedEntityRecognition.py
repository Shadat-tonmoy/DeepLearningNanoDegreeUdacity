from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

'''

Named entity recognition or NER is the task of recognizing entity fromm a large chunk of words or sentences

For example
United States   -   Country/Place
Jorge W Bush    -   Person
US Dollar       -   Currency

NLTK has ne.chunk() for performing named entity recognition tasks.

ne.chunk() works along with POS tagged words

'''


example_text = "The name of Staff Sergeant Travis Atkins will be etched alongside the names of America’s bravest warriors and written forever into America’s heart"

words = word_tokenize(example_text)             # tokenizing words from given sentence

tagged_words = pos_tag(words)                   # pos tag the tokenized words

print(tagged_words)

named_entity =  ne_chunk(tagged_words)          # extracting named entity from the tagged tokenized words

# print(named_entity)

named_entity.draw()                             # drawing named entity tree


