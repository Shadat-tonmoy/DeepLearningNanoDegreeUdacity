from nltk import word_tokenize, RegexpParser
from nltk import pos_tag


'''

Chunking is basically make chunk of similar parts of speech and group them together.

Chunk works with POS Tagging. The tagged words from POS tagger is chunked together  based on similar tag.

The chuking works based on some rule. The rules need to be defined using regular expression.

'''

example_text = "This is an example text to test chunking using nltk with pos tagger"

words = word_tokenize(example_text)

print(words)

tagged_words = pos_tag(words)

print(tagged_words)

chunking_rules = """chunks:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""

chunker = RegexpParser(chunking_rules)

chunked_tags = chunker.parse(tagged_words)

print(chunked_tags)

chunked_tags.draw()