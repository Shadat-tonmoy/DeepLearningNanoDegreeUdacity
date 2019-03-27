from nltk.tokenize import PunktSentenceTokenizer,word_tokenize
from nltk import pos_tag
from nltk.corpus import state_union


'''

Parts of Speech (POS) tagging is basically getting the parts of speech of a word inside a sentence

NLTK has pos tagger inside its package

'''

'''
NLTK Parts of Speech tags

CC	    coordinating conjunction
CD	    cardinal digit
DT	    determiner
EX	    existential there (like: "there is" ... think of it like "there exists")
FW	    foreign word
IN	    preposition/subordinating conjunction
JJ	    adjective	'big'
JJR	    adjective, comparative	'bigger'
JJS	    adjective, superlative	'biggest'
LS	    list marker	1)
MD	    modal	could, will
NN	    noun, singular 'desk'
NNS	    noun plural	'desks'
NNP	    proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	    predeterminer	'all the kids'
POS	    possessive ending	parent's
PRP	    personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	    adverb	very, silently,
RBR	    adverb, comparative	better
RBS	    adverb, superlative	best
RP	    particle	give up
TO	    to	go 'to' the store.
UH	    interjection	errrrrrrrm
VB	    verb, base form	take
VBD	    verb, past tense	took
VBG	    verb, gerund/present participle	taking
VBN	    verb, past participle	taken
VBP	    verb, sing. present, non-3d	take
VBZ	    verb, 3rd person sing. present	takes
WDT	    wh-determiner	which
WP	    wh-pronoun	who, what
WP$	    possessive wh-pronoun	whose
WRB	    wh-abverb	where, when


'''

example_text = "This is an example text to test the NLTK parts of speech tagging"

words = word_tokenize(example_text)

print(words)



for word in words:
    # word_list = [word]
    tokenized_word = word_tokenize(word) # return a list of word from a sentence or if a single word then covert it to a list
    # print(tokenized_word)
    print(pos_tag(tokenized_word)) # pos_tag function takes a list of word as input


train_text = state_union.raw("2005-GWBush.txt") # state union is a text corpus. we just use the raw format of a text file from that corpus as training data
sample_text = state_union.raw("2006-GWBush.txt") # use another text file as a sample data

custom_sentence_tokenizer = PunktSentenceTokenizer(train_text) # initialize PunktSentenceTokenizer with training data

tokenized = custom_sentence_tokenizer.tokenize(sample_text) # tokenize sample data. It is a sentence tokenizer

def process_content():
    try:
        for i in tokenized:
            words = word_tokenize(i) # tokenize words from sentences
            tagged = pos_tag(words) # getting the tags from words
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()



