import numpy as np
import pandas as pd
import pickle
import random


def getMcIntireDataset():
    filename = "dataset/mcIntire.csv"
    data = pd.read_csv(filename)
    x = data['text'].tolist()



    m = {'FAKE':1,'REAL':0}
    y = data['label'].map(m).tolist()
    y = np.array(y)
    print("CSV file loaded!!")
    return x,y




def getLiarLiarDataset():
    TRAIN = "dataset/liar_dataset/train.tsv"
    TEST = "dataset/liar_dataset/test.tsv"
    VALID = "dataset/liar_dataset/valid.tsv"

    def getLiarData(fileName):
        data = pd.read_csv(fileName, sep='\t')
        data = data[data['the label']!='half-true']
        x = data['the statement'].tolist()
        m = {'true': 0, 'mostly-true': 0, 'barely-true': 1, 'false': 1, 'pants-fire': 1}
        y = data['the label'].map(m).tolist()
        print("CSV file loaded!!")
        return x, y

    x_train, y_train = getLiarData(TRAIN)
    x_test, y_test = getLiarData(TEST)
    x_valid, y_valid = getLiarData(VALID)

    x = x_train + x_test + x_valid
    y = y_train + y_test + y_valid
    y = np.array(y)

    return x,y

def twitterHarvardDataset():
    filename = "dataset/dataHarvard"
    f = open(filename,"rb")
    nonRumor = pickle.load(f)
    rumor = pickle.load(f)

    random.seed(5)

    x1 = random.sample(nonRumor,k=50000)
    y1 = [0 for i in range(0,50000)]

    x2 = random.sample(rumor,k=50000)
    y2 = [1 for i in range(0,50000)]

    x = x1+x2
    y = y1+y2

    c = list(zip(x,y))

    random.shuffle(c)

    x,y = zip(*c)
    x = list(x)
    y = np.array(y)

    print(len(x))
    print(y.shape)

    return x,y


def textpreprocessing(x,y,file):
    ##Convert to lower case
    x = [item.lower() for item in x]

    ##Remove numbers
    import re
    x = [re.sub(r'\d+','', i) for i in x]

    ##Remove_hashtags_@
    x = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i).split()) for i in x]

    ##Remove punctuations
    import string
    translator = str.maketrans(string.punctuation+'—…', ' '*(len(string.punctuation)+2))
    x = [i.translate(translator) for i in x]

    ##Remove whitespaces
    x = [" ".join(i.split()) for i in x]

    ##Remove stopwords
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # remove stopwords function
    def remove_stopwords(text):
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        filtered_text = " ".join(filtered_text)
        return filtered_text

    x = [remove_stopwords(i) for i in x]

    ##Lemmatization
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lemmatizer = WordNetLemmatizer()

    # lemmatize string
    def lemmatize_word(text):
        word_tokens = word_tokenize(text)
        # provide context i.e. part-of-speech
        lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
        lemmas = " ".join(lemmas)
        return lemmas

    x = [lemmatize_word(i) for i in x]

    print("Text preprocessing done!!")


    ##save to file
    import pickle
    f = open(file,"wb")
    pickle.dump(x,f)
    pickle.dump(y,f)


x1,y1 = getMcIntireDataset()
x2,y2 = getLiarLiarDataset()
x3,y3 = twitterHarvardDataset()
x = list(x1)+list(x2)+list(x3)
y = np.concatenate((y1,y2,y3))
print(len(x),len(y))
textpreprocessing(x,y,"combined")

