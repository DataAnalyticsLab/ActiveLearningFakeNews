import numpy as np
import pandas as pd
import pickle
import re
import random
import string


random.seed(10)

def getFakeNews():
    filename1 = "bangla/Fake-1K.csv"
    filename2 = "bangla/LabeledFake-1K.csv"
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)

    data1['text'] = data1['headline']+data1['content']
    data2['text'] = data2['headline'] + data2['content']




    x = data1['text'].tolist() + data2['text'].tolist()

    y = data1['label'].tolist() + data2['label'].tolist()
    print("CSV file loaded!!")
    return x, y


def getRealNews():
    filename1 = "bangla/Authentic-48K.csv"
    filename2 = "bangla/LabeledAuthentic-7K.csv"
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)

    data1['text'] = data1['headline'] + data1['content']
    data2['text'] = data2['headline'] + data2['content']

    x = data1['text'].tolist() + data2['text'].tolist()

    x = random.sample(x,k=2600)
    y = [1 for i in range(0, 2600)]
    print("CSV file loaded!!")
    return x, y


def testBanglaNumber(word):
    l = ['১','২','৩','৪','৫','৬','৭','৮','৯','০']
    for i in l:
        if i in word:
            return True

    return False


def cleanDataset(lines):
    cleaned = []
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation+"।")
    for i in lines:
        ##Clean Bangla
        ##print("first ", bengLine)
        # tokenize on white space
        bengLine = i.split()
        ##print("second ", bengLine)

        # remove punctuation from each token
        bengLine = [word.translate(table) for word in bengLine]
        ##print("third ", bengLine)

        # remove tokens with numbers in them
        bengLine = [word for word in bengLine if not testBanglaNumber(word)]

        # remove non-Bangla words
        bengLine = [word for word in bengLine if not word.isalpha()]

        #print("Bengali line: ", bengLine)
        # store as string
        cleaned.append(' '.join(bengLine))
    return cleaned





x1,y1 = getRealNews()
x2,y2 = getFakeNews()

x = x1+x2
y = y1+y2


c = list(zip(x,y))
random.shuffle(c)
x,y = zip(*c)
x = list(x)
y = np.array(y)

print(len(x))
print(y.shape)

x = cleanDataset(x)
print(x[0])


import pickle
f = open("banglaDataset","wb")
pickle.dump(x,f)
pickle.dump(y,f)

