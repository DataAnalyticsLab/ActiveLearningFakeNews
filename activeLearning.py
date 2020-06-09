import numpy as np

from keras import layers
from keras.models import Sequential
import keras
from keras_self_attention import SeqSelfAttention
from gensim.models import Word2Vec
import tensorflow as tf
from keras.optimizers import TFOptimizer
from keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from keras import backend as K
from HANLayer import ElmoEmbeddingLayer


np.random.seed(10)

def error_conf(error, n):
    term = 1.96 * np.sqrt((error * (1 - error)) / n)
    lb = error - term
    ub = error + term

    return (1-ub)*100,(1-lb)*100

def tokenizedSequences(x,y):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    #from keras.utils import to_categorical

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)

    word_index = tokenizer.word_index
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    data = pad_sequences(sequences,
                         maxlen=MAX_SEQUENCE_LENGTH,
                         padding='pre',
                         truncating='pre')

    print('Found %s unique tokens.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    print(len(data))
    print(len(data[0]))
    print(data[0])

    print("Tokenizing done!!")

    return data,tokenizer,num_words


def tf_idf(x,y):
    # data vectorizer
    data = [i.split(" ") for i in x]
    vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=(2,2), max_features=1000,
                                 max_df=0.5,
                                 min_df = 3)
    docarray = vectorizer.fit_transform(x).toarray()
    #docterm = pd.DataFrame(docarray, columns=vectorizer.get_feature_names())



    return docarray




def readGlove2VecText(filename):
    f = open(filename, "r", encoding='utf-8')
    x = f.readlines()

    wordVectors = {}
    j = 0

    for line in x:
        i = line.rstrip().split()
        word = i[0]
        vector = [float(x) for x in i[1:]]
        # print(word)
        # print(vector)
        wordVectors[word] = np.array(vector)
        # print(len(wordVectors))

    f.close()
    return wordVectors


def getEmbeddingLayerForGlove2Vec(tokenizer, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    glove2Vec = readGlove2VecText("glove.txt")
    embedding_matrix = np.random.normal(loc=0,scale=0.01,size=(len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    j = 1
    for i in tokenizer.word_index.keys():
        index = i

        if(index not in glove2Vec.keys()):
            print(j,index)
            j+=1
        else:
            getArray = glove2Vec[index]
            embedding_matrix[tokenizer.word_index[i]] = getArray

    embedding_layer = layers.Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                       input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    return embedding_layer


def trainTestSplit(x,y,split):
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=split,random_state=2)
    return x_train, x_val, y_train, y_val



def specialModel(num_words,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,TF_IDF_LENGTH):
    #Multi-layer perceptron
    mlp_input = layers.Input(shape=(TF_IDF_LENGTH,))
    mlp_dense_1 = layers.Dense(256, activation='relu')(mlp_input)
    mlp_dense_2 = layers.Dense(128, activation='relu')(mlp_dense_1)


    #sequential_input = layers.Input(shape=(1,),dtype="string")
    sequential_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))
    word_embedding = layers.Embedding(num_words, EMBEDDING_DIM, trainable=True)(sequential_input)
    #word_embedding = ElmoEmbeddingLayer()(sequential_input)
    #word_embedding = layers.Input(shape=(24,32,))

    #Convolutional Neural Network
    convolution1D_1 = layers.Conv1D(128, 5, activation='relu')(word_embedding)
    convolution1D_2 = layers.Conv1D(128, 5, activation='relu')(convolution1D_1)
    globalMaxPooling1D_1 = layers.GlobalMaxPooling1D()(convolution1D_2)

    #LSTM
    lstm_1 = layers.CuDNNLSTM(64, return_sequences=True)(word_embedding)
    lstm_2 = layers.CuDNNLSTM(64, return_sequences=False)(lstm_1)


    #Output layer
    concatenated_layer = layers.Concatenate(axis=-1)([mlp_dense_2,globalMaxPooling1D_1,lstm_2])
    output_layer = layers.Dense(1, activation='sigmoid')(concatenated_layer)


    model = Model(inputs=[mlp_input, sequential_input], outputs=output_layer)
    #model = Model(inputs=[mlp_input, word_embedding], outputs=output_layer)

    model.compile(loss='binary_crossentropy', optimizer="RMSProp", metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()
    return model


def calculateEntropy(i):
    return -i*np.log2(i)-(1-i)*np.log2(1-i)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def loadDataset(filename):
    import pickle
    f = open(filename,"rb")
    xTemp = pickle.load(f)
    yTemp = pickle.load(f)

    x = []
    y = []

    for i in range(0,len(xTemp)):
        if(xTemp[i]==""):
            continue
        x.append(xTemp[i])
        y.append(yTemp[i])

    y = np.array(y)

    return x,y


def elmoDataSet(x):
    text = [' '.join(t.split()[0:min(len(t),MAX_SEQUENCE_LENGTH)]) for t in x]
    text = np.array(text, dtype=object)[:, np.newaxis]
    print(len(text),len(text[0]))
    return text


def leastConfidence(i):
    ret = []

    for j in i:
        if j < 0.5:
            ret.append(j)
        else:
            ret.append(1 - j)

    ret = np.array(ret)
    return ret



x,y = loadDataset("twitter")
print(len(x))
print(len(y))


MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2
BUDGET = 99
take = 1000


ACTIVE_LEARNING = True


sequenceData,tokenizer,num_words = tokenizedSequences(x,y)
#sequenceData = elmoDataSet(x)
#sequenceData,dummy = loadDataset("liarLiarBert")
#sequenceData = np.array(sequenceData)
#sequenceData = np.array(sequenceData)
#sequenceData = sequenceData.reshape(len(x),24,32)
tfData = tf_idf(x,y)

TF_IDF_LENGTH = tfData.shape[1]



x_train, x_val, y_train, y_val = trainTestSplit(sequenceData,y,TEST_SPLIT)
x_train_tf, x_val_tf, d1, d2 = trainTestSplit(tfData,y,TEST_SPLIT)









model = specialModel(MAX_NUM_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,TF_IDF_LENGTH)

fraction = []
acc = []
s = len(tfData)





if(ACTIVE_LEARNING):
    #main_x, query_x, main_y, query_y = trainTestSplit(x_train, y_train, 0.99)
    #main_x_tf, query_x_tf, d1, d2 = trainTestSplit(x_train_tf, y_train, 0.99)
    main_x, query_x, main_y, query_y = x_train[0:1000],x_train[1000:],y_train[0:1000],y_train[1000:]
    main_x_tf, query_x_tf = x_train_tf[0:1000],x_train_tf[1000:]

    del x_train
    del y_train

    accuracy = 0
    batch_size = 8
    epochs = 1

    while (accuracy<BUDGET):
        history = model.fit([main_x_tf,main_x], main_y,batch_size=batch_size,epochs=epochs,verbose=2)#,validation_data=([x_val_tf,x_val],y_val))
        score = model.evaluate([x_val_tf,x_val], y_val, verbose=2)
        accuracy = score[1]*100

        fraction.append(len(main_x)/s)
        acc.append(accuracy)

        print(history.history)
        print(fraction)
        print(acc)
        print("Size of training data:",len(main_x))
        print('Test loss:', score[0])
        print('Test accuracy: {:.2f}%'.format(score[1] * 100))
        print("F1_score:{:.4f}".format(score[2]))
        print("Precision:{:.4f}".format(score[3]))
        print("Recall:{:.4f}".format(score[4]))
        print("95% confidence interval:", error_conf((1 - score[1]), len(main_x) + len(query_x) + len(x_val)))

        class_dist = np.squeeze(model.predict([query_x_tf,query_x]))
        entropy_dist = calculateEntropy(class_dist)
        #lc_dist = leastConfidence(class_dist)

        #print(class_dist.shape)
        #print(class_dist)
        #print(entropy_dist.shape)
        #print(entropy_dist)

        #Get the top K entropy elements
        indices = entropy_dist.argsort()[-take:][::-1]
        #indices = lc_dist.argsort()[-take:][::-1]

        #print(indices)
        #print(indices.shape)
        #print(query_x[indices].shape)

        main_x = np.concatenate((main_x, query_x[indices]))
        main_x_tf = np.concatenate((main_x_tf, query_x_tf[indices]))
        main_y = np.concatenate((main_y, query_y[indices]))


        query_x = np.delete(query_x,indices,0)
        query_x_tf = np.delete(query_x_tf, indices, 0)
        query_y = np.delete(query_y, indices, 0)

        print("Query complete.")

else:
    batch_size = 8
    epochs = 5
    history = model.fit([x_train_tf, x_train], y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1,validation_data=([x_val_tf,x_val],y_val))
    score = model.evaluate([x_val_tf, x_val], y_val, verbose=2)
    accuracy = score[1] * 100

    print(history.history)
    print('Test loss:', score[0])
    print('Test accuracy: {:.2f}%'.format(score[1] * 100))
    print("F1_score:{:.4f}".format(score[2]))
    print("Precision:{:.4f}".format(score[3]))
    print("Recall:{:.4f}".format(score[4]))
    print("95% confidence interval:", error_conf((1 - score[1]), len(x_train) + len(x_val)))









