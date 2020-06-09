from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.externals import joblib
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding
from keras.optimizers import RMSprop
from keras import layers
from keras import backend as K
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from keras import Model
from HANLayer import HierarchicalAttentionNetwork


np.random.seed(10)

def tf_idf(x,y):
    # data vectorizer
    data = [i.split(" ") for i in x]
    vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=(1,1), max_features=1000,
                                 max_df=0.5,
                                 min_df = 2)
    docarray = vectorizer.fit_transform(x).toarray()
    return docarray


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


def evaluate_model(predict_fun, X_train, y_train, X_test, y_test):
    '''
    evaluate the model, both training and testing errors are reported
    '''
    # training error
    y_predict_train = predict_fun(X_train)
    train_acc = accuracy_score(y_train,y_predict_train)

    # testing error
    y_predict_test = predict_fun(X_test)
    test_acc = accuracy_score(y_test,y_predict_test)

    #precision,recall,f1score
    precision,recall,f1Score,dud = precision_recall_fscore_support(y_test,y_predict_test,average='binary')

    return train_acc, test_acc,precision,recall,f1Score

def error_conf(error, n):
    term = 1.96 * np.sqrt((error * (1 - error)) / n)
    lb = error - term
    ub = error + term

    return (1-ub)*100,(1-lb)*100


def countVectorizer(x,y):
    # data vectorizer
    data = [i.split(" ") for i in x]
    vectorizer = CountVectorizer(analyzer = "word",
                                 min_df = 2)
    docarray = vectorizer.fit_transform(x).toarray()
    docterm = pd.DataFrame(docarray, columns=vectorizer.get_feature_names())
    return docarray,docterm


def loadDataset(filename):
    import pickle
    f = open(filename,"rb")
    x = pickle.load(f)
    y = pickle.load(f)

    return x,y


def trainTestSplit(x,y):
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2)
    return x_train, x_val, y_train, y_val




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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def naiveBayes(x_train,y_train,x_test,y_test):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    train_acc, test_acc,precision, recall,f1Score = evaluate_model(model.predict, x_train, y_train, x_test, y_test)
    print("Training Accuracy: {:.2f}%".format(train_acc * 100))
    print("Testing Accuracy: {:.2f}%".format(test_acc * 100))
    print("F1_score:{:.4f}".format(f1Score))
    print("Precision:{:.4f}".format(precision))
    print("Recall:{:.4f}".format(recall))
    print("95% confidence interval:", error_conf((1 - test_acc), len(x_train) + len(x_test)))




def SVM(x_train,y_train,x_test,y_test):
    svm = SVC(kernel='linear')
    model = svm.fit(x_train, y_train)
    train_acc, test_acc, precision, recall, f1Score = evaluate_model(model.predict, x_train, y_train, x_test, y_test)
    print("Training Accuracy: {:.2f}%".format(train_acc * 100))
    print("Testing Accuracy: {:.2f}%".format(test_acc * 100))
    print("F1_score:{:.4f}".format(f1Score))
    print("Precision:{:.4f}".format(precision))
    print("Recall:{:.4f}".format(recall))
    print("95% confidence interval:", error_conf((1 - test_acc), len(x_train) + len(x_test)))


def xgBoost(x_train,y_train,x_test,y_test):
    import xgboost
    model = xgboost.XGBClassifier()
    model.fit(x_train, y_train)
    train_acc, test_acc, precision, recall, f1Score = evaluate_model(model.predict, x_train, y_train, x_test, y_test)
    print("Training Accuracy: {:.2f}%".format(train_acc * 100))
    print("Testing Accuracy: {:.2f}%".format(test_acc * 100))
    print("F1_score:{:.4f}".format(f1Score))
    print("Precision:{:.4f}".format(precision))
    print("Recall:{:.4f}".format(recall))
    print("95% confidence interval:", error_conf((1 - test_acc), len(x_train) + len(x_test)))


def MLP(x_train,y_train,x_test,y_test):
    batch_size = 32
    num_classes = 1
    epochs = 5

    #model
    model = Sequential()
    model.add(Dense(300, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy',f1_m,precision_m, recall_m])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(history.history)
    print('Test loss:', score[0])
    print('Test accuracy: {:.2f}%'.format(score[1]*100))
    print("F1_score:{:.4f}".format(score[2]))
    print("Precision:{:.4f}".format(score[3]))
    print("Recall:{:.4f}".format(score[4]))
    print("95% confidence interval:", error_conf((1 - score[1]), len(x_train) + len(x_test)))

def CNN(x_train,y_train,x_test,y_test):
    batch_size = 32
    num_classes = 1
    epochs = 5

    model = Sequential()
    model.add(layers.Embedding(NUM_WORDS,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH,trainable=True))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(history.history)

    print('Test loss:', score[0])
    print('Test accuracy: {:.2f}%'.format(score[1] * 100))
    print("F1_score:{:.4f}".format(score[2]))
    print("Precision:{:.4f}".format(score[3]))
    print("Recall:{:.4f}".format(score[4]))
    print("95% confidence interval:", error_conf((1 - score[1]), len(x_train) + len(x_test)))

def LSTM(x_train,y_train,x_test,y_test):
    batch_size = 32
    num_classes = 1
    epochs = 5

    model = Sequential()
    model.add(layers.Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    model.add(layers.CuDNNLSTM(64))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(history.history)

    print('Test loss:', score[0])
    print('Test accuracy: {:.2f}%'.format(score[1] * 100))
    print("F1_score:{:.4f}".format(score[2]))
    print("Precision:{:.4f}".format(score[3]))
    print("Recall:{:.4f}".format(score[4]))
    print("95% confidence interval:", error_conf((1 - score[1]), len(x_train) + len(x_test)))

def dExplain(x_train,y_train,x_test,y_test):
    batch_size = 32
    num_classes = 1
    epochs = 5

    time_steps = 1
    n_inputs = MAX_SEQUENCE_LENGTH

    x_train = [i.reshape((-1, time_steps, n_inputs)) for i in x_train]
    x_train = np.array(x_train).reshape((-1, time_steps, n_inputs))

    x_test = [i.reshape((-1, time_steps, n_inputs)) for i in x_test]
    x_test = np.array(x_test).reshape((-1, time_steps, n_inputs))

    from keras.layers import Dense, Input
    from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed, Conv1D, Dropout
    from keras.models import Model

    embedding_layer = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True)
    sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    # lstm_word = Bidirectional(GRU(16, return_sequences=True))(embedded_sequences)
    lstm_word_conv = Conv1D(64, 5, activation='relu')(embedded_sequences)
    # dropout_1 = Dropout(0.2)(lstm_word)
    attn_word = HierarchicalAttentionNetwork(64)(lstm_word_conv)
    sentenceEncoder = Model(sentence_input, attn_word)

    review_input = Input(shape=(1, MAX_SEQUENCE_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentenceEncoder)(review_input)
    # lstm_sentence = Bidirectional(GRU(16, return_sequences=True))(review_encoder)
    lstm_sentence_conv = Conv1D(64, 1, activation='relu')(review_encoder)
    # dropout_2 = Dropout(0.2)(lstm_sentence)
    attn_sentence = HierarchicalAttentionNetwork(64)(lstm_sentence_conv)
    # dense_1 = Dense(100, activation='relu')(attn_sentence)
    # dropout_3 = Dropout(0.2)(dense_1)
    preds = Dense(1, activation='sigmoid')(attn_sentence)
    model = Model(review_input, preds)
    model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(history.history)

    print('Test loss:', score[0])
    print('Test accuracy: {:.2f}%'.format(score[1] * 100))
    print("F1_score:{:.4f}".format(score[2]))
    print("Precision:{:.4f}".format(score[3]))
    print("Recall:{:.4f}".format(score[4]))
    print("95% confidence interval:", error_conf((1 - score[1]), len(x_train) + len(x_test)))

def tCNN(x_train,y_train,x_test,y_test):
    batch_size = 32
    num_classes = 1
    epochs = 5

    model = Sequential()
    model.add(layers.Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    model.add(layers.AveragePooling1D())
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(history.history)

    print('Test loss:', score[0])
    print('Test accuracy: {:.2f}%'.format(score[1] * 100))
    print("F1_score:{:.4f}".format(score[2]))
    print("Precision:{:.4f}".format(score[3]))
    print("Recall:{:.4f}".format(score[4]))
    print("95% confidence interval:", error_conf((1 - score[1]), len(x_train) + len(x_test)))
    return 0






MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 25000
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2





x,y = loadDataset("banglaDataset")
#x = tf_idf(x,y)
x,tokenizer,NUM_WORDS = tokenizedSequences(x,y)

x_train, x_val, y_train, y_val = trainTestSplit(x,y)


print(x.shape)
print(y.shape)



#naiveBayes(x_train, y_train, x_val, y_val)
#SVM(x_train, y_train, x_val, y_val)
#MLP(x_train, y_train, x_val, y_val)
#xgBoost(x_train, y_train, x_val, y_val)
#CNN(x_train, y_train, x_val, y_val)
#LSTM(x_train, y_train, x_val, y_val)
#dExplain(x_train, y_train, x_val, y_val)
tCNN(x_train, y_train, x_val, y_val)

