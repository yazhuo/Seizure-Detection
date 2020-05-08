import keras
import numpy as np
from random import shuffle
import math
import pandas as pd 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import itertools

# keras packages
from keras.models import Sequential
from keras.layers import  Dense, Conv2D, Conv3D, Dropout, Flatten, BatchNormalization , MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# sklearn packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def extract_480features(filename, outfile):
    n = 13641
    n_channel = 16
    m = 30
    df_data = pd.read_csv(filename, index_col=0)
    w = open(outfile, "w")
    
    #annotate at the first line
    w.write(',filenames,seizure labels,early labels,')
    for i in range(n_channel):
        for j in range(m):
            if (i == n_channel-1) & (j == m-1):
                w.write('channel-' + str(i) + 'feature-' + str(j) + '\n')
            else:
                w.write('channel-' + str(i) + 'feature-' + str(j) + ',')

    #write the data with 30 features
    for i in range(n):
        w.write(str(i) + ',' + str(df_data.iloc[i,0]) + ',' + str(df_data.iloc[i,1]) + ',' + str(df_data.iloc[i,2]) + ',')
        for j in range(n_channel):
            for k in range(m):
                index = 3 + 60 * (j+1) + k
                if (j == n_channel - 1) & (k == m-1):
                    w.write(str(df_data.iloc[i, index]) + '\n')
                else:
                    w.write(str(df_data.iloc[i, index]) + ',')
                    

    w.close()
    
    test = pd.read_csv(outfile)
    print(test)


def get_test_data(filename):
    
    df_data = pd.read_csv(filename, index_col=0).drop(columns=['filenames', 'seizure labels', 'early labels'])
    df_label = pd.read_csv(filename, usecols=[2])
    np_data = np.array(df_data)
    np_label = np.array(df_label)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_data = scaler.fit_transform(np_data)

    return np_data, np_label


def get_dog_data(file1, file2):

    df_ictal = pd.read_csv(file1,index_col=0).drop(columns=['filenames', 'labels'])
    df_interictal = pd.read_csv(file2, index_col=0).drop(columns=['filenames','labels'])

    np_ictal = np.array(df_ictal)
    np_interictal = np.array(df_interictal)

    dog1_ictal = np_ictal[0:178]
    dog2_ictal = np_ictal[178:350]
    dog4_ictal = np_ictal[350:607]
    dog3_ictal = np_ictal[607:-1]

    dog1_interictal = np_interictal[0:418]
    dog2_interictal = np_interictal[418:1566]
    dog4_interictal = np_interictal[1566:4356]
    dog3_interictal = np_interictal[4356:-1]

    scaler = MinMaxScaler(feature_range=(0,1))
    
    dog1_data = np.concatenate((dog1_ictal, dog1_interictal), axis=0)
    dog1_label0 = [-1 for i in range(dog1_ictal.shape[0])]
    dog1_label1 = [1 for i in range(dog1_interictal.shape[0])]
    dog1_label = np.concatenate((dog1_label0, dog1_label1), axis=0)
    dog1 = scaler.fit_transform(dog1_data)

    dog2_data = np.concatenate((dog2_ictal, dog2_interictal), axis=0)
    dog2_label0 = [-1 for i in range(dog2_ictal.shape[0])]
    dog2_label1 = [1 for i in range(dog2_interictal.shape[0])]
    dog2_label = np.concatenate((dog2_label0, dog2_label1), axis=0)
    dog2 = scaler.fit_transform(dog2_data)

    dog3_data = np.concatenate((dog3_ictal, dog3_interictal), axis=0)
    dog3_label0 = [-1 for i in range(dog3_ictal.shape[0])]
    dog3_label1 = [1 for i in range(dog3_interictal.shape[0])]
    dog3_label = np.concatenate((dog3_label0, dog3_label1), axis=0)
    dog3 = scaler.fit_transform(dog3_data)

    dog4_data = np.concatenate((dog4_ictal, dog4_interictal), axis=0)
    dog4_label0 = [-1 for i in range(dog4_ictal.shape[0])]
    dog4_label1 = [1 for i in range(dog4_interictal.shape[0])]
    dog4_label = np.concatenate((dog4_label0, dog4_label1), axis=0)
    dog4 = scaler.fit_transform(dog4_data)

    dog234_ictal = np.concatenate((dog2_ictal, dog3_ictal, dog4_ictal), axis=0)
    dog234_interictal = np.concatenate((dog2_interictal, dog3_interictal, dog4_interictal), axis=0)
    dog234_data = np.concatenate((dog234_ictal, dog234_interictal), axis=0)
    dog234_label0 = [-1 for i in range(dog234_ictal.shape[0])]
    dog234_label1 = [1 for i in range(dog234_interictal.shape[0])]
    dog234_label = np.concatenate((dog234_label0, dog234_label1), axis=0)
    dog234 = scaler.fit_transform(dog234_data)

    '''
    train_data = np.concatenate((np_ictal, np_interictal), axis=0)
    num_ictal = np_ictal.shape[0]
    num_interictal = np_interictal.shape[0]
    n = train_data.shape[0]
    label0 = [-1 for i in range(num_ictal)]
    label1 = [1 for i in range(num_interictal)]
    np_label0 = np.array(label0)
    np_label1 = np.array(label1)
    label_data = np.concatenate((np_label0, np_label1), axis=0)
    print(train_data.shape)
    '''
    return dog1, dog1_label, dog2, dog2_label, dog3, dog3_label, dog4, dog4_label, dog234, dog234_label


def get_patient_data(file1, file2):
    df_ictal = pd.read_csv(file1, index_col=0).drop(columns=['filenames', 'labels'])
    df_interictal = pd.read_csv(file2, index_col=0).drop(columns=['filenames','labels'])

    ictal = np.array(df_ictal)
    interictal = np.array(df_interictal)

    num_ictal = ictal.shape[0]
    num_interictal = interictal.shape[0]
    print(num_ictal)
    print(num_interictal)
    
    ictal_n = ictal
    interictal_n = interictal
    add = abs(num_ictal-num_interictal)

    if num_ictal < num_interictal:
        ictal_n = np.concatenate((ictal, ictal[0:add,:]), axis=0)
    else:
        interictal_n = np.concatenate((interictal, interictal[0:add,:]), axis=0)

    np_data = np.concatenate((ictal_n, interictal_n), axis=0)
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(np_data)

    label0 = [-1 for i in range(ictal_n.shape[0])]
    label1 = [1 for i in range(interictal_n.shape[0])]
    print(ictal_n.shape[0])
    print(interictal_n.shape[0])

    label = np.concatenate((label0, label1), axis=0)
    
    return data, label


# helper function to print performance metrics
def evaluate(model, X_test, y_test, predictions):
    """
    Evaluate the trained model and give accuracy, precision, and recall score.
    @model: trained CNN
    @test_features: test set for prediction
    @test_labels: ground truth
    @return: accuracy score
    """

    #predictions = model.predict_proba(test_features)[:,1]
    #predictions = model.predict_classes(X_test)
    #trues = list(np.hstack(test_labels))
    #accuracy = 100 - accuracy_score(predictions, trues)
    accuracy = 100 - accuracy_score(y_test, predictions)
    precision = 100 - precision_score(y_test, predictions)
    recall = 100 - recall_score(y_test, predictions)
    f1 = 100 - f1_score(y_test, predictions)
    print('=== Model Performance ===')
    print('Accuracy: {:0.2f}%.'.format(accuracy))
    print('Precision: {:0.2f}%'.format(precision))
    print('Recall: {:0.2f}%'.format(recall))
    print('F1-score: {:0.2f}%'.format(f1))
    return accuracy


# calculate roc curves
def plot_roc_curve(y_test, probabilities, model_name):
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # keep probabilities for the positive outcome only
    lr_probs = probabilities

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic Regression: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title("ROC Curve: %s" % model_name)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def plot_confusion(cm, title='Confusion Matrix', cmap=plt.cm.Blues, labels=['-1','1']):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)

    fmt = '.2f'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def convert_dataTo3D(scaled_train_data, num_feature, num_channel):
    
    data = np.reshape(scaled_train_data, (-1, num_channel, num_feature, 1))
    return data


def convert_dataTo4D(scaled_train_data, num_feature):
    data = np.reshape(scaled_train_data, (-1,16, num_feature, 1, 1))
    return data


def create_2DModel(num_feature, num_channel):
    
    input_shape = (num_channel, num_feature, 1)
    model = Sequential()

    #C1
    model.add(Conv2D(16, 3,  padding='valid',activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    #C2
    
    model.add(Conv2D(32, 2, padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(BatchNormalization())
    
    #C3
    
    model.add(Conv2D(64, 2, padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.1), activity_regularizer=l2(0.01)))
    return model


def create_3DModel(num_feature):
    print('Conv3D')
    input_shape=(16, num_feature, 1, 1)
    model = Sequential()
    #C1
    model.add(Conv3D(8, (3, 3, 1),  padding='valid',activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same'))
    model.add(BatchNormalization())
    
    #C2
    model.add(Conv3D(16, (2, 2, 1), padding='valid',  activation='relu'))
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same'))
    model.add(BatchNormalization())
    
    #C3
    #model.add(Conv3D(32, (2, 2, 1), padding='valid', activation='relu'))
    #model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same'))
    #model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    opt_adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
    
    return model


def baseline(X_train, X_test, y_train, y_test):

    baseline = LogisticRegression(solver='saga')
    baseline.fit(X_train, y_train.ravel())
    baseline_pred = baseline.predict(X_test)

    print(baseline_pred)
    evaluate(baseline, X_test, y_test, baseline_pred)
    cnf_matrix = confusion_matrix(y_test, baseline_pred, normalize='true')
    print(cnf_matrix)
    plot_confusion(cnf_matrix)
    plt.show()
    

def correct_pred(cnn_pred):

    for i in range(len(cnn_pred)):
        if(cnn_pred[i] == 0):
            cnn_pred[i] = -1
    
    return cnn_pred


def cnn(X_train, X_test, y_train, y_test, num_feature, num_channel):
    
    X_train = convert_dataTo3D(X_train, num_feature, num_channel)
    X_test = convert_dataTo3D(X_test, num_feature, num_channel)

    model = create_2DModel(num_feature, num_channel)
    opt_adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=opt_adam, metrics=['accuracy']) 
    
    #model.summary()
    model.fit(X_train, to_categorical(y_train), epochs=10, shuffle=True, validation_split=0.20, batch_size=32)

    
    #cnn_prob = model.predict_proba(X_test)
    #cnn_pred = model.predict(X_test)
    cnn_pred = model.predict_classes(X_test)
    pred = correct_pred(cnn_pred)
    print(pred)
    print(y_test)
    plot_roc_curve(y_test, pred, 'CNN')
    evaluate(model, X_test, y_test, cnn_pred)
    cnf_matrix = confusion_matrix(y_test, cnn_pred, normalize='true')
    print(cnf_matrix)
    plot_confusion(cnf_matrix)
    plt.show()


def dog():
    # load data
    file1 = "data/Revised_train_Dog_ictal_data_v1.csv"
    file2 = 'data/Revised_train_Dog_interictal_data_v1.csv'
    testfile = 'data/Revised_test_Dog_ictal_data_v1.csv'

    num_feature = 50
    dog1, dog1_label, dog2, dog2_label, dog3, dog3_label, dog4, dog4_label, dog234, dog234_label = get_dog_data(file1, file2)
    scaled_test_data, test_label = get_test_data(testfile)
    dog1_test = scaled_test_data[0:3181]
    dog2_test = scaled_test_data[3181:6178]
    dog4_test = scaled_test_data[6178:9191]
    dog3_test = scaled_test_data[9191:-1]
    dog1_test_label = test_label[0:3181]
    dog2_test_label = test_label[3181:6178]
    dog4_test_label = test_label[6178:9191]
    dog3_test_label = test_label[9191:-1]
    
    #X_train, X_test, y_train, y_test = train_test_split(dog3, dog3_label, train_size = 0.9)
    
    
    X_train = dog234
    y_train = dog234_label
    X_test = dog1
    y_test = dog1_label
    

    print("X_train data shape: ", X_train.shape)
    print("y_train data shape: ", y_train.shape)
    print("X_test data shape: ", X_test.shape)
    print("y_test data shape: ", y_test.shape)

    #baseline(X_train, X_test, y_train, y_test)
    cnn(X_train, X_test, y_train, y_test, num_feature, 16)

    


def patient():
    # load data
    file1 = "data/Revised_train_Patient_8_ictal_data_v1.csv"
    file2 = 'data/Revised_train_Patient_8_interictal_data_v1.csv'
    channels = [0, 68, 16, 55, 72, 64, 30, 36, 16]
    num_feature = 50
    num_channel = channels[8]

    data, label = get_patient_data(file1, file2)

    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size = 0.8)
    print("X_train data shape: ", X_train.shape)
    print("y_train data shape: ", y_train.shape)
    print("X_test data shape: ", X_test.shape)
    print("y_test data shape: ", y_test.shape)

    #baseline(X_train, X_test, y_train, y_test)
    cnn(X_train, X_test, y_train, y_test, num_feature, num_channel)


if __name__ == "__main__":
    #dog()
    patient()

