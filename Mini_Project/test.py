import keras.metrics
import numpy as np
import matplotlib.pyplot as plt
import mne
import tensorflow
from mne import Epochs, events_from_annotations, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import sys
import os
from EEGModels.EEGModels import EEGNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def subjectloader(range_start, range_end):
    #load subjectsto be used and split them
    subjects = [i for i in range(range_start, range_end)]
    trainsubjects, testsubjects = train_test_split(subjects, test_size=0.2)
    testsubjects, varsubjects = train_test_split(trainsubjects, test_size=0.5)

    return trainsubjects, testsubjects, varsubjects


def fileloader(trainsubjects,testsubjects,varsubjects,runs):
    #load files into raws
    path = "E:\\Documents\\Uni\\Master\\AdvancedML\\Mini_Project\\Data"
    test_names = eegbci.load_data(trainsubjects, runs)
    train_names = eegbci.load_data(testsubjects, runs)
    var_names = eegbci.load_data(varsubjects, runs)

    test_raw = concatenate_raws([read_raw_edf(f,preload=True) for f in test_names])
    test_raw.rename_channel(lambda x: x.strip('.'))
    train_raw = concatenate_raws([read_raw_edf(f, preload=True) for f in train_names])
    train_raw.rename_channel(lambda x: x.strip('.'))
    var_raw = concatenate_raws([read_raw_edf(f, preload=True) for f in var_names])
    var_raw.rename_channel(lambda x: x.strip('.'))

    return test_raw, train_raw, var_raw

def filefilter(train_raw, test_raw, var_raw):
    for entry in [train_raw, test_raw, var_raw]:
        eegbci.standardize(entry)
        entry.set_montage(mne.channels.make_standard_montage('standart_1020'))
        # Apply band-pass filter
        # remove bunch of frequencies that do not contain relevant information to the task
        entry.filter(7., 35., fir_design='firwin')

    return test_raw, train_raw, var_raw

def epochcreator(train_raw, test_raw, var_raw):
    # Select EEG channels and remove bad ones
    # we want Cz, C3,C4 for feet
    # we also want C3, Cz and C4 for hands as well as surronding parieto-occiptal areas P3,P4,O1,O2 (important for planing and direction)
    picks = pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       stim=False,
                       eog=False,
                       exclude='bads')
    tmin, tmax = 0.5,4.5
    epochlist = []
    labellist = []
    for entry in [train_raw, test_raw, var_raw]:
        events, _ = mne.events_from_annotations(entry,event_id=dict(T1=0,T2=1))
        epochs = Epochs(raw=entry,
                        events=events,
                        event_id=dict(hands=0, feet=1),
                        tmin=tmin,
                        tmax=tmax,
                        proj=True,
                        picks=picks,
                        baseline=None,
                        preload=True)
        epochlist.append(1e6 * epochs.get_data)
        label = epochs.events
        label2 = np.delete(label,[1],axis=1)
        labellist.append(label2)
    train_epoch = epochlist[0]
    test_epoch = epochlist[1]
    var_epoch = epochlist[2]
    test_label = labellist[0]
    train_label = labellist[1]
    var_label = labellist[2]
    train = (train_epoch,train_label)
    test = (test_epoch,test_label)
    var = (var_epoch,var_label)

    return train,test,var


#lots of things are taken from here https://github.com/JGalego/eeg-bci-tutorial/blob/master/eeg_bci.ipynb

if __name__ == '__main__':
    mne.set_log_level('WARNING')

    runs = [6,10,14]  # Motor imagery: hands vs feet
    #after a lot of manual testing S088 has something wrong, so it is excluded
    subjects = [i for i in range(1, 87)]
    path = "E:\\Documents\\Uni\\Master\\AdvancedML\\Mini_Project\\Data"

    # Load files
    raw_fnames = eegbci.load_data(subjects, runs)

    # Read all EDF files and combine them

    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    raw.rename_channels(lambda x: x.strip('.'))

    # Set Electrode Positions
    # This tells MNE where each electrode is on the scalp

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    #was originally done when I had it limited to 1 second so it fit the 128 samples
    #raw.resample(sfreq=127)

    # Apply band-pass filter
    #remove bunch of frequencies that do not contain relevant information to the task
    raw.filter(7., 35., fir_design='firwin')

    # Select EEG channels and remove bad ones
    picks = pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       stim=False,
                       eog=False,
                       exclude='bads')
    #kurz nach start erst anfangen weil am anfang oft nix ist
    tmin, tmax = 0.5, 3.5

    # Get events from an Annotations object.
    events, event_id = events_from_annotations(raw,event_id=dict(T1=0,T2=1))
    #every event is an annotaion which is explained here https://mne.tools/stable/documentation/glossary.html#term-events
    # Collum 1 tells us event onset + first_samp, first_samp  representing the number of time samples that passed between the onset of the hardware acquisition system and the time when data recording started
    # Column 2 tells us the signal value of the immediately preceeding sample reflects the fact that event arrays sometimes originate from analog voltage channels, it is usually 0 and can be ignored
    # Column 3 the annotation/event code, where T0 is rest, T1 is hands, T2 is feet
    print('Found {} events'.format(events.shape[0]))
    # print(events.shape)
    # print("first event",events[0])
    # print("second event",events[1])
    # print("events :",events)

    #Read epochs
    epochs = Epochs(raw = raw,
                    events = events,
                    event_id= dict(hands=0,feet=1),
                    tmin = tmin,
                    tmax = tmax,
                    proj=True,
                    picks=picks,
                    baseline=None,
                    preload=True)

    #not sure what exactly this does but it seems to help
    epochs_data = 1e6 * epochs.get_data()
    labels = epochs.events
    #deleting the 2nd column as it is useless, as explained above
    labels = np.delete(labels,[1],axis=1)
    handlabel = [x for x in labels if x[1]==0]
    feetlabel = [x for x in labels if x[1]==1]
    print("handlabel :",len(handlabel))
    print("feetlabel :",len(feetlabel))
    print("total label :",len(labels))

    #split into train and test set v1
    # testlen = int(len(epochs_data) * 0.8)
    # testlen2 = int(len(labels) * 0.8)
    # trainepochs, testepochs = np.split(epochs_data, indices_or_sections=[testlen],axis=0)
    # trainlabels, testlabels = np.split(labels,indices_or_sections=[testlen2],axis=0)
    # testepochs, validepochs = np.split(testepochs,indices_or_sections=2,axis=0)
    # testlabels, valilabels = np.split(testlabels,indices_or_sections=2,axis=0)
    #
    # print("trainhandlabel :", len(trainhandlabel))
    # print("trainfeetlabel :", len(trainfeetlabel))
    # print("testhandlabel :", len(testhandlabel))
    # print("testfeetlabel :", len(testfeetlabel))

    #split using train_test split using random state 42, as it is the answer to everything in the universe
    trainepochs, testepochs = train_test_split(epochs_data,test_size=0.2,random_state=42)
    trainlabels, testlabels = train_test_split(labels,test_size=0.2,random_state=42)
    trainhandlabel = [x for x in trainlabels if x[1] == 0]
    trainfeetlabel = [x for x in trainlabels if x[1] == 1]
    testhandlabel = [x for x in testlabels if x[1] == 0]
    testfeetlabel = [x for x in testlabels if x[1] == 1]
    valepochs, testepochs = train_test_split(testepochs,test_size=0.5,random_state=42)
    vallabels, testlabels = train_test_split(testlabels,test_size=0.5,random_state=42)
    print("trainhandlabel :", len(trainhandlabel))
    print("trainfeetlabel :", len(trainfeetlabel))
    print("testhandlabel :", len(testhandlabel))
    print("testfeetlabel :", len(testfeetlabel))



    # print("testepochs :", testepochs.shape)
    # print("trainepochs :", trainepochs.shape)
    # print("trainepochs :", trainepochs.shape)
    # print("testlabels :", testlabels)
    # print("label shape :", labels.shape)
    # print("epochs shape :", epochs_data.shape)


    #time to encode it to make sure there are no false relationships and normalize those bad boys
    #https: // www.geeksforgeeks.org / python / how - to - normalize - an - array - in -numpy - in -python /
    # https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
    s = StandardScaler()
    xtrain = s.fit_transform(trainepochs.reshape(-1,trainepochs.shape[-1])).reshape(trainepochs.shape)
    #xtrain = trainepochs
    xtest = s.fit_transform(testepochs.reshape(-1,testepochs.shape[-1])).reshape(testepochs.shape)
    #xtest = testepochs
    xval = s.fit_transform(valepochs.reshape(-1,valepochs.shape[-1])).reshape(valepochs.shape)
    #xval = valepochs
    #could not quite get this to work, would probably help to encode it like that
    # ytrain = to_categorical(trainlabels,num_classes=2)
    # ytest = to_categorical(testlabels,num_classes=2)
    # yval = to_categorical(testlabels,num_classes=2)
    ytrain = trainlabels
    ytest = testlabels
    yval = vallabels

    nb_classes = 2
    Channels = 64
    #samples/s is 160, multiply with time and add 1 for bias
    Samples = int(160*(tmax-tmin)+1)
    dropoutRate = 0.5
    Dimensions = 2
    F2 = 16
    norm_rate = 0.25
    #reccomended 1-32 so I choose in the middle
    batch_size = 16
    dropoutType = 'Dropout'
    epochs = 10
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-3)

    model = EEGNet(nb_classes=nb_classes, Chans=Channels, Samples=Samples, kernLength=int(Samples / 2),
                   dropoutRate=dropoutRate,dropoutType=dropoutType,F1=8,F2=8)
    #learning rate choosen based on googling what could be good
    #early stop
    early_stop = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
        #for some reason this is taken as unexpected argument when it should be valid https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        # start_from_epoch=10
    )
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        'accuracy',
        keras.metrics.FalseNegatives(),keras.metrics.FalsePositives(),
        keras.metrics.TrueNegatives(),keras.metrics.TruePositives()])

    fitted = model.fit(xtrain,ytrain, verbose=2, batch_size=batch_size, shuffle=True,epochs=epochs,validation_data=(xval,yval),callbacks=[early_stop])
    ypreds = model.predict(xtest)
    print(ypreds[0])
    loss, acc, falsenegs, truenegs, falspos, truepos = model.evaluate(xtest,ytest)
    print("Test False Positives:", falsenegs)
    print("Test True Positives:", truepos)
    print("Test False Negatives:", falsenegs)
    print("Test True Negatives:", truepos)
    print("Test Accuracy:", acc)
    print("Test Loss:", loss)

    #Code for visualizing learning progress from Arthurs Group

    f = plt.figure(figsize=(14, 4))

    ax = f.add_subplot(1, 2, 1)
    ax.plot(fitted.history['loss'], label='Train Loss')
    ax.plot(fitted.history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Evaluation of learning process')
    ax.legend()
    plt.savefig('learning1.png')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(fitted.history['accuracy'], label='Train Accuracy')
    ax.plot(fitted.history['val_accuracy'], label='Val Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model evaluation')
    ax.legend()
    plt.savefig("learning2.png")
    plt.show()
    plt.close()

    #clear plot so new one can be drawn
    plt.cla()
    plt.clf()


    #requires matploglib>3.10 see https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2411
    labels = ['TrueNegatives','FalseNegatives','TruePositives','FalsePositives']
    labels = np.asarray(labels).reshape(2,2)
    # fmt is thr string formatting
    data = [[int(truenegs), int(falspos)],[int(falsenegs), int(truepos)]]
    print(data)
    f2 = plt.figure(figsize=(8,8))
    sns.heatmap(data=data, fmt='d', cmap='Blues',annot=True)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion.png')
    plt.show()
    plt.close()