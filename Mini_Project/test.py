import keras.metrics
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import Epochs, events_from_annotations, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import sys
import os
from EEGModels.EEGModels import EEGNet

if __name__ == '__main__':
    runs = [6,10,14]  # Motor imagery: hands vs feet
    #after a lot of manual testing S088 has something wrong so it is excluded
    subjects = [i for i in range(1, 87)]
    path = "E:\\Documents\\Uni\\Master\\AdvancedML\\Mini_Project\\Data"

    # Load files
    raw_fnames = eegbci.load_data(subjects, runs)

    # Read all EDF files and combine them

    raw = concatenate_raws([read_raw_edf(f, preload=True,verbose='Info') for f in raw_fnames],verbose='Info',on_mismatch='ignore')
    raw.rename_channels(lambda x: x.strip('.'))

    # Metadata for the recording is available as an info object
    #print(raw.info)

    # Set Electrode Positions
    # This tells MNE where each electrode is on the scalp

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.resample(sfreq=127)

    # Check sample rate
    #print('sample rate:', raw.info['sfreq'], 'Hz')

    # Apply band-pass filter
    raw.filter(7., 35., fir_design='firwin')

    # Notice that the lowpass and highpass values have changed
    #print(raw.info)


    # Select EEG channels and remove bad ones
    picks = pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       stim=False,
                       eog=False,
                       exclude='bads')
    tmin, tmax = 1., 2.

    # Get events from an Annotations object.
    events, event_id = events_from_annotations(raw,event_id=dict(T1=1,T2=2))
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
                    event_id= dict(hands=1,feet=2),
                    tmin = tmin,
                    tmax = tmax,
                    proj=True,
                    picks=picks,
                    baseline=None,
                    preload=True)

    # print(epochs.info)
    # print('events x channels x samples:', epochs._data.shape)

    epochs_data = 1e6 * epochs.get_data()
    # print(epochs.events)
    labels = epochs.events
    #deleting the 2nd column as it is useless, as explained above
    labels = np.delete(labels,[1],axis=1)

    #split into train and test set
    testlen = int(len(epochs_data) * 0.8)
    testlen2 = int(len(labels) * 0.8)
    trainepochs, testepochs = np.split(epochs_data, indices_or_sections=[testlen],axis=0)
    trainlabels, testlabels = np.split(labels,indices_or_sections=[testlen2],axis=0)
    # print("testepochs :", testepochs.shape)
    # print("trainepochs :", trainepochs.shape)
    # print("trainepochs :", trainepochs.shape)
    # print("testlabels :", testlabels)
    # print("label shape :", labels.shape)
    # print("epochs shape :", epochs_data.shape)



    nb_classes = 2
    Channels = 64
    Samples = 128
    dropoutRate = 0.5
    Dimensions = 2
    F2 = 16
    norm_rate = 0.25
    dropoutType = 'Dropout'
    model = EEGNet(nb_classes=nb_classes, Chans=Channels, Samples=Samples, kernLength=int(Samples / 2),
                   dropoutRate=dropoutRate,dropoutType=dropoutType)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',keras.metrics.FalseNegatives(),keras.metrics.FalsePositives(),
                                                                         keras.metrics.TrueNegatives(),keras.metrics.TruePositives()])
    fitted = model.fit(trainepochs,trainlabels, verbose=2, batch_size=128, shuffle=True,epochs=3)
    preds = model.predict(testepochs)
    loss, accuracy, fallsnegs, falspos, truepos, trueneg = model.evaluate(testepochs, testlabels, verbose=0)
    print('Test accuracy:', accuracy)
    print('Test loss:', loss)
    print('Test fallsnegs:', fallsnegs)
    print('Test falspos:', falspos)
    print('Test truepos:', truepos)
    print('Test trueneg:', trueneg)
