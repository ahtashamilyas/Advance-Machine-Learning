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
    runs = [6, 10, 14]  # Motor imagery: hands vs feet
    subjects = [i for i in range(1, 40)]
    path = "E:\\Documents\\Uni\\Master\\AdvancedML\\Mini_Project\\Data"

    # Load files
    raw_fnames = eegbci.load_data(subjects, runs)

    # Read all EDF files and combine them

    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    raw.rename_channels(lambda x: x.strip('.'))

    # Metadata for the recording is available as an info object
    print(raw.info)

    # Set Electrode Positions
    # This tells MNE where each electrode is on the scalp

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.resample(sfreq=127)

    # Check sample rate
    print('sample rate:', raw.info['sfreq'], 'Hz')

    # Apply band-pass filter
    raw.filter(7., 35., fir_design='firwin')

    # Notice that the lowpass and highpass values have changed
    print(raw.info)


    # Select EEG channels
    picks = pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       stim=False,
                       eog=False,
                       exclude='bads')
    picks = picks
    tmin, tmax = 1., 2.
    event_id = dict(hands=2, feet=3)

    # Get events from an Annotations object.
    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    print('Found {} events'.format(events.shape[0]))

    # Read epochs
    epochs = Epochs(raw,
                    events,
                    event_id,
                    int(tmin),
                    int(tmax),
                    proj=True,
                    picks=picks,
                    baseline=None,
                    preload=True)

    print(epochs.info)
    print('events x channels x samples:', epochs._data.shape)

    epochs_data = 1e6 * epochs.get_data()
    #remove the 0 in the middle collum to make it shape events x samples since it starts as events x 0 x samples
    labels = epochs.events[:,:-1]
    print(labels.shape)
    print(labels)
    print(type(labels))


    print("test")
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    fitted = model.fit(epochs_data,labels, verbose=1, batch_size=128, shuffle=True)
    preds = model.predict(epochs_data)
    print(preds)
    print("test2")