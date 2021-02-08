from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import mne
from eeg_learn_functions import *

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



# Define a flask app
app = Flask(__name__)


MODEL_PATH = 'models/deep_learning_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model.predict_classes()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def clean_EEG(raw):
    raw.set_eeg_reference(ref_channels=['M1', 'M2'])
#   Band Pass filtering
    raw.filter(0.5, 45, fir_design='firwin')
#   Resampling of data
    raw.resample(250, npad="auto")
#       Removal of bad channels
    bad_channels=['CB1', 'CB2', 'HEOG', 'VEOG', 'EKG','M1','M2']
    x=raw.ch_names
    channels_to_remove = []
    for i in x:
        for j in bad_channels:
            if i==j:
                channels_to_remove.append(i)
    raw.drop_channels(ch_names=channels_to_remove)
    raw_tmp = raw.copy()
    raw_tmp.filter(1, None)
#       Run ICA on the data
    ica = mne.preprocessing.ICA(method="fastica",random_state=1)
    ica.fit(raw_tmp)
    picks = len(raw.ch_names)
    ica.exclude = [0,1]
    raw_corrected = raw.copy()
    ica.apply(raw_corrected)
#       Removal of Artifacts
    raw.del_proj()
#       Saving the data to dataframes
    df = raw.to_data_frame()
    df=df.set_index('time')
    numberofelectrodes = len(df.columns)
    electrodes = list(df.columns)
    res = {}
    j=0
    for i in electrodes:
        res[i]=j
        j+=1
    df=df.rename(columns=res)
    return df



def get_fft(snippet):
    Fs = 500.0;  # sampling rate
    #Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet)/Fs
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,snippet_time,Ts) # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
#     print('Ts: ',Ts)
#     print(t)
#     print(y.shape)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    #Added in: (To remove bias.)
    #Y[0] = 0
    return frq,abs(Y)
#f,Y = get_fft(np.hanning(len(snippet))*snippet)

def make_steps(samples,frame_duration,overlap):
    '''
    in:
    samples - number of samples in the session
    frame_duration - frame duration in seconds
    overlap - float fraction of frame to overlap in range (0,1)

    out: list of tuple ranges
    '''
    #steps = np.arange(0,len(df),frame_length)
    Fs = 500
    i = 0
    intervals = []
    samples_per_frame = Fs * frame_duration
    while i+samples_per_frame <= samples:
        intervals.append((i,i+samples_per_frame))
        i = i + samples_per_frame - int(samples_per_frame*overlap)
    return intervals

def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 500.0
    frame_length = Fs*frame_duration
    frames = []
    steps = make_steps(len(df),frame_duration,overlap)
    for i,_ in enumerate(steps):
        frame = []
        if i == 0:
            continue
        else:
            for channel in df.columns:
                snippet = np.array(df.loc[steps[i][0]:steps[i][1],int(channel)])
                f,Y =  get_fft(snippet)
                theta, alpha, delta = theta_alpha_delta_averages(f,Y)
                frame.append([theta, alpha, delta])

        frames.append(frame)
    return np.array(frames)

def make_data_pipeline(df,image_size,frame_duration,overlap):
    '''
    IN:
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    overlap - float fraction of frame to overlap in range (0,1)

    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''
    ##################################
    ###Still need to do the overlap###!!!
    ##################################

    Fs = 500.0   #sampling rate
    frame_length = Fs * frame_duration


    X_0 = make_frames(df,frame_duration)

    X_1 = X_0.reshape(len(X_0),60*3)

    images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
    images = np.swapaxes(images, 1, 3)
    # print(len(images), ' frames generated with label ', labels[i], '.')
    # print('\n')
    # if i == 0:
    X = images
        # y = np.ones(len(images))*labels[0]
    # else:
    # X = np.concatenate((X,images),axis = 0)
        # y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)


    return X

def model_predict(file_path, model):
    # img = image.load_img(img_path, target_size=(224, 224))
    raw = mne.io.read_raw_eeglab(file_path,preload=True)
    # Preprocessing the raw_eeg_file
    # x = image.img_to_array(img)
    x = clean_EEG(raw)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    image_size = 28                                                                               # 1 = current_mdd
    frame_duration = 1.0                                                                          # 2 = past_mdd
    overlap = 0.5
    images = make_data_pipeline(x,image_size,frame_duration,overlap)
    x = images
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result

    return None


if __name__ == '__main__':
    app.run(debug=True)
