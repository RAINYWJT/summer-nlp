from pywebio.input import file_upload
from pywebio.output import put_file
from pywebio.output import put_markdown, put_text, put_buttons
from pywebio.platform.flask import webio_view
from pywebio import start_server
from pywebio.input import select
from pywebio.output import *
from io import BytesIO
import pandas as pd
import tkinter.filedialog as filedialog
import pickle
from flask import Flask, render_template
from keras.models import load_model
from keras import backend as K
import numpy as np
import librosa
from python_speech_features import mfcc
import glob
import pyaudio
import pickle
from scipy import fftpack
import wave
from pydub import AudioSegment
from scipy.io import wavfile
import pywt
import soundfile as sf 
import noisereduce as nr
from pydub.utils import make_chunks


def wav_for_free(file_path):
    with open('dictionary.pkl', 'rb') as fr:
        [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

    mfcc_dim = 13
    model = load_model('NLP.h5')

    audio, sr = librosa.load(file_path)
    energy = librosa.feature.rms(y=audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    X_data = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    X_data = (X_data - mfcc_mean) / (mfcc_std + 1e-14)

    pred = model.predict(np.expand_dims(X_data, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()

    transcription = ''.join([id2char[i] for i in pred_ids if i != -1 and i in id2char])
    with open("transcription.txt", "w", encoding="utf-8") as file:
        file.write(transcription)

def wav_for_self(recording_seconds):
    def get_audio(filename,threshold=8000):
        CHUNK = 1024 
        FORMAT = pyaudio.paInt16  
        CHANNELS = 1  
        time = int(recording_seconds)
        RATE = 16000  
        RECORD_SECONDS = time  
        WAVE_OUTPUT_FILENAME = filename  
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        put_text("* 录音中...")
        frames = []
    
        if time > 0:
            # 根据指定录音时间循环读取音频数据并保存在frames列表中
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
        else:
            stopflag = 0
            stopflag2 = 0
            while True:
                data = stream.read(CHUNK)
                rt_data = np.frombuffer(data, np.dtype('<i2'))

                fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)
                fft_data = np.abs(fft_temp_data)[0:fft_temp_data.size // 2 + 1]

                # 判断音频能量是否超过阈值，用于检测声音和静音的转换
                if sum(fft_data) // len(fft_data) > threshold:
                    stopflag += 1
                else:
                    stopflag2 += 1
                oneSecond = RATE/CHUNK
            
                # 判断是否超过停止标志，用于自动结束录音
                if stopflag2 + stopflag > oneSecond:
                    if stopflag2 > oneSecond // 3 * 2:
                        break
                    else:
                        stopflag2 = 0
                        stopflag = 0
                frames.append(data)
        put_text("* 录音结束,请稍等")
        stream.stop_stream()
        stream.close()
        p.terminate()
    
        # 将录制的音频数据保存到wav文件中
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    get_audio("/home/rainy/python/NLP/speak/data_thchs30/test.wav")

    ############################################################################################################2.降噪

    data, sr = sf.read("/home/rainy/python/NLP/speak/data_thchs30/test.wav")
    noise_clip = data[0:10000]
    reduced_noise = nr.reduce_noise(y=data, y_noise=noise_clip, sr=sr)
    sf.write("processed.wav", reduced_noise, sr)
    # 使用 pyDub 读取处理后的音频文件
    audio = AudioSegment.from_file("processed.wav", format="wav")
    # 定义压缩的位速率（bps）
    bitrate = "96k"
    # 压缩音频并导出
    compressed_audio = audio.export("processed.wav", format="wav", bitrate=bitrate)

    ###################################################################################################3.语音处理
    wav_files = ['processed.wav']

    with open('dictionary.pkl', 'rb') as fr:
        [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

    mfcc_dim = 13
    model = load_model('NLP.h5')

    for wav_file in wav_files:
    
        audio, sr = librosa.load(wav_file)
        energy = librosa.feature.rms(y=audio)
        frames = np.nonzero(energy >= np.max(energy) / 5)
        indices = librosa.core.frames_to_samples(frames)[1]
        audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
        X_data = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
        X_data = (X_data - mfcc_mean) / (mfcc_std + 1e-14)

        pred = model.predict(np.expand_dims(X_data, axis=0))
        pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
        pred_ids = pred_ids.flatten().tolist()

        transcription = ''.join([id2char[i] for i in pred_ids if i != -1 and i in id2char])
    
        # 将transcription写入txt文件
        with open("transcription.txt", "w", encoding="utf-8") as file:
            file.write(transcription)

def main():
    put_markdown(r""" # <center> <font face="楷体">AC Team语音识别系统 </font> </center>
    """)
    select_res = select("请选择上传语音文件或自行录音:", ['语音文件', '自行录音'])
    if select_res == '语音文件':
        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        wav_for_free(file_path)
        put_text('转换完成！')
        with open("transcription.txt", "r", encoding="utf-8") as file:
            content = file.read()
        put_text(content)
    else:
        put_text('准备录音！')
        recording_seconds = select("请选择自行录音秒数:", ['1', '2','3', '4','5', '6','7', '8','9', '10','11', '12','13', '14','15', '16','17', '18','19', '20'])
        flag = True
        if flag:
            wav_for_self(recording_seconds)
            put_text('转换完成！')
            with open("transcription.txt", "r", encoding="utf-8") as file:
                content = file.read()
            put_text(content)

if __name__ == '__main__':
    main()