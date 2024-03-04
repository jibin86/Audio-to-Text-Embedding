'''
오디오를 waveform 형태에서 mel-spectrogram 형태로 변환하는 코드
train과 test 모두 실행해야한다
'''


from glob import glob
import librosa
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
# import parmap
import argparse
import os

def func(idx):
    try:
        wav_name = audio_lists[idx]        
        
        name = wav_name.split("/")[-1].split(".")[0]
        path = f"unav_curation/{train_or_test}/{name}"

        if not os.path.exists(path):
            y, sr = librosa.load(wav_name, sr=44100)
            audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
            audio_inputs = np.array([audio_inputs])
            np.save(path, audio_inputs)
        # os.remove(wav_name)
    except:
        print(wav_name)
    finally:
        return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_test', type=str, default="train")

    args = parser.parse_args()
    print(args)

    train_or_test = args.train_or_test
    audio_lists = glob(f"unav_dataset/data/unav100/audio_segments_2sec/{train_or_test}/*.wav")
    data_length = len(audio_lists)
    print(data_length)
    
    for i in tqdm(range(data_length)):
        func(i)
    # result = parmap.map(func, range(data_length), pm_pbar=True, pm_processes=16)