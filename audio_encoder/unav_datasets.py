'''
데이터셋을 2초 segment로 나눈 데이터를 처리하는 코드
Sound Event Dection 코드를 통해 얻은 레이블을 정답 레이블로 처리
'''


import os
import json
import torch
import librosa
from glob import glob
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from textaugment import EDA
import nltk
import time

nltk.download("stopwords")
nltk.download("wordnet")

class UnavCurationDataset(Dataset):
    def __init__(self):           
        self.time_length = 864
        self.n_mels = 128 # frequency축으로 mel-spectrogram을 128개로 쪼갬
        self.num_frames = 5 # 추가됨
        self.width_resolution = 768 // self.num_frames # 시간축으로 mel-spectrogram을 768 // self.num_frames (=153)개로 쪼갬
        self.frame_per_audio = self.time_length // self.num_frames

        # self.audio_lists = glob("./unav_curation/train/*.npy") # Put postprocessed audio files here
        self.audio_dir = "./unav_curation2/train"
        self.json_file = "../text_prompt/unav_train.json"
        self.audio_lists = []
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        cnt_none = 0
        for audio_file, audio_data in tqdm(data.items(), desc="Processing Data"):
            if not audio_data: # 텍스트 프롬프트가 없으면 제외하기
                cnt_none += 1
            else:
                audio_path = os.path.join(self.audio_dir, audio_file[:-4]+".npy")
                self.audio_lists.append([audio_path, audio_data])
        print(f"{cnt_none} audios have no prompt.")
        self.audio_lists = self.audio_lists[:20]

    def __getitem__(self, idx):
        audio_info = self.audio_lists[idx]
            
        audio_inputs = np.load(audio_info[0], allow_pickle=True) # __uEjp7_UDw_playing piano_0_49.90164.npy
        # print("audio_inputs.shape",audio_inputs.shape) # (1, 128, 173)
        text_prompt = audio_info[1]
        # print(text_prompt)
        
        audio_seg = audio_inputs[:,:,:self.width_resolution]
        # # print(audio_seg.shape) # (1, 128, 153)
        # audio_seg = audio_seg[0,:self.n_mels,:self.width_resolution] # n.mels => 128
        # # print(audio_seg.shape) # (128, 153)

        # audio_aug = self.spec_augment(audio_seg)
        # # print(audio_aug.shape) # (128, 153)

        # audio_seg = audio_seg.reshape(-1, self.n_mels, self.width_resolution)
        # audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)
            
        audio_seg = torch.from_numpy(audio_seg).float()
        # audio_aug = torch.from_numpy(audio_aug).float()

        # print(audio_seg.shape)  # torch.Size([1, 128, 153])

        return audio_seg, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape # 128,768
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_lists)


class UnavCurationTestDataset(Dataset):
    def __init__(self):
        self.time_length = 864
        self.n_mels = 128 # mel-spectrogram의 frequency축
        self.num_frames = 5
        self.width_resolution = 768 // self.num_frames # mel-spectrogram의 시간축
        self.frame_per_audio = self.time_length // self.num_frames

        # self.audio_lists = glob("./unav_curation/test/*.npy") # Put postprocessed audio files here
        self.audio_dir = "./unav_curation2/test"
        self.json_file = "../text_prompt/unav_test.json"
        self.audio_lists = []
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        cnt_none = 0
        for audio_file, audio_data in tqdm(data.items(), desc="Processing Data"):
            if not audio_data: # 텍스트 프롬프트가 없으면 제외하기
                cnt_none += 1
            else:
                audio_path = os.path.join(self.audio_dir, audio_file[:-4]+".npy")
                self.audio_lists.append([audio_path, audio_data])
        print(f"{cnt_none} audios have no prompt.")
        self.audio_lists = self.audio_lists[:20]


    def __getitem__(self, idx):
        audio_info = self.audio_lists[idx]
            
        audio_inputs = np.load(audio_info[0], allow_pickle=True) # __uEjp7_UDw_playing piano_0_49.90164.npy
        # print(audio_inputs.shape) # (1, 128, 173)
        text_prompt = audio_info[1]

        audio_seg = audio_inputs[:,:,:self.width_resolution]
        # audio_seg = audio_seg[0,:self.n_mels,:self.width_resolution] # n.mels => 128
        # audio_aug = self.spec_augment(audio_seg)
        
        # audio_seg = audio_seg.reshape(-1, self.n_mels, self.width_resolution)
        # audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)

        audio_seg = torch.from_numpy(audio_seg).float()
        # audio_aug = torch.from_numpy(audio_aug).float()
        
        return audio_seg, text_prompt
    

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape # 128,768
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_lists)

    
if __name__ == "__main__":
    
    datasets = UnavCurationDataset()
    test_datasets = UnavCurationTestDataset()
    start = time.time()
    ''' datasets[학습데이터개수][idx]
        idx:0 => audio_per_frame, 
        idx:1 => audio_per_frame_aug, 
        idx:2 => text_prompt'''
    
    print(len(datasets)) # 153386
    ran_idx = random.randint(0,100)
    print(f"datasets[{ran_idx}][0].size()",datasets[ran_idx][0].size()) # torch.Size([1, 128, 153])
    print(f"datasets[{ran_idx}][2]",datasets[ran_idx][2])

    print(f"test_datasets[{ran_idx}][1].size()",test_datasets[ran_idx][1].size()) # torch.Size([1, 128, 153])
    print(f"test_datasets[{ran_idx}][2]",test_datasets[ran_idx][2])
