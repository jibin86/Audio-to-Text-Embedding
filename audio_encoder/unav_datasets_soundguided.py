import os
import json
import torch
import librosa
from glob import glob
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from textaugment import EDA
import nltk
import time

nltk.download('omw-1.4')
nltk.download("stopwords")
nltk.download("wordnet")

class UnavCurationDataset(Dataset):
    def __init__(self):
        self.audio_lists = glob("./unav_curation/train/*.npy") # Put postprocessed audio files here
        self.time_length = 864
        self.n_mels = 128 # frequency축으로 mel-spectrogram을 128개로 쪼갬
        self.num_frames = 5 # 쪼갤 개수
        self.text_aug = EDA()
        self.width_resolution = 512 # 시간축으로 mel-spectrogram을 768개로 쪼갬

        # self.audio_lists = glob("./unav_curation/train/*.npy") # Put postprocessed audio files here
        self.audio_dir = "./unav_curation/train"
        self.json_file = "../text_prompt/audio_results_train2.json"
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

    def __getitem__(self, idx):
        audio_info = self.audio_lists[idx]
            
        audio_inputs = np.load(audio_info[0], allow_pickle=True) # __uEjp7_UDw_playing piano_0_49.90164.npy
        
        text_prompt = audio_info[1]
        # print("origin text_prompt: ",text_prompt)
        c, h, w = audio_inputs.shape # h는 frequency axis, w는 time axis
        # audio_inputs.shape (1, 128, xxx)

        ''' 1. 오디오를 특정 크기 time_length = 864로 만들기'''
        if w >= self.time_length: # 크면 자르기
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length: # 작으면 0으로 패딩하기
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero
        # audio_inputs.shape (1, 128, 864)

        ''' 2. audio 512로 리사이즈'''
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))

        # print(audio_inputs.shape) # (512,128)
            
        ''' 3. audio 증강'''
        audio_aug = self.spec_augment(audio_inputs)
        audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
        audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)
            
        audio_inputs = torch.from_numpy(audio_inputs).float()
        audio_aug = torch.from_numpy(audio_aug).float()


        # print(audio_inputs.shape) # torch.Size([1, 128, 512])
        # print(audio_aug.shape) # torch.Size([1, 128, 512])

        # ''' 3. text 증강 필요없으면 세 줄 주석처리하기'''
        # text_prompt = self.text_aug.synonym_replacement(text_prompt)
        # text_prompt = self.text_aug.random_swap(text_prompt)
        # text_prompt = self.text_aug.random_insertion(text_prompt)

        return audio_inputs, audio_aug, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
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
        self.audio_lists = glob("./unav_curation/test/*.npy") # Put postprocessed audio files here
        self.time_length = 864
        self.n_mels = 128 # mel-spectrogram의 frequency축
        self.num_frames = 5
        self.text_aug = EDA()
        self.width_resolution = 512 # mel-spectrogram의 시간축

        # self.audio_lists = glob("./unav_curation/test/*.npy") # Put postprocessed audio files here
        self.audio_dir = "./unav_curation/test"
        self.json_file = "../text_prompt/audio_results_test2.json"
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

    def __getitem__(self, idx):
        audio_info = self.audio_lists[idx]
            
        audio_inputs = np.load(audio_info[0], allow_pickle=True) # __uEjp7_UDw_playing piano_0_49.90164.npy
        
        text_prompt = audio_info[1]
        # print("origin text_prompt: ",text_prompt)
        c, h, w = audio_inputs.shape # h는 frequency axis, w는 time axis
        # audio_inputs.shape (1, 128, xxx)

        ''' 1. 오디오를 특정 크기 time_length = 864로 만들기'''
        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero
       
        ''' 2. audio 512로 리사이즈'''
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
            
        ''' 3. audio 증강'''
        audio_aug = self.spec_augment(audio_inputs)
        audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
        audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)
            
        audio_inputs = torch.from_numpy(audio_inputs).float()
        audio_aug = torch.from_numpy(audio_aug).float()

        # ''' 3. text 증강 필요없으면 세 줄 주석처리하기'''
        # text_prompt = self.text_aug.synonym_replacement(text_prompt)
        # text_prompt = self.text_aug.random_swap(text_prompt)
        # text_prompt = self.text_aug.random_insertion(text_prompt)

            
        return audio_inputs, audio_aug, text_prompt
            

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
    start = time.time()
    # print(datasets[3535]) # datasets[학습데이터개수][idx], idx:0 => audio_per_frame, 1 => audio_per_frame_aug, 2 => text_prompt
    print(len(datasets)) # 17465
    ran_idx = random.randint(0,100)
    print(f"datasets[{ran_idx}][1].size()",datasets[ran_idx][1].size()) # torch.Size([1, 128, 512])
    print(f"datasets[{ran_idx}][2]",datasets[ran_idx][2]) # skateboarding
    ran_idx = random.randint(0,100)
    print(f"datasets[{ran_idx}][2]",datasets[ran_idx][2]) # skateboarding

    # print(time.time() - start)

    # # 데이터로더 생성
    # data_loader = DataLoader(datasets, batch_size=1, shuffle=True)

    # # 데이터 로더를 통해 데이터 접근 및 확인
    # for audio, prompt in data_loader:
    #     print("Prompt:", prompt)
    #     print("Audio Data:", audio)
    #     print("Audio Data Shape:", audio.shape)
    #     print()