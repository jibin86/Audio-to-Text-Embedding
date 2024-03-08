""" MLP를 거쳐 mapping 된 text embedding을 input으로 사용하여 diffusion model output 확인해보기 
https://huggingface.co/stabilityai/stable-diffusion-2-1-base """

import torch
import numpy as np
import librosa
import sys
import os
import glob
from tqdm import tqdm
from pydub import AudioSegment
from audio_encoder.unav_model import Mapping_Model, Audio_Emb_Loss, FrozenOpenCLIPEmbedder, copyStateDict

# 오디오를 2초 단위로 자르고 저장하는 함수
def split_audio(audio_file, segment_length=2000, output_dir="outputs"):
    audio = AudioSegment.from_file(audio_file)
    num_segments = len(audio) // segment_length

    # Split the audio into segments
    segment_list = []
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segment = audio[start_time:end_time]
        output_file = os.path.join(output_dir, f"{audio_file.split('/')[-1][:-4]}_segment_{i}.wav")
        segment.export(output_file, format="wav")
        segment_list.append(output_file)
    audio_path = os.path.join(output_dir, f'{audio_file.split("/")[-1]}')
    audio.export(audio_path, format="wav")

    return segment_list

def curate(segment):   
    y, sr = librosa.load(segment, sr=44100)
    audio_input = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    audio_input = librosa.power_to_db(audio_input, ref=np.max) / 80.0 + 1
    audio_input = np.array([audio_input])

    return audio_input

def encode_audio(audio_npy):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    audioencoder = Audio_Emb_Loss('pretrained_models/audio_encoder_23.pth')
    audioencoder = audioencoder.to(device)
    audioencoder.eval()

    map_model = Mapping_Model()
    map_model.load_state_dict(torch.load('pretrained_models/map_model_0_audio_emb_loss.pth'))
    map_model = map_model.to(device)
    map_model.eval()

    # audio_lists = glob.glob('audio_encoder/unav_curation/test/*.npy')
    width_resolution = 153

    with torch.no_grad():  # 학습 x

        audio_seg = audio_npy[:,:,:width_resolution]
        audio_seg = torch.from_numpy(audio_seg).float()
        audio_seg = audio_seg.to(device)
    
        # audio encoder 통과
        audio_embedding = audioencoder(audio_seg) # audio_embedding: [1, 768]
        map_result = map_model(audio_embedding.clone().unsqueeze(1)) # map_result: [1, 76, 1024]
        embedding = map_result.cpu().numpy()

    return embedding



# # embedding 저장
# all_mlp_embeddings_array = np.array(embeddings)
# embeddings_tensor = torch.from_numpy(all_mlp_embeddings_array).float() # embedding -> tensor로 변환
# embeddings_tensor = embeddings_tensor.to(device)
# print(embeddings_tensor.shape)

# # disk에 저장
# embeddings_path = 'audio_encoder/results/embeddings.npy'
# np.save(embeddings_path, all_mlp_embeddings_array)
