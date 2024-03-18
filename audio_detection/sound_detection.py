'''
오디오가 어떤 event를 포함하는지 감지하는 코드
'''

import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import torch

from models import *
from pytorch_utils import move_data_to_device
import config
import matplotlib.pyplot as plt
import uuid



class SoundDetection():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 14000
        self.model_type = 'PVT'
        self.checkpoint_path = '../pretrained_models/audio_detection.pth'
        self.classes_num = config.classes_num
        self.labels = config.labels
        self.frames_per_second = self.sample_rate // self.hop_size
        Model = eval(self.model_type)
        self.model = Model(sample_rate=self.sample_rate, window_size=self.window_size, 
            hop_size=self.hop_size, mel_bins=self.mel_bins, fmin=self.fmin, fmax=self.fmax, 
            classes_num=self.classes_num)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)


    def save_images(self, waveform, top_result_mat, top_k, sorted_indexes, audio_path):
        stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=self.window_size, 
            hop_length=self.hop_size, window='hann', center=True)
        frames_num = stft.shape[-1]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        im1 = axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')

        im2 = axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, self.frames_per_second)) # 0, 832, 100 => 100이 초 단위!
        axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / self.frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(np.array(self.labels)[sorted_indexes[0 : top_k]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')

        draw_mat = top_result_mat.copy()
        draw_mat[top_result_mat < 0.2] = 0
        im3 = axs[2].matshow(draw_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[2].xaxis.set_ticks(np.arange(0, frames_num, self.frames_per_second)) # 0, 832, 100 => 100이 초 단위!
        axs[2].xaxis.set_ticklabels(np.arange(0, frames_num / self.frames_per_second))
        axs[2].yaxis.set_ticks(np.arange(0, top_k))
        axs[2].yaxis.set_ticklabels(np.array(self.labels)[sorted_indexes[0 : top_k]])
        axs[2].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[2].set_xlabel('Seconds')
        axs[2].xaxis.set_ticks_position('bottom')

        fig.colorbar(im1)
        fig.colorbar(im2)
        fig.colorbar(im3)
        plt.tight_layout()
        image_filename = os.path.join('results',f"{audio_path.split('/')[-1][:-4]}_"+str(uuid.uuid4())[0:4] + ".png")
        plt.savefig(image_filename)
        print(f"saving {image_filename}")

        return None
    
    def find_segments(self, arr, threshold):
        segments = []
        start_idx = None
        for i, value in enumerate(arr):
            if value > threshold:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segments.append((start_idx, i - 1))
                    start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(arr) - 1))
        return np.array(segments)/100

    def sound_event_detection_top_k(self, audio_path):
        """Inference sound event detection result of an audio clip.
        """
        # print(self.device)
        
        # Load audio
        (waveform, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)

        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, self.device)

        # Forward
        with torch.no_grad():
            self.model.eval()
            batch_output_dict = self.model(waveform, None)
            # print("batch_output_dict",batch_output_dict['framewise_output'].size())
            # print("batch_output_dict",batch_output_dict['clipwise_output'].size())

        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""

        # print('Sound event detection result (time_steps x classes_num): {}'.format(
        #     framewise_output.shape))

        sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

        top_k = 10  # Show top results
        top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]  
        top_result_mat_sort = np.max(top_result_mat, axis=0)
        top_result_label_sort = np.array(self.labels)[sorted_indexes[0 : top_k]]
            
        # for i in range(top_k):
        #     print(f"{top_result_label_sort[i]}: {top_result_mat_sort[i]:.3f}")

        return top_result_mat_sort, top_result_label_sort
    
    def detect_sound_with_threshold(self, audio_path):
        """Inference sound event detection result of an audio clip.
        """
        # print(self.device)
        
        # Load audio
        (waveform, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        
        ### cut audio (생략 가능)
        start_time = 0
        end_time = start_time + 15
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        print(waveform.shape)
        waveform = waveform[start_sample:end_sample]
    

        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, self.device)

        # Forward
        with torch.no_grad():
            self.model.eval()
            batch_output_dict = self.model(waveform, None)

        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""

        threshold = 0.1
        selected_indexes = np.where(np.max(framewise_output, axis=0) > threshold)[0]
        top_k = len(selected_indexes)

        # threshold를 넘는 값만 선택하여 결과 행렬 생성
        top_result_mat = framewise_output[:, selected_indexes]

        top_result_mat_sort = np.max(top_result_mat, axis=0)
        top_result_label_sort = np.array(self.labels)[selected_indexes]

        segments_dic = {}

        ### Result Images
        self.save_images(waveform, top_result_mat, top_k, selected_indexes, audio_path)
            
        # for i in range(top_k):
        #     print(f"{top_result_label_sort[i]}: {top_result_mat_sort[i]:.3f}")

        return top_result_mat_sort, top_result_label_sort, segments_dic


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=str, required=True)
    args = parser.parse_args()

    sound = SoundDetection()

    # print("### Start Sound Event Detection Top K ###")
    # top_result_mat_sort, top_result_label_sort = sound.sound_event_detection_top_k(args.a)
    # for i in range(len(top_result_mat_sort)):
    #     print(f"{top_result_label_sort[i]}: {top_result_mat_sort[i]:.3f}")


    print("\n### Start Sound Event Detection with Segments ###")
    top_result_mat_sort, top_result_label_sort, segments_dic = sound.detect_sound_with_threshold(args.a)
    for i in range(len(top_result_mat_sort)):
        if i in segments_dic:
            print(f"{top_result_label_sort[i]}: {top_result_mat_sort[i]:.3f}, Segments: {segments_dic[i].tolist()}")
        else:
            print(f"{top_result_label_sort[i]}: {top_result_mat_sort[i]:.3f}")