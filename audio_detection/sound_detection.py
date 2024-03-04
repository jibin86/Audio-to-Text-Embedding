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
        self.checkpoint_path = 'pretrained_models/audio_detection.pth'
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

    def sound_event_detection(self, audio_path):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    args = parser.parse_args()

    sound = SoundDetection()
    top_result_mat_sort, top_result_label_sort = sound.sound_event_detection(args.audio_path)

    for i in range(len(top_result_mat_sort)):
        print(f"{top_result_label_sort[i]}: {top_result_mat_sort[i]:.3f}")
