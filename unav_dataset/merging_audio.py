'''
오디오의 여러 특정 구간들을 합치는 코드
'''

import torchaudio
import torch

def merge_wav_file():
    id1 = input("id1: ")
    start_time1 = float(input("start_time: "))
    end_time1 = float(input("end_time: "))
    input_path1 = f"data/unav100/selected_audios/{id1}.wav"
    # Load the WAV file
    audio, sr = torchaudio.load(input_path1)

    # Calculate the start and end samples
    start_sample1 = int(start_time1 * sr)
    end_sample1 = int(end_time1 * sr)

    # Extract the desired portion of the audio
    extracted_audio1 = audio[:, start_sample1:end_sample1]

    id2 = input("id2: ")
    start_time2 = float(input("start_time: "))
    end_time2 = float(input("end_time: "))
    input_path2 = f"data/unav100/selected_audios/{id2}.wav"
    
    # Load the WAV file
    audio, sr = torchaudio.load(input_path2)

    # Calculate the start and end samples
    start_sample2 = int(start_time2 * sr)
    end_sample2 = int(end_time2 * sr)

    # Extract the desired portion of the audio
    extracted_audio2 = audio[:, start_sample2:end_sample2]

    extracted_audio = torch.cat((extracted_audio1, extracted_audio2), dim=1)

    # Save the extracted audio as a new WAV file
    output_path = f"data/unav100/selected_trimmed_audios/{id1+'_'+id2}.wav"
    torchaudio.save(output_path, extracted_audio, sr)
    print("done")

if __name__ == '__main__':
    merge_wav_file()
