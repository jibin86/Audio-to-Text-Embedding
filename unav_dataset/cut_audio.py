'''
오디오의 원하는 구간을 자르는 코드
'''

import torchaudio

def cut_wav_file():
    id = input("id: ")
    start_time = float(input("start_time: "))
    end_time = float(input("end_time: "))
    input_path = f"data/unav100/selected_audios/{id}.wav"
    output_path = f"data/unav100/selected_trimmed_audios/{id}.wav"
    
    # Load the WAV file
    audio, sr = torchaudio.load(input_path)

    # Calculate the start and end samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the desired portion of the audio
    extracted_audio = audio[:, start_sample:end_sample]

    # Save the extracted audio as a new WAV file
    torchaudio.save(output_path, extracted_audio, sr)

if __name__ == '__main__':
    cut_wav_file()
