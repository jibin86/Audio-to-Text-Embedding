'''
오디오를 2초 단위로 자르고 train와 test로 나누어 저장하는 코드
'''

import os
import json
from pydub import AudioSegment
from tqdm import tqdm

# 오디오를 2초 단위로 자르고 저장하는 함수
def split_audio(json_file, audio_dir, train_output_dir, test_output_dir, segment_length=2000):
    with open(json_file, 'r') as file:
        data = json.load(file)

    for audio_id, audio_data in tqdm(data["database"].items(), desc="Processing Data"):
        audio_file = os.path.join(audio_dir, f"{audio_id}.wav")
        if os.path.exists(audio_file):
            audio = AudioSegment.from_file(audio_file)
            num_segments = len(audio) // segment_length

            # Split the audio into segments
            for i in range(num_segments):
                start_time = i * segment_length
                end_time = (i + 1) * segment_length
                segment = audio[start_time:end_time]

                output_dir = test_output_dir if audio_data["subset"] == "test" else train_output_dir
                output_file = os.path.join(output_dir, f"{audio_id}_segment_{i}.wav")
                segment.export(output_file, format="wav")



# 오디오 세그먼트를 분할하고 저장하는 함수 호출
annotations_json_file = "../unav_dataset/data/unav100/annotations/unav100_annotations.json"
audio_dir = "../unav_dataset/data/unav100/raw_audios"
train_output_dir = "../unav_dataset/data/unav100/audio_segments_2sec/train"
test_output_dir = "../unav_dataset/data/unav100/audio_segments_2sec/test"


split_audio(annotations_json_file, audio_dir, train_output_dir, test_output_dir)
