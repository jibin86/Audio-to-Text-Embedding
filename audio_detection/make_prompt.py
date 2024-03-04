'''2초 오디오에 대해 audio detection을 진행하여 어떤 event가 포함되는지 확인하고, 이를 json에 저장하는 코드'''

import os
import json
import argparse
import sound_detection
from tqdm import tqdm

def process_audio_directory(args):
    sound = sound_detection.SoundDetection()
    results = {}
    audio_files = sorted([filename for filename in os.listdir(args.audio_dir) if filename.endswith(".wav")])

    try:

        for filename in tqdm(audio_files):
            audio_path = os.path.join(args.audio_dir, filename)
            top_result_mat_sort, top_result_label_sort = sound.sound_event_detection(audio_path)
            
            result_text = ""
            for i in range(len(top_result_mat_sort)):
                if top_result_mat_sort[i] > 0.5:
                    result_text += top_result_label_sort[i] + ", "

            # 마지막 콤마 제거
            result_text = result_text[:-2]  

            # 결과를 딕셔너리에 저장
            results[filename] = result_text
    finally:
        # 결과를 JSON 파일에 저장
        output_file = os.path.join(args.json_dir, f"audio_results_{args.audio_dir.split('/')[-1]}.json") # "audio_results_train.json", "audio_results_test.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, default="../unav_dataset/data/unav100/audio_segments_2sec/train")
    parser.add_argument('--json_dir', type=str, default="../audio_encoder/text_prompt")

    args = parser.parse_args()
    print(args)

    process_audio_directory(args)
