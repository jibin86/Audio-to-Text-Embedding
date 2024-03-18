import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import argparse

# JSON 파일을 읽어오고 데이터를 처리하는 함수
def process_data(args):

    json_file = args.annotations_json_file
    audio_dir = args.audio_dir
    train_output_dir = args.train_output_dir
    test_output_dir = args.test_output_dir
    json_dir = args.json_dir

    with open(json_file, 'r') as file:
        data = json.load(file)
    
    results_test = {}
    results_train = {}

    for audio_id, audio_data in tqdm(data["database"].items(), desc="Processing Data"):
        audio_file = os.path.join(audio_dir, f"{audio_id}.wav")
        if os.path.exists(audio_file):
            sound = AudioSegment.from_file(audio_file)
            annotations = audio_data["annotations"]
            for idx, annotation in enumerate(annotations):
                start, end = annotation['segment']
                segment = sound[start * 1000:end * 1000]  # milliseconds로 변환
                prompt = annotation['label']
                output_dir = test_output_dir if audio_data["subset"] == "test" else train_output_dir

                # 2초보다 짧은 세그먼트는 건너뜁니다.
                if len(segment) < 2000:  # 2초를 밀리초로 변환
                    continue

                # 2초 단위로 segment를 나누기
                segment_duration = 2 * 1000  # 2초를 밀리초로 변환
                for i in range(0, len(segment) // segment_duration * segment_duration, segment_duration):
                    segment_chunk = segment[i:i+segment_duration]
                    filename = f"{audio_id}_{idx}_{int(i/1000)}.wav"
                    output_file = os.path.join(output_dir, filename)
                    segment_chunk.export(output_file, format="wav")

                    if audio_data["subset"] == "test":
                        results_test[filename] = prompt

                    else:
                        results_train[filename] = prompt
                    
    # 결과를 JSON 파일에 저장
    output_file = os.path.join(json_dir, f"unav_audio_results_test.json") # "audio_results_train.json", "audio_results_test.json"
    with open(output_file, 'w') as f:
        json.dump(results_test, f, indent=4)

    output_file = os.path.join(json_dir, f"unav_audio_results_train.json") # "audio_results_train.json", "audio_results_test.json"
    with open(output_file, 'w') as f:
        json.dump(results_train, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_json_file', type=str, default="../unav_dataset/data/unav100/annotations/unav100_annotations.json")
    parser.add_argument('--audio_dir', type=str, default="../unav_dataset/data/unav100/raw_audios")
    parser.add_argument('--train_output_dir', type=str, default="../unav_dataset/data/unav100/audio_segments/train")
    parser.add_argument('--test_output_dir', type=str, default="../unav_dataset/data/unav100/audio_segments/test")
    parser.add_argument('--json_dir', type=str, default="../text_prompt")

    args = parser.parse_args()
    print(args)

    process_data(args)