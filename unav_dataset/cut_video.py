'''
비디오의 원하는 구간을 자르는 코드
'''

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2

def cut_wav_file():
    id = input("id: ")
    start_time = float(input("start_time: "))
    end_time = float(input("end_time: "))
    video_path = f"data/unav100/selected_videos/{id}.mp4"
    output_path = f"data/unav100/selected_trimmed_videos/{id}.mp4"

    # 비디오 자르기
    ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)

    print(f"추출이 완료되었습니다. 저장된 파일: {output_path}")
    


if __name__ == '__main__':
    cut_wav_file()
