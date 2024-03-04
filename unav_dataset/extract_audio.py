'''
비디오에서 오디오를 추출하는 코드
'''

from moviepy.editor import *
import os

def extract_all(video_dir, audio_dir):
	"""
	Function that extract audio from video
	Assintotic: O(1)
	"""

	fail = []

	audio_format = 'wav'
	videos = os.listdir(video_dir)
	n = len(videos)
	cnt = 1
	for video_path in videos:
		try:
			video = VideoFileClip(video_dir + video_path)
			audio = video.audio
			audio.write_audiofile(audio_dir+video_path[:-4] + '.' + audio_format)
			print(f"#### No.{cnt}/{n}")
			cnt += 1
		except:
			print(f"fail to download :{video_path}")
			fail.append(video_path)
	print(fail)

def extract_one(video_path, audio_dir):
	"""
	Function that extract audio from video
	Assintotic: O(1)
	"""

	audio_format = 'wav'
	video = VideoFileClip(video_path)
	audio = video.audio
	audio.write_audiofile(audio_dir+video_path.split("/")[-1][:-4] + '.' + audio_format)
	print(f"done")


if __name__ == '__main__':

	video_dir = "data/unav100/raw_videos/"
	audio_dir = "data/unav100/raw_audios/"

	extract_all(video_dir, audio_dir)

	# video_path = "data/unav100/raw_videos/-0TTFAArJ9k.mp4"
	# extract_one(video_path, audio_dir)


	
