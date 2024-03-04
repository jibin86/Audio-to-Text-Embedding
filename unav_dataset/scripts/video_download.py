import pandas as pd
import os

'''
00W3TGhk0I0_full_video.mp4.part
A2PmKTPNBxo_full_video.mp4.part
-DPpAs6owBI_full_video.mp4.part
-ixUrPNPogg_full_video.mp4.part

yt-dlp --ignore-config https://www.youtube.com/watch?v=00W3TGhk0I0 -o /home/jibin/storage1/sjb/UnAV/data/unav100/raw_videos/00W3TGhk0I0_full_video.mp4 -f b 
'''

def download_cut_video(root_path, info):
    link_prefix = "https://www.youtube.com/watch?v="
    link = link_prefix + info[0]
    filename_full_video = os.path.join(root_path, info[0]) + "_full_video.mp4"

    print( "download the whole video for: [%s] - [%s]" % (root_path, info[0]))
    # yt-dlp package need to be installed first
    command1 = 'yt-dlp --ignore-config ' 
    command1 += link + " "
    command1 += "-o " + filename_full_video + " "
    command1 += "-f b "
    print(command1)
    os.system(command1)
    if not os.path.exists(filename_full_video):
        print("can't download the video: " + filename_full_video) 
    else:
        cut_video(root_path, filename_full_video, info)
        return

def cut_video(root_path, video_path, info):
    global cnt, n
    filename = os.path.join(root_path, info[0]) + ".mp4"
    t_start, t_end = eval(info[1]), eval(info[2])
    t_dur = t_end - t_start
    print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
    # ffmpeg should be installed first
    command2 = 'ffmpeg '
    command2 += '-ss '
    command2 += str(t_start) + ' '
    command2 += '-i '
    command2 += video_path + ' '
    command2 += '-t '
    command2 += str(t_dur) + ' '
    command2 += '-vcodec libx264 '
    command2 += '-acodec aac -strict -2 '
    command2 += filename + ' '
    command2 += '-y '  # overwrite without asking
    command2 += '-loglevel -8 '  # print no log
    print(command2)
    os.system(command2)

    os.remove(video_path)
    print(f"No. {cnt} / {n}")
    cnt += 1
    print("finish cutting the video as: " + video_path)   
    return

if __name__ == '__main__':
    #provided file 'download_video.csv' that includes YouTube links of raw videos
    filename_source = "./download_video.csv"
    root_path = "../data/unav100/raw_videos"

    df = pd.read_csv(filename_source, header=None, sep='\t')
    samples = df[0]
    n = len(samples)
    cnt = 1
    for sample in samples:
        sample_list = sample.split(',')
        download_cut_video(root_path, sample_list)

