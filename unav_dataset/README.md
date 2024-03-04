<pre>
unav_dataset
│── configs
│   └── avel_unav100.yaml
│── data
│   └── unav100
│       ├── annotations
│       │   └── unav100_annotations.json
│       ├── audio_segments_2sec
│       │   │── train
│       │   │   │── 0012y1s1bJI_segment_0.wav
│       │   │   │── 0012y1s1bJI_segment_1.wav
│       │   │   └── ...
│       │   └── test
│       │       │── 00dadPtnWZI_segment_0.wav
│       │       │── 00dadPtnWZI_segment_1.wav
│       │       └── ...
│       ├── raw_audios
│       │   │── 0012y1s1bJI.wav
│       │   │── 007P6bFgRCU.wav
│       │   └── ...
│       └── raw_videos
│           │── 0012y1s1bJI.mp4
│           │── 007P6bFgRCU.mp4
│           └── ...
│── scripts
│   │── download_video.csv
│   └── video_download.py
│── extract_audio.py
│── cut_audio.py
│── cut_video.py
│── merging_audio.py
└── README.md
</pre>