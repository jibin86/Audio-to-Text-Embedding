# Audio-to-Text-Embedding
This is an implementation of a deep learning model for embedding 2-second audio signals into text representations. This repository contains code for training models that convert audio data into text embeddings.

## Installation

Clone the repository:

```
git clone https://github.com/jibin86/Audio-to-Text-Embedding.git
cd Audio-to-Text-Embedding
```

Create and activate the conda environment:

```
conda env create --file env.yaml
conda activate audio_emb
```

Install required packages:

```
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
pip install transformers
pip install open_clip_torch
```

## Pretrained models
This code is built upon [The Power of Sound(TPoS)](https://github.com/ku-vai/TPoS) and [AudioGPT](https://github.com/AIGC-Audio/AudioGPT/tree/main). Obtain the checkpoints for the audio extractor and audio detector.

- Audio Extraction

    Pretrained weights can be found at the following link: [link](https://drive.google.com/drive/folders/11kDpSAp6wKyDU13rVT66dB0H2vJwXk5D?usp=drive_link). Once downloaded, place the weights in the `pretrained_models` directory.
    
- Audio Detection

    ```
    cd pretrained_models
    wget https://huggingface.co/Dongchao/pre_trained_model/resolve/main/audio_detection.pth
    ```

## Usage

### 1. Download UnAV Dataset
```
cd unav_dataset/scripts
python video_download.py
```

### 2. Extract Audio from Videos
```
cd unav_dataset
python extract_audio.py
```

### 3. Split Audio into 2-Second Segments and Save as "Train" and "Test"
```
cd audio_encoder
python unav_segment_2sec.py
```

### 4. Convert Audio from Waveform to Mel-Spectrogram Format
```
cd audio_encoder
python unav_curate.py --train_or_test train
python unav_curate.py --train_or_test test
```

### 5. Generate Text Prompts (Audio Detection)
```
cd audio_detection
python make_prompt.py --audio_dir "../unav_dataset/data/unav100/audio_segments_2sec/train" --json_dir "../audio_encoder/text_prompt"
python make_prompt.py --audio_dir "../unav_dataset/data/unav100/audio_segments_2sec/test" --json_dir "../audio_encoder/text_prompt"
```

### 6. Train Audio Encoder
```
cd audio_encoder
python unav_train_audio_encoder_tpos.py
```

