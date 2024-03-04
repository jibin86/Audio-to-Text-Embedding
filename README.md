# Audio-to-Text-Embedding
This is an implementation of a deep learning model for embedding 2 seconds audio signals into text representations. This repository contains code for training models that convert audio data into text embeddings.
<!-- <style>
red { color: red }
yellow { color: yellow }
</style> -->

## Usage
This code is based on [The Power of Sound(TPoS): Audio Reactive Video Generation with Stable Diffusion (ICCV 2023)](https://github.com/ku-vai/TPoS). Get the checkpoints for audio extracter.   

Pretrained weights from following link: [link](https://drive.google.com/drive/folders/11kDpSAp6wKyDU13rVT66dB0H2vJwXk5D?usp=drive_link). Locate downloaded weights in `pretrained_models`.


### Train Audio Encoder
 You can find `audio_encoder/train.py` to train the Audio Encoder. You need [UnAV-100](https://unav100.github.io/) datasets. 
    
You can use the following codes for training. 


```
cd audio_encoder
python train_audio_encoder.py
```
