'''
오디오 인코더를 학습하는 코드
'''

import torch
import random
import argparse
import torch.optim as optim
from unav_datasets import UnavCurationDataset, UnavCurationTestDataset
# from unav_datasets_soundguided import UnavCurationDataset, UnavCurationTestDataset
from unav_model import Mapping_Model, Audio_Emb_Loss, FrozenOpenCLIPEmbedder, SoundCLIPLoss

import torch.nn.functional as F
import torch.nn as nn
import time
import os
from tqdm import tqdm
import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Audio Text Clip Implementation")

parser.add_argument("--epochs", default=100, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=1, type=int,
                help="batch size of training")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')        

args = parser.parse_args()

os.makedirs("../pretrained_models/",exist_ok=True)

if __name__ == "__main__":
    random.seed(42)
    unav_dataset = UnavCurationDataset()
    print(f"trainset length: {unav_dataset.__len__()}")
    unav_test_dataset = UnavCurationTestDataset()
    print(f"testset length: {unav_test_dataset.__len__()}")

    train_dataset=unav_dataset
    validation_dataset=unav_test_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    audioencoder = Audio_Emb_Loss()
    # audioencoder = SoundCLIPLoss()
    # audioencoder=nn.DataParallel(audioencoder).to(device)
    audioencoder = audioencoder.to(device)
    map_model = Mapping_Model()
    # map_model = nn.DataParallel(map_model).to(device)
    map_model = map_model.to(device)

    mse_loss = torch.nn.MSELoss()
    clip_model = FrozenOpenCLIPEmbedder()

    def freeze_models(models_to_freeze):
        for model in models_to_freeze:
            if model is not None: model.requires_grad_(False)
    
    def check_trainable(models):
        for model in models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"Parameter: {name}, Trainable: {param.requires_grad}")

    freeze_models([audioencoder, clip_model])
    check_trainable([map_model, audioencoder, clip_model])
    # print(map_model)

    # print("audioencoder", audioencoder)
    # print("clip_model",clip_model)
    # print("map_model",map_model)    

    # optimizer = optim.SGD(audioencoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    map_optimizer = optim.Adam(map_model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(map_optimizer, step_size=100, gamma=0.5)

    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="triangular")
    
    # wandb 연동
    wandb.init(project="audio_embedding")
    cfg = {
        "learning_rate": args.lr,
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "dropout": 0.2
    }
    wandb.config.update(cfg)

    # Measure time in PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)


    min_validation_loss_value = 50000
    for epoch in range(args.epochs):
        start = time.time()
        train_loss_value, validation_loss_value = 0, 0
        map_model.train()
        # audioencoder.train()
        pbar = tqdm(train_dataloader)
        for idx, (batch_audio, batch_text) in enumerate(pbar):
            # print(batch_audio.size()) # torch.Size([batch, 1, 128, 153])
            audio_embedding = audioencoder(batch_audio.clone().cuda())
            # audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
            # print("audio_embedding",audio_embedding.size()) # torch.Size([batch, 768])
            # audio_embedding_aug  = audioencoder(batch_audio_aug.cuda())

            clip_model_data = torch.cat([clip_model(text) for text in batch_text])

            # optimizer.zero_grad()
            map_optimizer.zero_grad()

            map_result = map_model(audio_embedding.unsqueeze(1))

            # map_norm_result = map_result / map_result.norm(dim=-1, keepdim=True)
            loss = mse_loss(map_result, clip_model_data[:,1:,:]) # map_result: torch.Size([batch, 76, 1024]), clip_model_data: torch.Size([batch, 76, 1024])

            loss.backward()
            
            # optimizer.step()
            map_optimizer.step()
            
            train_loss_value += loss.item()        

            pbar.set_postfix({"epoch": {epoch}, "total loss" : {loss.item()}})

        # scheduler.step()

        audioencoder.eval()
        map_model.eval()        
        
        print("Validation !")
        pbar2 = tqdm(validation_dataloader)
        validation_loss_value = 0
        for idx, (batch_audio, batch_text) in enumerate(pbar2):
            
            with torch.no_grad():
                
                audio_embedding = audioencoder(batch_audio.cuda())
                # audio_embedding_aug  = audioencoder(batch_audio_aug.cuda())


                clip_model_data = torch.cat([clip_model(text) for text in batch_text])

                map_optimizer.zero_grad()

                map_result = map_model(audio_embedding.clone().unsqueeze(1))
                # torch.autograd.set_detect_anomaly(True)

                loss = mse_loss(map_result, clip_model_data[:,1:])

            validation_loss_value += loss.item()

            pbar2.set_postfix({"epoch": {epoch}, "total loss" : {loss.item()}})
        
        print("Epoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))


        with open("../pretrained_models/unav_audio_mlp4.txt", "a") as f:
                    f.write("\n\nEpoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))
        
        if min_validation_loss_value > validation_loss_value:
            print(f"update min_validation_loss_value: {validation_loss_value} <= {min_validation_loss_value}")
            save_path = "../pretrained_models/unav_audio_mlp4" + str(epoch) +"_audio_emb_loss" + ".pth"
            torch.save(map_model.state_dict(), save_path)
            min_validation_loss_value = validation_loss_value

        wandb.log({'train_loss':train_loss_value / len(train_dataloader), 'val_loss':validation_loss_value / len(validation_dataloader)})
    wandb.finish()
                    

        # scheduler.step()
        