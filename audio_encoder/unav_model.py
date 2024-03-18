'''
모델 클래스를 포함하는 코드
'''


import torch
import torch 
from collections import OrderedDict
import timm
import torch.nn as nn
import open_clip

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
    

class Audio_Encoder(nn.Module):
    '''Audio_Emb_Loss 클래스에서 호출함'''
    
    def __init__(self, sequence_length=5, input_size=768, hidden_size=768, backbone_name="resnet18"):

        super(Audio_Encoder,self).__init__()

        self.sequence_length = sequence_length
        
        self.backbone_name = backbone_name
        self.input_size = input_size
    
        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=self.input_size, pretrained=True)
    

    def forward (self,x):

        ''' 기존 tpos: x.shape (batch, 5, 128, 153) => x[:,i,:,:].shape (batch, 128, 153) reshape => (batch, 1, 128, 153) => feature_extractor output => (batch, 768)
        a=torch.zeros(self.size,self.sequence_length,768).cuda() # a => (batch=batch, 5, 768)
        for i in range(self.sequence_length):                    # 기존 tpos: x => (batch, 5, 128, 153) => reshape => (batch, 1, 128, 153) => feature_extractor output => (batch, 768)
            a[:,i,:] = self.feature_extractor(self.conv(x[:,i,:,:].reshape(self.size,1,128,self.hidden_size//self.sequence_length)))
        x=a  # x => (batch, 5, 768)'''

        ''' 수정된 방법: x => (batch, 128, 153) => reshape => (batch, 1, 128, 153) => feature_extractor output => (batch, 768) '''
        x = self.feature_extractor(self.conv(x.reshape(x.shape[0],1,128,self.hidden_size//self.sequence_length)))

        return x    # x.shape => (batch, 768)
    
class Audio_Emb_Loss(nn.Module):
    
    def __init__(self, model_path="../pretrained_models/audio_encoder_23.pth"):

        super(Audio_Emb_Loss,self).__init__()

        self.model = Audio_Encoder()
        self.model_path = model_path
        model_dict = self.model.state_dict()

        # print(model_dict.keys())

        pretrained_model = TPoS_Audio_Encoder()
        pretrained_model.load_state_dict(copyStateDict(torch.load(self.model_path)))

        pretrained_dict = pretrained_model.state_dict()
        # print(pretrained_dict.keys())

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

        # # 모델의 각 레이어의 weight를 출력하여 확인
        # for name, param in model.named_parameters():
        #     print(name, param)  

        self.model = self.model.cuda()
        self.model.eval()
    
    def forward (self,x):
        x = self.model(x).float()
        x = x/x.norm(dim=-1,keepdim=True)
        return x # (batch, 768)
    
    
class Mapping_Model(nn.Module):
    def __init__(self, max_length=77):
        super().__init__()
        self.max_length = max_length-1
        self.input_c = 768
        self.output_c = 1024

        # 768, 76//7x1024
        # self.linear1 = torch.nn.Linear(self.input_c,self.max_length//7*self.output_c) 
        self.linear1 = torch.nn.Linear(self.input_c,1024) 
        
        # 76//7x1024, 76x1024
        self.linear2 = torch.nn.Linear(1024,self.max_length*self.output_c)
        # self.linear3 = torch.nn.Linear(self.max_length//7*self.output_c,self.max_length//7*self.output_c)
        # self.linear4 = torch.nn.Linear(self.max_length//7*self.output_c,self.max_length*self.output_c)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        # x = self.linear1(x)
        # x = self.drop(x)
        # x = self.act(x)

        # x = self.linear2(x)
        # x = self.drop(x)
        # x = self.act(x)

        # x = x.reshape(-1,self.max_length,1024) # x.shape => torch.Size([batch, 76, 1024])



        # 첫 번째 레이어의 출력
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)

        # 두 번째 레이어의 출력
        x = self.linear2(x)
        x = self.act(x)
        x = self.drop(x)

        # # 3 번째 레이어의 출력
        # x = self.linear3(x)
        # x = self.act(x)
        # x = self.drop(x)

        # # 4 번째 레이어의 출력
        # x = self.linear4(x)
        # x = self.act(x)
        # x = self.drop(x)

        # 최종 출력 형태 조정
        x = x.reshape(-1, self.max_length, 1024)

        return x


class TPoS_Audio_Encoder(nn.Module):
    '''모델 가중치 일부분만 가져오는 용도'''
    
    def __init__(self, sequence_length=5, lstm_hidden_dim=768, input_size=768, hidden_size=768, num_layers=1,backbone_name="resnet18", ngpus = 4):

        super(TPoS_Audio_Encoder,self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_dim=lstm_hidden_dim
        
        
        self.T_A = nn.Linear(sequence_length*lstm_hidden_dim, 512)
        self.T_A2 = nn.Linear(self.sequence_length*lstm_hidden_dim, self.sequence_length*512)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.backbone_name = backbone_name
        self.num_layers = num_layers
        self.input_size = input_size
    
        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.conv2 = torch.nn.Conv2d(1,77,(1,1)) 
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=self.input_size, pretrained=True)
    
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=num_layers, batch_first=True)
        self.ngpus=ngpus
    
        self.cnn = nn.Conv1d(768,1, kernel_size=1)

    def forward (self,x):

        a=torch.zeros(x.shape[0],self.sequence_length,768).cuda()
        for i in range(self.sequence_length):
            a[:,i,:] = self.feature_extractor(self.conv(x[:,i,:,:].reshape(x.shape[0],1,128,self.hidden_size//self.sequence_length)))
        x=a
        h_0 = Variable(torch.zeros( self.num_layers,x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros( self.num_layers,x.size(0),  self.hidden_size)).cuda()
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        output = output/output.norm(dim=-1,keepdim=True)
        
        output_permute = output.permute(0,2,1)

        beta_t = self.cnn(output_permute).squeeze()

        beta_t=self.softmax(beta_t)

        out=output[:,0,:].mul(beta_t[:,0].reshape(x.shape[0],-1))

        out=out.unsqueeze(1)


        for i in range(1,self.sequence_length):
            next_z=output[:,i,:].mul(beta_t[:,i].reshape(x.shape[0],-1) )
            out=torch.cat([out,next_z.unsqueeze(1)],dim=1)

        return output[:,-1,:], out, beta_t


class AudioEncoder(torch.nn.Module):
    '''아직 사용 안 함'''
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x) # 오디오 feature 추출
        return x
    

class SoundCLIPLoss(torch.nn.Module):
    '''아직 사용 안 함'''

    def __init__(self, model_path="../pretrained_models/resnet18_57.pth"):
        super(SoundCLIPLoss, self).__init__()
        # self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        # self.upsample = torch.nn.Upsample(scale_factor=7)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=512 // 32)
        self.model_path = model_path

        self.audio_encoder = AudioEncoder()
        self.audio_encoder.load_state_dict(copyStateDict(torch.load(self.model_path)))

        self.audio_encoder = self.audio_encoder.cuda()
        self.audio_encoder.eval()

    def forward(self, audio):
        # print(audio.shape)
        audio_features = self.audio_encoder(audio).float()
        # print(audio_features.shape)
        return audio_features

# if __name__ == "__main__":

    # # model = Audio_Emb_Loss()

    # # # 모델의 각 레이어의 weight를 출력하여 확인
    # # for name, param in model.named_parameters():
    # #     print(name, param)  


    # # model = model.cuda()
    # # model.eval()


    # map_model = Mapping_Model()
    # # 모델의 각 레이어의 weight를 출력하여 확인
    # map_model.load_state_dict(torch.load("../pretrained_models/unav_map_model2_0_audio_emb_loss.pth"))
    # for name, param in map_model.named_parameters():
    #     print(name, param)  

    # map_model2 = Mapping_Model()
    # # 모델의 각 레이어의 weight를 출력하여 확인
    # map_model2.load_state_dict(torch.load("../pretrained_models/unav_map_model2_40_audio_emb_loss.pth"))
    # for name, param in map_model2.named_parameters():
    #     print(name, param)  

    
    
    

    
