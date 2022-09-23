"""

This is main model source code for "Retinal_Fundus_Image_based_Deep_Alzheimer_s_Disease_Diagnosis_Network_for_Mobile_Devices".
We used pytorch and timm for this source code.


"""



import torch
import timm
import torch.nn as nn
import torchvision.transforms.functional as F
from collections import OrderedDict


from kornia.contrib.vit_mobile import *
from typing import Tuple


class Decoder_block(nn.Module):
    def __init__(self,x_ch,y_ch,out_ch):
        
        super(Decoder_block, self).__init__()
        
        self.x_to_y=nn.Sequential(
            nn.Conv2d(x_ch, y_ch, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(y_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
        )
        
        self.layer=nn.Sequential(
            nn.Conv2d(y_ch, y_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=y_ch, bias=False),
            nn.BatchNorm2d(y_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(y_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
        )
  
    def forward(self,x, y):
        
        x=F.resize(x,(y.shape[-2],y.shape[-1]))
        x=self.x_to_y(x)
        
        out=self.layer(x+y)
        
        return out
   
class Dech_block(nn.Module):
    def __init__(self,x_ch,y_ch,out_ch):
        
        super(Dech_block, self).__init__()
        
        self.x_to_y=nn.Sequential(
            nn.Conv2d(x_ch, y_ch, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(y_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
        )
        
    def forward(self,x,y):
        
        x=F.resize(x,(y.shape[-2],y.shape[-1]))
        out=self.x_to_y(x)
        
        return out
    
        


class Mainmodel(nn.Module):
    def __init__(self,weight=True):
        super(Mainmodel, self).__init__()
        backbone=timm.models.tf_mobilenetv3_large_100(pretrained=True)
        backbone.reset_classifier(1)
        
        self.o_enc=Dech_block(3,112,112)
        

        
        self.stem=nn.Sequential(
            backbone.conv_stem,
            backbone.bn1
        )
        # self.stem=backbone.bn1
        
#         32
#       Encoder  ---------------------------------------------------------------
        self.enc=backbone.blocks[0:5]
        # 32-16
        # 16-24
        # 24-40
        # 40-80
        # 80-112
        
#       Decoder  ---------------------------------------------------------------
        self.dec=nn.Sequential(OrderedDict([
                    ('0',Decoder_block(112,80,40)),
                    ('1',Decoder_block(40,40,24)),
                    ('2',Decoder_block(24,24,16)),
                    ('3',Decoder_block(16,16,1))])
        )
        
        self.dec_enc=Dech_block(1,112,112)
#       classifier -------------------------------------------------------------
        self.classifier=nn.Sequential(
            backbone.blocks[5],
            backbone.blocks[6],
            backbone.global_pool,
            backbone.conv_head,
            backbone.act2,
            nn.Flatten(),
            backbone.classifier
        )
    

        
        
        if weight:
            self.o_enc_weight=nn.Parameter(torch.ones(1))
            self.enc_4_weight=nn.Parameter(torch.ones(1))
            self.dec_enc_weight=nn.Parameter(torch.ones(1))
        else:
            
            self.o_enc_weight=nn.Parameter(torch.ones(1),requires_grad=False)
            self.enc_4_weight=nn.Parameter(torch.ones(1),requires_grad=False)
            self.dec_enc_weight=nn.Parameter(torch.ones(1),requires_grad=False)
        
        self.sf=torch.nn.Softmax2d()
        
        
    def forward(self,x):
        
        stem_out=self.stem(x)
        
        enc_0=self.enc[0](stem_out)
        enc_1=self.enc[1](enc_0)
        enc_2=self.enc[2](enc_1)
        enc_3=self.enc[3](enc_2)
        enc_4=self.enc[4](enc_3)
#         -------------------------------------------------------------
        dec_0=self.dec[0](enc_4,enc_3)
        dec_1=self.dec[1](dec_0,enc_2)
        dec_2=self.dec[2](dec_1,enc_1)
        dec_3=self.dec[3](dec_2,enc_0)
#         -------------------------------------------------------------
        
        o_enc=self.o_enc(x,enc_4)
        dec_enc=self.dec_enc(dec_3,enc_4)
        
#         Scaled dot product attention -------------------------------------------------------------
        attention=torch.matmul(self.dec_enc_weight*dec_enc,self.o_enc_weight*o_enc)
        attention=torch.divide(attention,torch.sqrt(torch.tensor(enc_4.shape[-2]*enc_4.shape[-1])))
        attention=self.sf(attention)
        attention=torch.mul(attention,self.enc_4_weight*enc_4)
        
#         print(attention.shape)
#         -------------------------------------------------------------        
        out=self.classifier(enc_4+attention)
#         -------------------------------------------------------------    
        
        
        return out

  