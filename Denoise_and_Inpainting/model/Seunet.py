import torch
import torch.nn as nn
class SEnetGenerator(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SEnetGenerator,self).__init__()
        self.down_conv0 = nn.Conv2d(1,64,4,stride=2,padding=1)
        self.down_conv1 = nn.Conv2d(64,128,4,stride=2,padding=1)
        self.down_conv2 = nn.Conv2d(128,256,4,stride=2,padding=1)
        self.down_conv3 = nn.Conv2d(256,512,4,stride=2,padding=1)
        self.down_conv4 = nn.Conv2d(512,512,4,stride=2,padding=1)
        self.leakyrelu        = nn.LeakyReLU(0.2)
        self.relu                   = nn.ReLU()
        self.upsample       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)       
        self.up_conv5       = nn.ConvTranspose2d(512,512,3,stride=1,padding=1)
        self.up_conv6       = nn.ConvTranspose2d((512+512),256,3,stride=1,padding=1)
        self.up_conv7       = nn.ConvTranspose2d((256+256),128,3,stride=1,padding=1)
        self.up_conv8       = nn.ConvTranspose2d((128+128),64,3,stride=1,padding=1)
        self.up_conv9       = nn.ConvTranspose2d((64+64),1,3,stride=1,padding=1)
    def forward(self,input):
        conv0 = self.down_conv0(input)
        #conv0 = torch.squeeze(conv0)
        conv1 =self.leakyrelu(self.down_conv1(conv0))
        conv2 = self.leakyrelu(self.down_conv2(conv1))
        conv3 = self.leakyrelu(self.down_conv3(conv2))
        conv4 = self.leakyrelu(self.down_conv4(conv3))
                
        conv5     =  self.relu(self.up_conv5(self.upsample(conv4)))
        concat5 = torch.cat([conv5,conv3],dim=1)
        conv6     =  self.relu(self.up_conv6(self.upsample(concat5)))
        concat6 = torch.cat([conv6,conv2],dim=1)
        conv7     =  self.relu(self.up_conv7(self.upsample(concat6)))
        concat7 = torch.cat([conv7,conv1],dim=1)
        conv8     =  self.relu(self.up_conv8(self.upsample(concat7)))
        concat8 = torch.cat([conv8,conv0],dim=1)
        out          = self.up_conv9(self.upsample(concat8))
        ##upsample = torch.nn.functional.grid_sample(out,self.grid,mode='bilinear',padding_mode='zeros')
        return out
