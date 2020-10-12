from loader import datas
from torch.utils.data import  DataLoader
from net import AUTOMAP
from torch import nn,optim,min,max
from cv2 import imwrite
import torch
import numpy as np
from torch import nn
from radon import Radon,IRadon
def main():
    llr = 0.0001
    dataset = datas()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    process =0
    theta = torch.arange(180*4)
    net = IRadon(512, theta, False)
    net.to(device)
    criterion = nn.MSELoss()
    optimzer = optim.RMSprop(net.parameters(),lr=0.00002,weight_decay=0.9)
    for img,label in dataloader:
            circle = False
            optimzer.zero_grad()
            # label = (label-min(label)) / (max(label)-min(label))
            # label = label*255
            r = Radon(label.shape[2], theta, circle)
           # ir = IRadon(label.shape[2], theta, circle)
            label = label.to(device)
            img = img.to(device)
            sino = r(label)
            sino = sino.to(device)
            reco = net(sino)
            # reco = (reco-min(reco)) / (max(reco)-min(reco))
            # reco = reco*255
            loss = criterion(reco,label)
            sino = (sino-min(sino)) / (max(sino)-min(sino))
            sino = sino*255
            print(loss.data.item())
            loss.backward()
            optimzer.step()
            if process%50==0:
                   imwrite("./try/%draw.png" %process  , label.cpu().detach().squeeze().numpy())
                   imwrite("./try/%dsino.png" % process, sino.cpu().detach().squeeze().numpy())
                   imwrite("./try/%dreco.png" %process , reco.cpu().detach().squeeze().numpy())
            process+=1
            
    # net = AUTOMAP([1,128,256],[1,128,128])
    # #net.to(device)
    # #net.apply(weight_init)
    # criterion = nn.MSELoss()
    # #optimzer = optim.Adam(net.parameters(),lr=0.00001)
    # optimzer = optim.RMSprop(net.parameters(),lr=0.000000002,weight_decay=0.9)
    # iter = 0
    # proces = 0
    # #sumloss=0.0
    # while iter<100:
    #     done = 0
    #     sumloss=0.0
    #     for img,label in dataloader:
    #         done += 1
    #        # img,label = img.to(device),label.to(device)
    #         optimzer.zero_grad()
    #         predict = net.forward(img)
           
    #         loss = criterion(predict,label)
    #         label = (label-min(label)) / (max(label)-min(label))
    #         label = label*255
    #         predict = (predict-min(predict)) / (max(predict)-min(predict))
    #         predict = predict*255
    #        # loss = criterion(predict,label)
    #         print(loss.data.item())
    #         sumloss+=loss.data.item()
    #         loss.backward()
    #         optimzer.step()
    #         #llr = adjust_learning_rate(optimzer,iter,llr)
    #         if proces%10==0:
    #             imwrite("./result/%dr.png" % proces, label[0].cpu().detach().squeeze().numpy())
    #            # imwrite("./result/%ds.png" % proces,img.cpu().detach().squeeze().numpy())
    #             imwrite("./result/%d.png" % proces, predict[0].cpu().detach().squeeze().numpy())
    #         proces+=1
    #     print("avg:")
    #     print(sumloss/done)
    #     # if iter%2==0:
    #     #     torch.save(net,"./model/%4d.pt" % iter)
    #     iter+=1

main()
