from loader import datas
from torch.utils.data import  DataLoader
from net import AUTOMAP
from torch import nn,optim,min,max
from cv2 import imwrite
import torch
import numpy as np
def main():
    dataset = datas()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net = AUTOMAP([1,256,256],[1,256,256])
    net = torch.load("/media/Data/data_prepare/adn-master/tt/model/   6.pt")
    net.to(device)
    criterion = nn.MSELoss()
    optimzer = optim.RMSprop(net.parameters(),lr=0.00005)
    iter = 0
    proces = 0
    #sumloss=0.0
    with torch.no_grad():
        done = 0
        sumloss=0.0
        for img,label in dataloader:
            done += 1
            img,label = img.to(device),label.to(device)
            #optimzer.zero_grad()
            predict = net.forward(img)

            predict = (predict-min(predict)) / (max(predict)-min(predict))
            predict = predict*255
            loss = criterion(predict,label)
            print(torch.sqrt(loss).data.item())
            sumloss+=loss.data.item()
            #loss.backward()
            #optimzer.step()
            if proces%100==0:
                imwrite("./test/%d.png" % proces, predict[0].cpu().detach().squeeze().numpy())
            proces+=1
        print("avg:")
        print(sumloss/done)

main()
