import os
import os.path as path
import argparse
import matlab
import numpy as np
import matlab.engine
from Preprocess.dataset import get_dataset
from Preprocess.utils import get_config, update_config, save_config, read_dir, add_post
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
from model.model import UNet
from model import pytorch_unet
import torch
from model import Seunet
from model.DnCNN import DnCNN
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from model.PconvNet import PConvUNet
from model.PconvNet import VGG16FeatureExtractor
from model.loss import InpaintingLoss
from model import opt
from PIL import Image

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(
            size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


if __name__ == "__main__":

    # 确定配置文件
    parser = argparse.ArgumentParser(description="Train the Net")
    parser.add_argument("--default_config",
                        default="config/adn.yaml", help="default configs")
    parser.add_argument(
        "--run_config", default="runs/adn.yaml", help="run configs")

    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
    args = parser.parse_args()

    # Get  options
    opts = get_config(args.default_config)
    run_opts = get_config(args.run_config)
    run_opts = run_opts["deep_lesion"]["train"]
    update_config(opts, run_opts)
    run_dir = path.join(opts["checkpoints_dir"], "deep_lesion")
    if not path.isdir(run_dir): os.makedirs(run_dir)
    save_config(opts, path.join(run_dir, "train_options.yaml"))
    def save_image(raw_image,denoise,output,index):
        raw_image = raw_image.cpu().detach().numpy()
        #temp = raw_image[0].reshape((256,256))
        #name = "{}_iter_1.png".format(index)
        output = output.cpu().detach().numpy()
        denoise = denoise.cpu().detach().numpy()
        #img1 = Image.fromarray(raw_image[0].reshape((256,256)))
        name = "./train/{}a_iter_1.png".format(index)
        plt.imsave(name,raw_image[0].reshape((256,256)),cmap="gray")
        #img2 = Image.fromarray(output[0].reshape((256,256)))
        name = "./train/{}c_iter_1.png".format(index)
        plt.imsave(name,output[0].reshape((256,256)),cmap="gray")
        #img3 = Image.fromarray(raw_image[1].reshape((256,256)))
        name = "./train/{}d_iter_2.png".format(index)
        plt.imsave(name,raw_image[1].reshape((256,256)),cmap="gray")
        #img2 = Image.fromarray(output[0].reshape((256,256)))
        name = "./train/{}f_iter_2.png".format(index)
        plt.imsave(name,output[1].reshape((256,256)),cmap="gray")
        name = "./train/{}b_iter_1.png".format(index)
        plt.imsave(name,denoise[0].reshape((256,256)),cmap="gray")
        #img4 = Image.fromarray(output[0].reshape((256,256)))
        name = "./train/{}e_iter_2.png".format(index)
        plt.imsave(name,denoise[1].reshape((256,256)),cmap="gray")
    def save_image(raw_image,proj,index):
        raw_image = raw_image.cpu().detach().numpy()
        #temp = raw_image[0].reshape((256,256))
        #name = "{}_iter_1.png".format(index)
        proj = proj.cpu().detach().numpy()
        #denoise = denoise.cpu().detach().numpy()
        #img1 = Image.fromarray(raw_image[0].reshape((256,256)))
        name = "./train/{}a_iter_1.png".format(index)
        plt.imsave(name,raw_image[0][0].reshape((256,256)),cmap="gray")
        #img2 = Image.fromarray(output[0].reshape((256,256)))
        name = "./train/{}c_iter_1.png".format(index)
        plt.imsave(name,proj[0][0].reshape((256,256)),cmap="gray")
        #img3 = Image.fromarray(raw_image[1].reshape((256,256)))
        # name = "./train/{}d_iter_2.png".format(index)
        # plt.imsave(name,raw_image[1].reshape((256,256)),cmap="gray")
        #img2 = Image.fromarray(output[0].reshape((256,256)))
        # name = "./train/{}f_iter_2.png".format(index)
        # plt.imsave(name,output[1].reshape((256,256)),cmap="gray")
        # name = "./train/{}b_iter_1.png".format(index)
        # plt.imsave(name,denoise[0].reshape((256,256)),cmap="gray")
        #img4 = Image.fromarray(output[0].reshape((256,256)))
        # name = "./train/{}e_iter_2.png".format(index)
        # plt.imsave(name,denoise[1].reshape((256,256)),cmap="gray")
    def get_image(data):
        dataset_type = dataset_opts['dataset_type']
        if dataset_type == "deep_lesion":
            if dataset_opts[dataset_type]['load_mask']: return data['lq_image'], data['hq_image'], data['mask']
            else: return data['lq_image'], data['hq_image']
        elif dataset_type == "spineweb":
            return data['a'], data['b']
        elif dataset_type == "nature_image":
            return data["artifact"], data["no_artifact"]
        else:
            raise ValueError("Invalid dataset type!")
    # Get dataset
    dataset_opts = opts['dataset']
    train_dataset = get_dataset(**dataset_opts)
    train_loader = DataLoader(train_dataset,
        batch_size=opts["batch_size"], num_workers=0, shuffle=True)  # num_workrt debug=0 run =2
    ######################################
    #########Prepare to Train################
    ######################################
    #Denoise
    # Denoisecriterion = sum_squared_error()
    # Denoisemodel = DnCNN()
    # Denoisemodel = Denoisemodel.cuda()
    # Denoiseoptimizer = optim.Adam(Denoisemodel.parameters(), lr=args.lr)
    #Inpainting
    # Inpaintingmodel = PConvUNet().cuda()
    # Inpaintingoptimizer = optim.Adam(filter(lambda p:p.requires_grad,Inpaintingmodel.parameters()),lr =2e-4)
    # Inpaintingcriterion = InpaintingLoss(VGG16FeatureExtractor()).to(torch.device('cuda'))
    unet = Seunet.SEnetGenerator(1,1)
    unet = unet.to(torch.device('cuda'))
    #summary(model,input_size(1,196,320))
    optimizer = optim.Adam(unet.parameters(),lr = 2e-4)
    criterion = torch.nn.L1Loss()
    eng = matlab.engine.start_matlab()
    for epoch in range(0, 60):
            epoch_loss = 0
           # Inpaintingmodel.train()
            for n_count, data in enumerate(train_loader):
                    ##Denoise##
                    image = data['lq_image'].tolist() [0][0]
                    hello = matlab.double(image)
                    #para_dic = {"FanSensorSpacing":0.1,"FanRotationIncrement" :360.0/320}
                    proj = eng.myfanbeam(hello,1075.0)
                    proj = np.asarray(proj)
                    proj = torch.from_numpy(proj).float().to(torch.device('cuda'))
                    proj = proj.expand(1,1,-1,-1)
                    optimizer.zero_grad()
                    #Denoiseoptimizer.zero_grad()
                    #proj = proj.cuda()
                    gt = data['lq_image'][0][0].cuda()
                    gt = gt.expand(1,1,-1,-1)
                    #mask = data['mask'].cuda()
                    #image = data['lq_image'].cuda()

                    # mask = 1-mask
                    # gt = gt*mask
                    # image = image*mask
                    output = unet(proj)
                    loss = criterion(output,gt)
                    epoch_loss += loss.item()
                    loss.backward()

                    optimizer.step()
                    # output = Denoisemodel(image)
                    # loss = Denoisecriterion(output, gt)
                    
                    # loss.backward(retain_graph=True)
                    # Denoiseoptimizer.step()
                    # denoiseImage = output
                    # ##Inpainting##
                    # #if epoch>=30:
                    # Inpaintingoptimizer.zero_grad()
                    # finaloutput,_ = Inpaintingmodel(denoiseImage,mask)
                    # loss_dict = Inpaintingcriterion(denoiseImage,mask,finaloutput,data['hq_image'].cuda())
                    # Inpaintingloss = 0.0
                    # for key,coef in opt.LAMBDA_DICT.items():
                    #     value = coef*loss_dict[key]
                    #     Inpaintingloss += value
                    
                    # Inpaintingloss.backward(retain_graph=True)
                    # Inpaintingoptimizer.step()
                    #if n_count%10==0 and epoch<30:
                        #print('%4d %4d  Denoiseloss = %2.4f' % (epoch+1, n_count, loss.item()/2))
                        #print('%4d %4d Inpaintloss = %2.4f' % (epoch+1, n_count, Inpaintingloss.item()/2))
                    #if n_count%100==0 and epoch<30:
                        #save_image(image,denoiseImage,denoiseImage,epoch*1000+n_count)
                    if n_count%10==0: #and epoch>=30:
                        print('%4d %4d  Denoiseloss = %2.4f' % (epoch+1, n_count, loss.item()))
                    if n_count%100==0:
                        save_image(gt,output,n_count)
            #             print('%4d %4d Inpaintloss = %2.4f' % (epoch+1, n_count, Inpaintingloss.item()/2))
            #         if n_count%100==0:# and epoch>=30:
            #             save_image(image,denoiseImage,denoiseImage+(1-mask)*finaloutput,epoch*1000+n_count)
            # print("epoch = %4d,loss = %4.4f  "% (epoch+1,epoch_loss/n_count))
            # torch.save(Denoisemodel, os.path.join(".", 'denoisemodel_%03d.pth' % (epoch+1)))
            # torch.save(Inpaintingmodel, os.path.join(".", 'inpaintingmodel_%03d.pth' % (epoch+1)))
