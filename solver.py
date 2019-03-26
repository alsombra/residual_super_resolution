import torch
import torch.nn as nn
import os
from torchvision.utils import save_image, make_grid
from model import SRMD
import numpy as np
from data_loader import Scaling, Scaling01

torch.set_default_tensor_type(torch.DoubleTensor)


class Solver(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.num_blocks = config.num_blocks  # num_blocks = 11
        self.num_channels = config.num_channels  # num_channels = 18
        self.conv_dim = config.conv_dim # conv_dim = 128
        self.scale_factor = config.scale_factor  # scale_factor = 2

        # Training settings
        self.total_step = config.total_step # 20000
        self.lr = config.lr 
        self.beta1 = config.beta1 # 0.5  ????????????????? testar 0.9 ou 0.001
        self.beta2 = config.beta2 # 0.99  ???????????????
        self.trained_model = config.trained_model
        self.use_tensorboard = config.use_tensorboard
        self.start_step = -1
        
        # Path and step size
        self.log_path = config.log_path
        self.result_path = config.result_path
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step # log_step = 10
        self.sample_step = config.sample_step # sample_step = 100
        self.model_save_step = config.model_save_step # model_save_step = 1000

        # Device configuration
        self.device = config.device

        # Initialize model
        self.build_model() 
        
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def build_model(self):
        # model and optimizer
        self.model = SRMD(self.num_blocks, self.num_channels, self.conv_dim, self.scale_factor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2])

        self.model.to(self.device)

    def load_trained_model(self):
        self.load(os.path.join(self.model_save_path, '{}'.format(self.trained_model)))
        print('loaded trained model (step: {})..!'.format(self.trained_model.split('.')[0])) 

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S['SR'])
        try:
            self.optimizer.load_state_dict(S['optimizer_state_dict'])
        except KeyError as error:
            print('There is no '+str(error)+' in loaded model. Loading model without optimizer_params')
        try:
            self.start_step = S['epoch'] - 1
        except KeyError as error:
            print('There is no '+str(error)+' in loaded model. Loading model without epoch info')
        
    #############################################
    def build_tensorboard(self):
        pass
    ######################################################
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.optimizer.zero_grad()
    ######################################################################
    def detach(self, x):  # NOT USED. To learn more SEE https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
        return x.data
    #############################################################
    def train(self):
        self.model.train()

        # Reconst loss
        reconst_loss = nn.MSELoss()

        # Data iter
        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)
        
        #Initialize steps
        start = self.start_step + 1 # if not loading trained start = 0     

        for step in range(self.start_step, self.total_step):
            
            self.model.train() # adicionei pq o cara no fim (p/ samples) colocou modo eval() e esqueceu de voltar

            # Reset data_iter for each epoch  
            if (step+1) % iter_per_epoch == 0:     
                data_iter = iter(self.data_loader)  

            lr_image, hr_image, x, y = next(data_iter)
            lr_image, hr_image, x, y = lr_image.to(self.device), hr_image.to(self.device), x.to(self.device), y.to(self.device)

            y = y.to(torch.float64)

            out = self.model(x)
            loss = reconst_loss(out, y)

            self.reset_grad()

            # For decoder
            loss.backward(retain_graph=True)

            self.optimizer.step()

            # Print out log info
            if (step+1) % self.log_step == 0:
                print("[{}/{}] loss: {:.5f}".format(step+1, self.total_step, loss.item()))

              # Sample images

            if (step+1) % self.sample_step == 0:
                from PIL import Image
                self.model.eval()
                reconst = self.model(x)

                #tmp = nn.functional.interpolate(input = x.data[:,0:3,:], scale_factor=scale_factor, mode = 'nearest')
                tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
                image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) for i in range(self.data_loader.batch_size)]
                image_hr_bicubic= np.stack(image_list).transpose(0,3,1,2)
                image_hr_bicubic = Scaling(image_hr_bicubic)
                image_hr_bicubic = torch.from_numpy(image_hr_bicubic).double().to(self.device) # NUMPY to TORCH
                hr_image_hat = reconst + image_hr_bicubic
                
                hr_image_hat = hr_image_hat.data.cpu().numpy()
                hr_image_hat = Scaling01(hr_image_hat)
                hr_image_hat = torch.from_numpy(hr_image_hat).double().to(self.device) # NUMPY to TORCH

                pairs = torch.cat((image_hr_bicubic.data, \
                                hr_image_hat.data,\
                                hr_image.data), dim=3)
                grid = make_grid(pairs, 1) 
                tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
                tmp = (255 * tmp).astype(np.uint8)
                Image.fromarray(tmp).save('./samples/test_%d.jpg' % (step + 1))

#             # Sample images
#             if (step+1) % self.sample_step == 0:
#                 self.model.eval()  #aqui ele botou o modelo em modo eval() mas esqueceu de voltar pro modo train
#                 reconst = self.model(x)

#                 def to_np(x):
#                     return x.data.cpu().numpy()
                
#                 #tmp = nn.functional.interpolate(input = x.data[:,0:3,:], scale_factor=self.scale_factor, mode = 'nearest') ####NEAREST É O MELHOR PRA COMPARAR?##########
#                 tmp = x.data[:,0:3,:]
#                 for i in range(tmp.shape[0]):
#                     Image.fromtmp[i].
#                 #talvez seja melhor fazer um upsample bicubico pra comparar! Segue:
#                 # image_lr = to_np(x)
#                 # simple_upscaled_lr = image_lr.resize((128,128),Image.BICUBIC)
#                 # tmp = simple_upscaled_lr
#                 pairs = torch.cat((tmp.data[0:2,:,:], reconst.data[0:2,:,:], y.data[0:2,:,:]), dim=3)
#                 grid = make_grid(pairs, 2)
#                 from PIL import Image
#                 tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
#                 tmp = (255 * tmp).astype(np.uint8)
#                 Image.fromarray(tmp).save('./samples/test_%d.jpg' % (step + 1))

            # Save check points
            if (step+1) % self.model_save_step == 0:                
                self.save(step, loss.item(), os.path.join(self.model_save_path, '{}.pth.tar'.format(str(step+1))))

    def save(self, step, current_loss, filename):
        model = self.model.state_dict()
        torch.save({
            'epoch': step+1,
            'SR': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': str(current_loss)
            }, filename)

    def valid(self, img_lr, img_hr): #receives batch from dataloader
        'Receives a pair in the same dataloader batch format'
        self.model.eval()
        x, y = img_lr, img_hr
        x, y = x.to(self.device), y.to(self.device)
        y = y.to(torch.float64)
        print('x shape: ', x.shape)
        print('y_shape: ', y.shape)
        reconst = self.model(x)
        reconst_loss = nn.MSELoss()
        loss = reconst_loss(reconst, y)

        # Print out log info 
        print("model trained for {} epochs, loss: {:.4f}".format(self.start_step, loss.item()))

        tmp = nn.functional.interpolate(input = x.data[:,0:3,:], scale_factor=self.scale_factor, mode = 'nearest')
                
        pairs = torch.cat((tmp.data, reconst.data, y.data), dim=3)
        grid = make_grid(pairs, 1)
        from PIL import Image,ImageDraw
        tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
        tmp = (255 * tmp).astype(np.uint8)
        
        img = Image.new('RGB', (300, tmp.shape[0]), color = 'white')
        d = ImageDraw.Draw(img)
        d.text((10,10), "model trained for {} epochs, loss: {:.4f}".format(self.start_step, loss.item()), fill='black')
        imgs_comb = np.hstack((np.array(tmp), img))
        imgs_comb = Image.fromarray(imgs_comb)
    
        from IPython.display import display
        grid_PIL = imgs_comb
        grid_PIL.save('./test_results/valid_{}_{}.jpg'.format(self.start_step + 1,np.random.rand(1)))

    def test(self, lr_image): #receives single image --> can be easily modified to handle multiple images
        'receives single LR image as input. Returns LR image + (models approx) HR image concatenated'
        self.model.eval()
        # input (low-resolution image)
        transform = transforms.Compose([
                    transforms.Lambda(lambda x: Scaling(x)), # LR --> [0,1]
                    transforms.Lambda(lambda x: randkern.ConcatDegraInfo(x))
        ])
        lr_image = transform(lr_image)
        transform = transforms.ToTensor()
        lr_image.to(torch.float64)
        #Add one more dimension, the batch_size (to pass tensor to model with correct input shape)
        lr_image_expanded = lr_image.reshape(1, lr_image.shape[0], lr_image.shape[1], lr_image.shape[2]) #ex: reshape(1,18,64,64)
        x = lr_image_expanded
        x = x.cuda() #cuda
        reconst = self.model(x)
        
        tmp = nn.functional.interpolate(input = x.data[:,0:3,:], scale_factor=self.scale_factor, mode = 'nearest')

        pairs = torch.cat((tmp.data[0:2,:], reconst.data[0:2,:]),dim=3)
        grid = make_grid(pairs, 1)
        from PIL import Image
        tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
        tmp = (255 * tmp).astype(np.uint8)
        Image.fromarray(tmp).save('./test_results/test_%d.jpg' % (self.start_step + 1))