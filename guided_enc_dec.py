import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d


from nconv_sd import CNN as unguided_net


class CNN(nn.Module):

    def __init__(self, pos_fn='SoftPlus'):
        
        super().__init__() 
        
        # Import the unguided network
        self.d_net = unguided_net(pos_fn)

                
        # U-Net
        self.conv1 = Conv2d(5, 80, (3,3), 2, 1, bias=True)
        self.conv2 = Conv2d(80, 80, (3,3), 2,1, bias=True)
        self.conv3 = Conv2d(80, 80, (3,3), 2, 1, bias=True)
        self.conv4 = Conv2d(80, 80, (3,3), 2, 1, bias=True)
        self.conv5 = Conv2d(80, 80, (3,3), 2, 1, bias=True)
                
        self.conv6 = Conv2d(80+80, 64, (3,3), 1, 1, bias=True)
        self.conv7 = Conv2d(64+80, 64, (3,3), 1, 1, bias=True)
        self.conv8 = Conv2d(64+80, 32, (3,3), 1, 1, bias=True)
        self.conv9 = Conv2d(32+80, 32, (3,3), 1, 1, bias=True)
        self.conv10 = Conv2d(32+1, 1, (3,3), 1, 1, bias=True)
        

            
        # Init Weights
        cc = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, \
        self.conv6, self.conv7, self.conv8, self.conv9, self.conv10,]
        for m in cc:            
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        
        self.x0_d = []
        self.xout_d = []
        self.x0_rgb = []
        self.xout_rgb = []
        self.xf = []
        self.x5 = []

        
        self.c0 = []
        self.cout_d = []
        
            
    def forward(self, x0 ):  

        assert x0.shape[1] == 4, "The input is not RGB-D"
        x0_rgb = x0[:,:3,:,:]
        x0_d = x0[:,3:,:,:]
        c0 = (x0_d>0).float()

        self.x0_d = x0_d
        self.x0_rgb = x0_rgb
        self.c0 = c0
        
        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)
        
        self.xout_d = xout_d
        self.cout_d = cout_d
        
        # U-Net
        x1 = F.relu(self.conv1(torch.cat((xout_d, x0_rgb,cout_d),1)))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
               
        self.x5 = x5 

        
        # Upsample 1 
        x5u = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x6 = F.leaky_relu(self.conv6(torch.cat((x5u, x4),1)), 0.2)
        
        # Upsample 2
        x6u = F.interpolate(x6, x3.size()[2:], mode='nearest')
        x7 = F.leaky_relu(self.conv7(torch.cat((x6u, x3),1)), 0.2)
        
        # Upsample 3
        x7u = F.interpolate(x7, x2.size()[2:], mode='nearest')
        x8 = F.leaky_relu(self.conv8(torch.cat((x7u, x2),1)), 0.2)
        
        # Upsample 4
        x8u = F.interpolate(x8, x1.size()[2:], mode='nearest')
        x9 = F.leaky_relu(self.conv9(torch.cat((x8u, x1),1)), 0.2)
                
        # Upsample 5
        x9u = F.interpolate(x9, x0_d.size()[2:], mode='nearest')
        x10 = F.leaky_relu(self.conv10(torch.cat((x9u, x0_d),1)), 0.2)
        
        xout = x10

       
        self.xf = xout

        return xout
       
