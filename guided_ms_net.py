import torch
import torch.nn as nn

from nconv_sd import CNN as unguided_net

class CNN(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 
        
        # Import the unguided network
        self.d_net = unguided_net(pos_fn)

        
        self.d = nn.Sequential(
          nn.Conv2d(1,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU()                                                  
        )#11,664 Params
        
        # RGB stream
        self.rgb = nn.Sequential(
          nn.Conv2d(4,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU()                                                  
        )#186,624 Params

        # Fusion stream
        self.fuse = nn.Sequential(
          nn.Conv2d(80,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,32,3,1,1),
          nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          #nn.Conv2d(64,64,3,1,1),
          #nn.ReLU(),
          nn.Conv2d(32,1,1,1),
          #nn.Sigmoid()                                                  
        )# 156,704 Params
            
        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

        
        self.x0_d = []
        self.xout_d = []
        self.x0_rgb = []
        self.xout_rgb = []
        self.xf = []

        
        self.c0 = []
        self.cout_d = []
        
            
    def forward(self, x0):  

        assert x0.shape[1] == 4, "The input is not RGB-D"

        x0_rgb = x0[:,:3,:,:]
        x0_d = x0[:,3:,:,:]
        c0 = (x0_d>0).float()

        self.x0_d = x0_d
        self.x0_rgb = x0_rgb
        self.c0 = c0
        
        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)
        
        xout_d = self.d(xout_d)
        
        self.xout_d = xout_d
        self.cout_d = cout_d
        

        # RGB network
        xout_rgb = self.rgb(torch.cat((x0_rgb, cout_d),1))
        self.xout_rgb = xout_rgb
        
        # Fusion Network
        xout = self.fuse(torch.cat((xout_rgb, xout_d),1))

        self.xf = xout
        self.cf = cout_d
        
        return xout

