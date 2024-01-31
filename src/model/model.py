import torch                    # PyTorch
import torch.nn as nn           # Neural network module
from icecream import ic

class CNN_model(nn.Module):
    def __init__(self, input_shape, padding, pad_out, args, pred_fluct=False):
        super().__init__()

        # Define the model architecture.
        self.nb_out = args.N_VARS_OUT

        #1st
        self.conv1  = nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=5, padding=padding)
        self.bch1 =  nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        #2nd
        self.conv2  = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=padding)
        self.bch2 =  nn.BatchNorm2d(128)

        #3rd
        self.conv3  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=padding)
        self.bch3 =  nn.BatchNorm2d(256)

        self.conv4  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=padding)
        self.bch4 =  nn.BatchNorm2d(256)

        self.conv5  = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=padding)
        self.bch5 =  nn.BatchNorm2d(128)

        # Branch 1 
        self.cnv_b1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=padding)


        if pred_fluct == True:
            self.act_b1 = ThresholdReLU(args.threshold)
            self.output_b1 = Cropping2D(pad_out)
        else:
            self.act_b1 = nn.ReLU()
            self.output_b1 = Cropping2D(pad_out)

        if args.N_VARS_OUT == 2:
            # Branch 2
            cnv_b2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=padding)
            if pred_fluct == True:
                act_b2 = ThresholdReLU(args.threshold)
                output_b2_ = Cropping2D(pad_out)
            else:
                act_b2 = nn.ReLU()
                output_b2_ = Cropping2D(pad_out)
    	    
            self.output_b2 = nn.Sequential(cnv_b2, act_b2, output_b2_)
        
        
        elif args.N_VARS_OUT == 3:
            # Branch 2
            cnv_b2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=padding)
            if pred_fluct == True:
                act_b2 = ThresholdReLU(args.threshold)
                output_b2_ = Cropping2D(pad_out)
            else:
                act_b2 = nn.ReLU()
                output_b2_ = Cropping2D(pad_out)
            
            self.output_b2 = nn.Sequential(cnv_b2, act_b2, output_b2_)
            """     self.output_b2.add_module('cnv_b2', cnv_b2)
            self.output_b2.add_module('act_b2', act_b2)
            self.output_b2.add_module('output_b2_', output_b2_) """

            # Branch 3
            cnv_b3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=padding)
            if pred_fluct == True:
                act_b3 = ThresholdReLU(args.threshold)
                output_b3_ = Cropping2D(pad_out)
            else:
                act_b3 = nn.ReLU()
                output_b3_ = Cropping2D(pad_out)
            
            self.output_b3 = nn.Sequential(cnv_b3, act_b3, output_b3_)
            """ self.output_b3.add_module('cnv_b3', cnv_b3)
            self.output_b3.add_module('act_b3', act_b3)
            self.output_b3.add_module('output_b3_', output_b3_) """

    def forward(self, x):
        """
        This method forwards the input data through the model.

        Args:
            x: The input data.

        Returns:
            The output data.


        """
        #ic(x.shape)
        out = self.conv1(x)
        out = self.bch1(out)
        out = self.relu(out)
        #ic(out.shape)

        out = self.conv2(out)
        out = self.bch2(out)
        #ic(out.shape)

        out = self.conv3(out)
        out = self.bch3(out)
        #ic(out.shape)

        out = self.conv4(out)
        out = self.bch4(out)
        #ic(out.shape)

        out = self.conv5(out)
        out = self.bch5(out)
        #ic(out.shape)

        #Branch 1
        out1 = self.cnv_b1(out)
        out1 = self.act_b1(out1)
        out1 = self.output_b1(out1)
        #ic(out1.shape)
        
        
        if self.nb_out == 2:
            out = torch.cat([out1, self.output_b2(out)], dim=1)
        elif self.nb_out == 3:
            out = torch.cat([out1, self.output_b2(out), self.output_b3(out)], dim=1)
        else:
            out = out1
        
        """ if self.nb_out == 2:
            print('hell_yeah')
            return out1, self.output_b2(out)
        elif self.nb_out == 3:
            return out1, self.output_b2(out), self.output_b3(out) """
        #ic(out.shape)
        return out


#Pytorch Modules

class ThresholdReLU(nn.Module):
    """
    This class implements a threshold ReLU activation function.

    Args:
        threshold: The threshold value.

    Returns:
        A PyTorch module that implements the threshold ReLU activation function.
    """

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        """
        This method forwards the input data through the threshold ReLU activation function.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        return torch.where(x < self.threshold, 0, x)


class Cropping2D(nn.Module):
    """
    This class implements a cropping operation on a 2D tensor.

    Args:
        cropping: The cropping parameters.
        data_format: The data format.
        name: The name of the operation.

    Returns:
        A PyTorch module that implements the cropping operation.
    """

    def __init__(self, cropping):
        super().__init__()
        #self.cropping = int(cropping)
        self.crop_layers = torch.nn.Sequential()
        for i in  range(int(cropping/2)):
            nm = 'crop'+str(i)
            self.crop_layers.add_module(nm, nn.ZeroPad2d(-1))

    

    def forward(self, x):
        """
        This method forwards the input tensor through the cropping operation.

        Args:
            x: The input tensor.

        Returns:
            The cropped tensor.
        """
        return self.crop_layers(x)

class CustomLoss(torch.nn.Module):
    def __init__(self, loss_fn=torch.nn.MSELoss):
        super().__init__(self)
        self.loss_fn = loss_fn()

    def forward(self, pred_list, target_list):
        assert len(pred_list) == len(target_list)
        value = 0
        for _pred, _target in zip(pred_list, target_list):
            value = value + self.loss_fn(_pred, _target)

        return value