import torch as T
import numpy as np
from icecream import ic

# Dataset loader
class Dataset_loader(T.utils.data.Dataset):
    def __init__(self, input_fields, output_fields, args):
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.args = args
        
    def __len__(self):
        return len(self.input_fields)
    
    def periodic_padding(self, tensor, padding):
        lower_pad = tensor[:padding[0][0],:]
        upper_pad = tensor[-padding[0][1]:,:]
        #print('hey')
        
        partial_tensor = T.cat((upper_pad, tensor, lower_pad), dim=0)
        #ic(partial_tensor.shape)
        
        left_pad = partial_tensor[:,-padding[1][0]:]
        right_pad = partial_tensor[:,:padding[1][1]]
        
        padded_tensor = T.cat((left_pad, partial_tensor, right_pad), dim=1)
        #ic(padded_tensor.shape)
        return padded_tensor

    def __getitem__(self, idx):
        input_fields = T.zeros([3, self.input_fields.shape[2]+self.args.PADDING_SIZE, 
                                self.input_fields.shape[3]+self.args.PADDING_SIZE])
        input_field = self.input_fields[idx]
        output_fluctuation = self.output_fields[idx]

        if self.args.PADDING_INPUT:
            for i in range(self.args.N_VARS_IN):
                pad = int(self.args.PADDING_SIZE/2)
                input_fields[i] = self.periodic_padding(input_field[i], ((pad, pad), (pad, pad)))

        return input_fields, output_fluctuation




#Training Functions
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def validate(network, valid_dataloader, loss_function, device):
  """
  This function validates convnet parameter optimizations
  """
  #  creating a list to hold loss per batch
  loss_per_batch = 0.0

  #  defining model state
  network.eval()

  #  defining dataloader

  print('validating...')
  #  preventing gradient calculations since we will not be optimizing
  with T.no_grad():
    #  iterating through batches
    for input, true_output in valid_dataloader:
      #--------------------------------------
      #  sending images and labels to device
      #--------------------------------------
      images, true_output = input.to(device), true_output.to(device)
      outputs = network.forward(images)
      # calculate the loss
      loss_x1 = loss_function(outputs, true_output.float())

      #-----------------
      #  computing loss
      #-----------------
      loss = loss_x1
      loss_per_batch+= loss.item()

  return loss_per_batch 
