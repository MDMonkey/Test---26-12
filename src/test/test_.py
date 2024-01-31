from utils.read_utils import *
from utils.config_file import *
from model.model import *


#Testing the model
from torchvision import models
from torchsummary import summary


args = load_args()
x_train, x_val, x_test, y_train, y_val, y_test = import_data(args)
print(x_train.shape)
print(y_train.shape)

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))


ala = rms(x_train, axis=0)
ic(ala.shape)


args.N_VARS_OUT = 3
shape = 3
model = CNN_model(shape, 0, 2, args)

model.to('cuda')

summary(model, (3, 64+16, 64+16))
print(model)  
dir = r'C:\Users\totti\OneDrive\Work\ITA-PUC\Data\Meio_canal_Retau180'

exit()
#Testing the read_utils
import os
from read_utils import read_h5
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm

# Define the directory and file name

file = 'analysis_s2.h5'

dir_p_file = os.path.join(dir, file)

# Read the data
data = read_h5(dir_p_file) 

for name,value in data.items():
    print(name)
    #print(value)

ina, out = read_data(data)
print(ina.shape)


du_dy0 = data['dUdy0']
print(du_dy0.shape)
#du_dy0 = du_dy0.reshape((64,64,du_dy0.shape[0],))
du_dy0 = np.transpose(du_dy0[:,:,:,0], axes = (1,2,0))
print(du_dy0.shape)
#du_dy0 = du_dy0[:,:,122]
dw_dy0 = data['dWdy0']
x = data['x']
z = data['z']

print(du_dy0.shape)
#du_dy0_ = du_dy0.reshape((du_dy0.shape[0],1,64,64))
#print(du_dy0_.shape)
#print(du_dy0_[1])

dw_dy0 = dw_dy0[0].reshape((64,64))
pad = 8
ayala = periodic_padding(dw_dy0, ((pad,pad),(pad,pad)))
print(ayala.shape)
ila = ((pad,pad),(pad,pad))
print(type(ila))

print(dw_dy0.shape)
exit()

fig, (ax0, ax1) = plt.subplots(2, 1) 
X, Y = np.meshgrid(x, z)

for i in range(25,100):
    print(i)
    c = ax0.pcolor(X, Y, du_dy0[:,:,i]) 
    c = ax1.pcolor(X, Y, ina[i,0,:,:])
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events() 