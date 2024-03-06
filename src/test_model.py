import torch as T
from torch.utils.data import DataLoader


import numpy as np
from tqdm import tqdm


from model.model import *
from model.utils_pytorch import *
from utils.read_utils import *
from utils.config_file import *

import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm
import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp2d

args = load_args()


#Seed
T.manual_seed(32)
np.random.seed(32)

#Device
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(device)

#import data
x_train, x_val, x_test, y_train, y_val, y_test, mean_output = import_data(args, mean_output = True)

dir_p_file = args.file_location
data = read_h5(dir_p_file)
inputs, outputs = read_data(args,data)

ic(inputs.shape)



u15 = data['dUdy0']
u15 = u15[:,:,:,0]
u15 = np.transpose(u15, axes = (0,2,1))




x = data['x']
z = data['z']


#fig = plt.figure()
X, Y = np.meshgrid(x, z)
""" ax = fig.add_subplot(111, aspect='equal')

#plt.set_cmap('bwr') # a good start: blue to white to red colormap

#ax = fig.add_subplot(111, aspect='equal')
c = ax.pcolormesh(X, Y, outputs[0,0,:,:]) 
plt.show() """

data = outputs[0,0,:,:]

X, Y = np.meshgrid(x, z)
fig = plt.figure()
plt.pcolormesh(X, Y, data, cmap='BrBG',shading='gouraud')
plt.show()

exit()


print('hey')
# colormesh original
plt.subplot(3, 2, 1)
plt.pcolormesh(X, Y, data, cmap='BrBG')


# pcolormesh with special shading
plt.subplot(3, 2, 2)

# imshow bilinear interp.
plt.subplot(3, 2, 3)
plt.imshow(data, cmap='BrBG', interpolation = 'bilinear')

# imshow bicubic interp.
plt.subplot(3, 2, 4)
plt.imshow(data, cmap='BrBG', interpolation = 'bicubic')
plt.show()
exit()

plt.show()


exit()

#Normalization
#x_train, x_val, x_test, y_train, y_val, y_test = normalization(x_train, x_val, x_test, y_train, y_val, y_test)

x_train, x_val, x_test = T.from_numpy(x_train).float(), T.from_numpy(x_val).float(), T.from_numpy(x_test).float(), 
y_train, y_val, y_test = T.from_numpy(y_train).float(), T.from_numpy(y_val).float(), T.from_numpy(y_test).float()

#Dataset
train_dataset = Dataset_loader(x_train, y_train, args)
val_dataset = Dataset_loader(x_val, y_val, args)
test_dataset = Dataset_loader(x_test, y_test, args)

#Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=True)

#Model
padding = 0
pad_out = 2
model = CNN_model(args.N_VARS_IN, padding, pad_out, args)
model.to(device)

model.load_state_dict(T.load(r'C:\Users\totti\OneDrive\Work\ITA-PUC\Code\Test - 26-12\models\model.pth'))

input_data = x_test.to(device)
fluctuation = y_test.to(device)
#ic(input_data.shape)

output_ = []
t, a = next(iter(train_dataloader))
t = t.to(device)
ic(a.shape)
ic(a[0,0,:,:])

outputs  = model.forward(t)
output = outputs.cpu().detach().numpy()


# Read the data
dir_p_file = r"C:\Users\totti\OneDrive\Work\ITA-PUC\Data\Meio_canal_Retau180\analysis_s2.h5"
data = read_h5(dir_p_file) 

ic(outputs.shape)
ic(fluctuation.shape)

fluctuation = a.cpu().detach().numpy()
outputs = output

x = data['x']
z = data['z']


#fig, (ax0, ax1) = plt.subplots(2, 1) 





""" for i in range(22,30):
    print(i)
    c = ax0.pcolor(X, Y, fluctuation[i,0,:,:]) 
    c = ax1.pcolor(X, Y, outputs[i,0,:,:])
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events() 
 """

#Plot velocity correctly
#ic(outputs[2,0,:,:])
#ic(fluctuation[2,0,:,:])
#Mean of U_labels

#Plotting
#


fig, (ax0, ax1) = plt.subplots(2, 1) 
#plt.set_cmap('bwr') # a good start: blue to white to red colormap
print('hey')

#ax = fig.add_subplot(111, aspect='equal')
c = ax0.pcolor(X, Y, fluctuation[0,0,:,:]) 
c = ax1.pcolor(X, Y, outputs[0,0,:,:]) 
plt.show()


exit()

for nt in range(0,32):

    u_labels = np.copy(fluctuation[nt,0,:,:])
    u_hat = np.copy(outputs[nt,0,:,:])
    ic(u_labels.shape)
    exit()

    #-mean
    u_labels = u_labels - np.average(fluctuation[:,0,:,:])
    u_hat = u_hat - np.average(u_hat)

