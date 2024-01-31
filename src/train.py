import torch as T
from torch.utils.data import DataLoader


import numpy as np
from tqdm import tqdm


from model.model import *
from model.utils_pytorch import *
from utils.read_utils import *
from utils.config_file import *


args = load_args()


#Seed
T.manual_seed(32)
np.random.seed(32)

#Device
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(device)

#import data
x_train, x_val, x_test, y_train, y_val, y_test = import_data(args)

#Normalization
#x_train, x_val, x_test, y_train, y_val, y_test = normalization(x_train, x_val, x_test, y_train, y_val, y_test)

x_train, x_val, x_test = T .from_numpy(x_train).float(), T.from_numpy(x_val).float(), T.from_numpy(x_test).float(), 
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


#Loss and Optimizer
loss_func = T.nn.MSELoss()
optimizer = T.optim.Adam(model.parameters(), lr=args.INIT_LR)
step_lr_scheduler = T.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7,gamma=0.1)
best_loss = 10
early_stopper = EarlyStopper(patience=3, min_delta=0.01)

#Training
for epoch in range(args.N_EPOCHS):
    train_loss = 0.0
    
    for data in tqdm(train_dataloader):
        input_data, fluctuation = data
        input_data = input_data.to(device)
        fluctuation = fluctuation.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        model.train() 
        #ic(input_data.shape)
        outputs  = model.forward(input_data)
        #ic(outputs.shape)
   
        # calculate the loss
        #ic(fluctuation.shape)
        loss_x1 = loss_func(outputs, fluctuation.float())

        # backward pass: compute gradient of the loss with respect to model parameters
        loss = loss_x1 
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()

    valid_loss = validate(network=model, valid_dataloader=valid_dataloader, loss_function=loss_func, device=device)
    
    step_lr_scheduler.step()

    train_loss = train_loss/len(train_dataloader)

    valid_loss = valid_loss/len(valid_dataloader)
    if valid_loss < best_loss:
        best_loss = valid_loss
    if early_stopper.early_stop(valid_loss):
        break
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

