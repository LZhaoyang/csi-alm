
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from alm import Model
import numpy as np
from torch.optim.lr_scheduler import StepLR
import time
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from metrics import NMSELoss, SE_Loss


# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0001
epochs = 1000
batch_size = 1024
device = torch.device('cuda:7')

best_loss = 100
save_path = "/home/ubuntu/LLM/Weights/csi.pth"
train_TDD_r_path = "/home/ubuntu/LLM/data/H_U_his_train.mat"
train_TDD_t_path = "/home/ubuntu/LLM/data/H_D_pre_train.mat"
key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']

train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=1, is_few=0)  # creat data for training
validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=1)  # creat data for validation
# model = torch.load(save_path, map_location=device, weights_only=False)
model = Model(gpu_id=7,
              pred_len=4, prev_len=16,
              UQh=1, UQv=1, BQh=1, BQv=1, lora = True).to(device)




def save_best_checkpoint(model):  # save model function
    model_out_path = save_path
    torch.save(model, model_out_path)



def train(training_data_loader, validate_data_loader):
    global epochs, best_loss
    print('Start training...')
    for epoch in range(epochs):
        epoch_train_loss, epoch_val_loss = [], []
        total_batch_time = 0
        batch_count = 0
        # 训练模型
        model.train()
        with tqdm(total=len(training_data_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for iteration, (pred_t, prev) in enumerate(training_data_loader, 1):
                pred_t, prev = pred_t.to(device), prev.to(device)
                optimizer.zero_grad()  
                start_time = time.time()
                

                
           
                pred_m = model(prev, None, None, None)

                
        
                loss = criterion(pred_m["outputs_time"], pred_t)

                epoch_train_loss.append(loss.item())  
                loss.backward()
                optimizer.step()
             
                end_time = time.time()
                
            
                batch_time = end_time - start_time
                total_batch_time += batch_time
                batch_count += 1
                
                pbar.set_postfix({"Train Loss": f"{loss.item():.6f}"})
                pbar.update(1)
                
                
        # scheduler.step()
        # 计算并打印平均batch训练时间
        avg_batch_time = total_batch_time / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}: Average Batch Time: {avg_batch_time:.4f}s")
        t_loss = np.nanmean(np.array(epoch_train_loss))  # 计算一个epoch的平均损失
        print(f'Epoch: {epoch+1}/{epochs}, training loss: {t_loss:.7f}')  # 输出每个epoch的训练损失

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            # 使用 tqdm 包裹验证数据加载器
            with tqdm(total=len(validate_data_loader), desc=f"Validating {epoch+1}/{epochs}", unit="batch") as pbar_val:
                for iteration, (pred_t, prev) in enumerate(validate_data_loader, 1):
                    pred_t, prev = pred_t.to(device), prev.to(device)

                    pred_m = model(prev, None, None, None)


                 
                    loss = criterion(pred_m["outputs_time"], pred_t)
       
                    
                    epoch_val_loss.append(loss.item()) 

               
                    pbar_val.set_postfix({"Val Loss": f"{loss.item():.6f}"})
                    pbar_val.update(1)

            v_loss = np.nanmean(np.array(epoch_val_loss))
            print(f'Epoch: {epoch+1}/{epochs}, validation loss: {v_loss:.7f}')
            
            # 保存最佳模型
            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model)


if __name__ == "__main__":
    pred_len = 4 
    prev_len = 16
    label_len = 12
    
    # for name, param in model.gpt2.named_parameters():
    #     if 'weight' in name:
    #         nn.init.normal_(param, mean=0.0, std=0.02)  # 采用 GPT-2 论文的标准初始化
    #     elif 'bias' in name:
    #         nn.init.zeros_(param)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)


    scheduler = StepLR(optimizer, step_size=250, gamma=0.5)  # 每10个epoch减少学习率的10%
    criterion = NMSELoss().to(device)

    train(training_data_loader, validate_data_loader)  # call train function (

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
