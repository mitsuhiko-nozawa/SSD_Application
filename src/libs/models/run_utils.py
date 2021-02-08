import time
import os.path as osp
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau


def run_training(model, trainloader, validloader, epochs, optimizer, scheduler, loss_fn, early_stopping_steps, verbose, device, seed, weight_path):
    
    early_step = 0
    best_loss = np.inf
    best_epoch = 0
    best_val_preds = -1
    
    start = time.time()
    t = time.time() - start
    for epoch in range(epochs):
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device)
        valid_preds = valid_fn(model, loss_fn, validloader, device)
        valid_loss = valid_preds[0]

        # scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        
        if epoch % verbose==0 or epoch==epoch_-1:
            t = time.time() - start
            print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}, time: {t}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_val_preds = valid_preds
            torch.save(model.state_dict(), osp.join( weight_path,  f"seed_{seed}.pt") )
            early_step = 0
            best_epoch = epoch
        
        elif early_stopping_steps != 0:
            early_step += 1
            if (early_step >= early_stopping_steps):
                t = time.time() - start
                print(f"early stopping in iteration {epoch},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
                return best_val_preds[1:]

    t = time.time() - start       
    print(f"training until max epoch {epochs},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
    return best_val_preds[1:]

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    s = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, (images, targets) in pbar:
        optimizer.zero_grad()
        images = images.to(device)
        targets = [ann.to(device) for ann in targets]  # リストの各要素のテンソルをGPUへ

        outputs = model(images)
        loss_l, loss_c = loss_fn(outputs, targets)
        loss = loss_l + loss_c
        loss.backward()  
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        optimizer.step()

        if i % 10 == 0: 
            description = f"iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.16f}"
            pbar.set_description(description)

        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_loc = []
    valid_conf = []
    valid_dbox = []
    s = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (images, targets) in pbar:
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            outputs = model(images)
            valid_loc.append(outputs[0].to('cpu').detach().numpy().copy())
            valid_conf.append(outputs[1].to('cpu').detach().numpy().copy())
            valid_dbox = outputs[2]

            loss_l, loss_c = loss_fn(outputs, targets)
            loss = loss_l + loss_c
            final_loss += loss.item()
            if i % 10 == 0: 
                description = f"iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.16f}"
                pbar.set_description(description)
        
    final_loss /= len(dataloader)
    valid_loc = np.concatenate(valid_loc)
    valid_conf = np.concatenate(valid_conf)    
    
    return final_loss, valid_loc, valid_conf, valid_dbox


def inference_fn(model, dataloader, device): # need debug
    model.eval()
    preds = []
    s = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, images in pbar:
            images = images.to(device)
            outputs = model(images)
            preds.append(outputs)
            if i % 10 == 0: 
                description = f"iteration {i} | time {time.time() - s:.4f}"
                pbar.set_description(description)

    preds = np.concatenate(preds)
    
    return preds