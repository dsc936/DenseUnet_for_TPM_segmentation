import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import os
import hdf5storage
import pywt
from scipy import misc, ndimage
import random

class Loader3D_segment(Dataset):
    def __init__(self, x_files, y_files, imsize = 144, t_slices = -1):
        super(Loader3D_segment, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices

    def crop(self, data, imsize):
        nx = data.shape[1]
        ny = data.shape[2]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return data[...,resx: nx-resx, resy: ny-resy]

    def normalize(self, data):
        nx = data.shape[1]//2
        data_center = self.crop(data,nx)
        # return data/np.amax(data_center)
        return data/np.percentile(np.abs(data_center),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        if (x.shape[0]//8)*8 < x.shape[0]:
            self.t_slices = (x.shape[0]//8)*8
        if self.t_slices != -1:
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]

        x = self.normalize(x)[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.LongTensor(y)}

class Loader3D_TPM(Dataset):
    def __init__(self, x_files, y_files, imsize = (128,96), t_slices = -1, mode = None):
        super(Loader3D_TPM, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices
        self.mode = mode

    def crop(self, data, imsize):
        nx = data.shape[-2]
        ny = data.shape[-1]
        padx = 0
        pady = 0
        if len(data.shape) == 3:
            output = np.zeros((data.shape[0],imsize,imsize))
        else:
            output = np.zeros((data.shape[0],data.shape[1],imsize,imsize))

        if nx>imsize:
            resx = (nx-imsize)//2
            data = data[...,resx: nx-resx,:]
        else:
            padx = (imsize-nx)//2
        if ny>imsize:
            resy = (ny-imsize)//2
            data = data[...,:,resy: ny-resy]
        else:
            pady = (imsize-ny)//2
        output[...,padx:imsize-padx,pady:imsize-pady] = data
        return output

    #def normalize(self, data):
    #    nx = data.shape[2]//2
    #    data_center = self.crop(data,nx)
    #    return data/np.percentile(data_center,95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        x = np.transpose(x, (3, 2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = self.crop(x, self.imsize)
        #x[0] = self.normalize(x[0])
        y = self.crop(y, self.imsize)

        if self.t_slices != -1:
            x = x[:,0:self.t_slices]
            y = y[0:self.t_slices]
        if self.mode =="mag":
            x_mag = x[0]
            x = x_mag[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.LongTensor(y)}

def extract_patches_random(full_imgs,full_masks, patch_size, N_patches):
    assert (len(full_imgs.shape)==5 and len(full_masks.shape)==4)  # [batch, channel, nt,nx,ny]
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==4)  #check the channel is 1 or 4
    assert (full_imgs.shape[2] == full_masks.shape[1] and full_imgs.shape[3] == full_masks.shape[2])
    patches = []
    patches_masks = []
    img_h = full_imgs.shape[3]  #height of the full image
    img_w = full_imgs.shape[4] #width of the full image
    # (0,0) in the center of the image
    k=0
    while k <N_patches:
        x_center = random.randint(0+int(patch_size/2),img_w-int(patch_size/2))
        y_center = random.randint(0+int(patch_size/2),img_h-int(patch_size/2))
        patch = full_imgs[:,:,:,y_center-int(patch_size/2):y_center+int(patch_size/2),x_center-int(patch_size/2):x_center+int(patch_size/2)]
        patch_mask = full_masks[:,:,y_center-int(patch_size/2):y_center+int(patch_size/2),x_center-int(patch_size/2):x_center+int(patch_size/2)]
        patches.append(patch)
        patches_masks.append(patch_mask)
        k+=1  #per full_img
    patch_x = torch.cat(patches,dim=0)
    patch_y = torch.cat(patches_masks,dim=0)
    return patch_x, patch_y       

def extract_patches_ordered(full_imgs,full_masks, patch_size, stride):
    assert (len(full_imgs.shape)==5 and len(full_masks.shape)==4)  # [batch, channel, nt,nx,ny]
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==4)  #check the channel is 1 or 4
    assert (full_imgs.shape[2] == full_masks.shape[1] and full_imgs.shape[3] == full_masks.shape[2])
    patches = []
    patches_masks = []
    img_h = full_imgs.shape[3]  #height of the full image
    img_w = full_imgs.shape[4] #width of the full image
    # (0,0) in the center of the image
    for h in range((img_h-patch_size)//stride+1):
        for w in range((img_w-patch_size)//stride+1):
            patch = full_imgs[:,:,:,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
            patch_mask = full_masks[:,:,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
            patches.append(patch)
            patches_masks.append(patch_mask)
    patch_x = torch.cat(patches,dim=0)
    patch_y = torch.cat(patches_masks,dim=0)
    return patch_x, patch_y   

def recompone_overlap(preds, img_size, stride):
    assert (len(preds.shape)==5)  #4D arrays
    patch_h = preds.shape[3]
    patch_w = preds.shape[4]
    # N_patches_h = (img_size-patch_h)//stride+1
    # N_patches_w = (img_size-patch_w)//stride+1
    # N_patches_img = N_patches_h * N_patches_w
    # assert (preds.shape[0]%N_patches_img==0)
    full_prob = torch.zeros((1,preds.shape[1],preds.shape[2],img_size,img_size))  #itialize to zero mega array with sum of Probabilities
    full_sum = torch.zeros((1,preds.shape[1],preds.shape[2],img_size,img_size))
    k = 0 #iterator over all the patches
    for h in range((img_size-patch_h)//stride+1):
        for w in range((img_size-patch_w)//stride+1):
            full_prob[:,:,:,h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=preds[k]
            full_sum[:,:,:,h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=1
            k+=1
    final_avg = full_prob/full_sum
    return final_avg

# def extract_patches_ordered(full_imgs, patch_size, stride):
#     assert (len(full_imgs.shape)==5) # [batch, channel, nt,nx,ny]
#     assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==4)  #check the channel is 1 or 4
#     img_h = full_imgs.shape[3]  #height of the full image
#     img_w = full_imgs.shape[4] #width of the full image
#     # (0,0) in the center of the image
#     assert ((img_h-patch_size)%stride==0 and (img_w-patch_size)%stride==0)
#     N_patches_img = ((img_h-patch_size)//stride+1)*((img_w-patch_size)//stride+1)  #// --> division between integers
#     patches = np.empty((N_patches_img,full_imgs.shape[1],full_imgs.shape[2],patch_size,patch_size))
#     for h in range((img_h-patch_size)//stride+1):
#         for w in range((img_w-patch_size)//stride+1):
#             patch = full_imgs[:,:,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
#             patches[iter_tot]=patch
#             iter_tot +=1  #per full_img
#     return patches  

class Loader3D_TPM_mag(Dataset):
    def __init__(self, x_files, y_files, imsize = (128,96), t_slices = -1, mode = None):
        super(Loader3D_TPM_mag, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices
        self.mode = mode

    def crop(self, data, imsize):
        nx = data.shape[-2]
        ny = data.shape[-1]
        padx = 0
        pady = 0
        if len(data.shape) == 3:
            output = np.zeros((data.shape[0],imsize,imsize))
        else:
            output = np.zeros((data.shape[0],data.shape[1],imsize,imsize))

        if nx>imsize:
            resx = (nx-imsize)//2
            data = data[...,resx: nx-resx,:]
        else:
            padx = (imsize-nx)//2
        if ny>imsize:
            resy = (ny-imsize)//2
            data = data[...,:,resy: ny-resy]
        else:
            pady = (imsize-ny)//2
        output[...,padx:imsize-padx,pady:imsize-pady] = data
        return output

    def normalize(self, data):
        nx = data.shape[2]//2
        data_center = self.crop(data,nx)
        return data/np.percentile(data_center,95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = self.crop(x, self.imsize)
        x[0] = self.normalize(x[0])
        y = self.crop(y, self.imsize)

        if self.t_slices != -1:
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]
        x = x[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.LongTensor(y)}

class Loader3D_TPM_test(Dataset):
    def __init__(self, x_files, imsize = 144, t_slices = -1, mode = None):
        super(Loader3D_TPM_test, self).__init__()
        self.x_files = x_files
        self.imsize = imsize
        self.t_slices = t_slices
        self.mode = mode

    def crop(self, data, imsize):
        nx = data.shape[-2]
        ny = data.shape[-1]
        padx = 0
        pady = 0
        if len(data.shape) == 3:
            output = np.zeros((data.shape[0],imsize,imsize))
        else:
            output = np.zeros((data.shape[0],data.shape[1],imsize,imsize))

        if nx>imsize:
            resx = (nx-imsize)//2
            data = data[...,resx: nx-resx,:]
        else:
            padx = (imsize-nx)//2
        if ny>imsize:
            resy = (ny-imsize)//2
            data = data[...,:,resy: ny-resy]
        else:
            pady = (imsize-ny)//2
        output[...,padx:imsize-padx,pady:imsize-pady] = data
        return output

    def normalize(self, data):
        nx = data.shape[2]//2
        data_center = self.crop(data,nx)
        return data/np.percentile(np.abs(data_center),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        x = np.transpose(x, (3, 2, 0, 1))
        x = self.crop(x, self.imsize)

        if self.t_slices != -1:
            x = x[:,0:self.t_slices]
        if self.mode =="mag":
            x_mag = x[0]
            x = x_mag[np.newaxis]
        return {"x": torch.FloatTensor(x)}
        
