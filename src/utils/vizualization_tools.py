import matplotlib.pyplot as plt
import numpy as np

def add_center_window(a, window_size=64):
    b = np.copy(a)
    start = int(a.shape[0] / 2 - window_size / 2)
    end   = int(a.shape[0] / 2 + window_size / 2)
    
    b[start:end, start] = 1
    b[start:end, end]   = 1
    b[start, start:end] = 1
    b[end, start:end]   = 1
    
    return b

def center_window_std(a, window_size=64):
    start = int(a.shape[0] / 2 - window_size / 2)
    end   = int(a.shape[0] / 2 + window_size / 2)
    
    return np.std(a[start:end, start:end])

def plot2(a, b, view='axial', slc_i=128, title=[], img_padding=True):
    figure, axis = plt.subplots(1, 2, figsize=(20, 14))
    
    a_, b_ = a, b
    
    if 1 not in a.shape and a.ndim == 3: # check that images are actually 3D
        if view == 'axial':
            a_, b_ = a_[slc_i,:,:], b_[slc_i,:,:]
        elif view == 'coronal':
            a_, b_ = a_[:,slc_i,:], b_[:,slc_i,:]
        elif view == 'sagittal':
            a_, b_ = a_[:,:,slc_i], b_[:,:,slc_i]
    
    axis[0].imshow(a_, cmap='gray')
    axis[0].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(a_), np.max(a_), np.mean(a_), np.std(a_)))
    axis[1].imshow(b_, cmap='gray')
    axis[1].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(b_), np.max(b_), np.mean(b_), np.std(b_)))
    
    if type(title) == list and len(title) > 0:
        axis[0].set_title(title[0])
        axis[1].set_title(title[1])
    else:
        axis[0].set_title(title)
        axis[1].set_title(title)
        
    if not img_padding:
        axis[0].axis('off')
        axis[1].axis('off')
        plt.subplots_adjust(0,0,1,1)
        plt.tight_layout(pad=0.00)
    

def plot4(a, view='2D', title=[], slc_i=128):
    figure, axis = plt.subplots(1, 4, figsize=(30, 20))
    
    if type(a) is not list:
        if view == 'axial':
            a = a[slc_i,:,:]
        elif view == 'coronal':
            a = a[:,slc_i,:]
        elif view == 'sagittal':
            a = a[:,:,slc_i]

        vol = np.real(a)
        axis[0].imshow(vol, cmap='gray')
        axis[0].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(vol), np.max(vol), np.mean(vol), np.std(vol)))
        axis[0].set_title(title)

        vol = np.imag(a)
        axis[1].imshow(vol, cmap='gray')
        axis[1].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(vol), np.max(vol), np.mean(vol), np.std(vol)))
        axis[1].set_title(title)

        vol = np.abs(a)
        axis[2].imshow(vol, cmap='gray')
        axis[2].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(vol), np.max(vol), np.mean(vol), np.std(vol)))
        axis[2].set_title(title)

        vol = np.angle(a)
        axis[3].imshow(vol, cmap='gray')
        axis[3].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(vol), np.max(vol), np.mean(vol), np.std(vol)))
        axis[3].set_title(title)
    else:
        if view == 'axial':
            a[0] = a[0][slc_i,:,:]
            a[1] = a[1][slc_i,:,:]
            a[2] = a[2][slc_i,:,:]
            a[3] = a[3][slc_i,:,:]
        elif view == 'coronal':
            a[0] = a[0][:,slc_i,:]
            a[1] = a[1][:,slc_i,:]
            a[2] = a[2][:,slc_i,:]
            a[3] = a[3][:,slc_i,:]
        elif view == 'sagittal':
            a[0] = a[0][:,:,slc_i]
            a[1] = a[1][:,:,slc_i]
            a[2] = a[2][:,:,slc_i]
            a[3] = a[3][:,:,slc_i]
            
        for i in range(4):
            vol = a[i]
            axis[i].imshow(vol, cmap='gray')
            axis[i].set(xlabel='slc min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(vol), np.max(vol), np.mean(vol), np.std(vol)))
            if type(title) == list:
                axis[i].set_title(title[i])
            else:
                axis[i].set_title(title)

def plot_patches(X, img_i=0):
    '''
    plot first row * col patches which form a single image
    '''
    rows = 2
    cols = 4

    figure, axis = plt.subplots(rows, cols, figsize=(5 * cols,7 * rows))
    print(f'Showing {rows * cols}/{len(X)} patches')
    for i in range(rows):
        for j in range(cols):
            slc = X[img_i]
            axis[i, j].imshow(slc, cmap='gray')
            label = 'min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(slc), np.max(slc), np.mean(slc), np.std(slc))
            axis[i, j].set(xlabel=label)
            axis[i, j].set_title(f'Patch {img_i}')
            img_i += 1

    plt.show()
    
def plot_slices(X, slc_i=0):
    '''
    plot first row * col full slices
    '''

    rows = 2
    cols = 3

    figure, axis = plt.subplots(rows, cols, figsize=(30,15))
    for i in range(rows):
        for j in range(cols):
            slc = X[slc_i]
            axis[i, j].imshow(slc, cmap='gray')
            label = 'min: {:.3f}, max: {:.3f}, mean: {:.3f}, std: {:.3f}'.format(np.min(slc), np.max(slc), np.mean(slc), np.std(slc))
            axis[i, j].set(xlabel=label)
            slc_i += 1