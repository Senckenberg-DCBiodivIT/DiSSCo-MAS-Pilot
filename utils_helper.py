import gc
import torch
import numpy as np
from torchvision.transforms import v2 as T



def get_random_color():
    return tuple(np.random.randint(0, 256, 3))

def clear_cuda():
    torch.cuda.empty_cache()

def garbage_collect():
    gc.collect()

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, device, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def load_model_checkpoint(checkpoint_path, model, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated()
    print(f"Allocated GPU memory: {allocated / 1024**2:.2f} MB")
