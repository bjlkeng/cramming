import torch


def dump_gpu_info():
    available = torch.cuda.is_available()
    curr_device = torch.cuda.current_device()
    device = torch.device("cuda:0" if available else "cpu") 
    device_count = torch.cuda.device_count() 
    device_name =  torch.cuda.get_device_name(0)
    
    print(f'Cuda available: {available}')
    print(f'Current device: {curr_device}')
    print(f'Device: {device}')
    print(f'Device count: {device_count}')
    print(f'Device name: {device_name}')