import torch


def preferred_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def device_description(device=None):
    device = preferred_device() if device is None else str(device)
    if device == 'cuda' and torch.cuda.is_available():
        return f'cuda ({torch.cuda.get_device_name(0)})'
    return device
