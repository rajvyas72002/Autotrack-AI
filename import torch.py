import torch
print(torch.version.cuda)   # should show 12.1
print(torch.cuda.is_available())  # should be True
