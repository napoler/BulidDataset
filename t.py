from typing import List, Tuple, Dict, Any
import torch
from torch import Tensor

data=torch.tensor([[1,2,3],[4,5,6]])
def greeting(data: Tensor) -> Tensor:
    return data

print(greeting(data))