import random
import numpy as np
import torch
import os

def seed_everything(seed=42):
    """
    Mandatory seed freezing as per IMPORTANT section in Phase 0.
    Ensures reproducibility across all models.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
