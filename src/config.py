import random
import numpy as np
import torch

# globalny random state
GLOBAL_SEED = 42

def set_seed(seed=GLOBAL_SEED):
    # stałe ziarno losowości dla wszystkich bibliotek
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
