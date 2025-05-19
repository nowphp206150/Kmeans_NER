import random
import os
import numpy as np
import torch
from functools import lru_cache
from torch.optim.lr_scheduler import LambdaLR


def seed_everything(seed=1029):
    """
    :param seed:
    :param device:
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one, num_warmup_steps_stage_two, num_training_steps_stage_two, stage_one_lr_scale, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(
                    1, num_training_steps_stage_one - num_warmup_steps_stage_one)) * stage_one_lr_scale
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            return max(
                0.0, float(num_training_steps_stage_two - current_step) /
                float(max(1, num_training_steps_stage_two -
                      num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
