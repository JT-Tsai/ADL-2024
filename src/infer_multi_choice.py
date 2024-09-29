import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from itertools import chain
from utils_mc import parse_args, DataCollatorForMulitpleChoice
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)

def inference():
    args