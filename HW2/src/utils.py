import nltk
import jsonlines
from filelock import FileLock
from transformers.utils import is_offline_mode
import matplotlib.pyplot as plt

# import ipdb

# download nltk package
try:
    nltk.data.find("tokenizer/punkt")
except(LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first tot download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeSum expects newline after each sentence
    # remove space or newline character on the beginning.
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    # ipdb.set_trace()

    return preds, labels

def write_jsonl_file(output_file, id_set, pred_set):
    """This is for validation format output"""
    with jsonlines.open(output_file, 'w') as writer:
        for id, pred in zip(id_set, pred_set):
            writer.write({"title": pred, "id": id})

def plot()