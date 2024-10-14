import nltk
import jsonlines

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeSum expects newline after each sentence
    # remove space or newline character on the beginning.
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def write_jsonl_file(output_file, id_set, pred_set):
    """This is for validation format output"""
    with jsonlines.open(output_file, 'w') as writer:
        for id, pred in zip(id_set, pred_set):
            writer.write({"title": pred, "id": id})