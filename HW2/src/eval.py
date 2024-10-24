import json
import argparse
from tw_rouge import get_rouge

def eval(reference_path, submission_path):
    refs, preds = {}, {}

    with open(reference_path) as file:
        for line in file:
            line = json.loads(line)
            refs[line["id"]] = line["title"].strip() + '\n'
        
    with open(submission_path) as file:
        for line in file:
            line = json.loads(line)
            preds[line["id"]] = line["title"].strip() + '\n'
        
    keys = refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    print(json.dumps(get_rouge(preds, refs), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    rouge_score = eval(args.reference, args.submission)
    
    print(f"rouge-1 is {rouge_score['rouge-1']['f']}")
    print(f"rouge-2 is {rouge_score['rouge-2']['f']}")
    print(f"rouge-l is {rouge_score['rouge-l']['f']}")