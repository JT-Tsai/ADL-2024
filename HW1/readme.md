# ADL_HW1
Department: NTNU EE
Student ID: 41375007h
Name: 蔡杰宇

## Env setup
To set up the environment, ensure using Python 3.10 installed,
And can install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## code reference
[multiple choice](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py)
[question answering](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py)


## Step to run training
1. Prepare the training data:
    - Ensure the following files are placed in the ./data/ folder:
    - `./data/context.json`
    - `./data/train.json`
    - `./data/valid.json`

2. Execute the training script:
    - The training scripts `multi_choice.py` and `question_answering.py` are in `./src` folder.
    - the training script accepts useful argument such as --debug and --dynamic_save
    - the model trained on a RTX4060Ti 16GB, you need to adjust the hyperparameters from `train.sh` for customized training
    - run the command to start training:
    ```bash!
    bash ./train.sh
    ```

## Step to run inference
- the `inference_mc.py` and `inference_qa.py` are also in `./src` folder.
1. download trained model when execute inference process.
    - run the command to download model and tokenizer from google drive.
    - when executed this, it will create `output` folder to save the trained model.
    ```bash!
    bash ./download.sh
    ```
2. execute inference process
    - run the commmand to start inference.
    ```bash
    bash ./run.sh /path/to/context.json /path/to/test.json path/to/pred/prediction.csv
    ```
