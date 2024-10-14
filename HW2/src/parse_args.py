import argparse
from transformers import SchedulerType, MODEL_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument( "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use",
    )
    parser.add_argument("--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use",
    )
    parser.add_argument("--jsonl_data_file", 
        type=str, default=None, help = "use this when training data and validation in the same jsonl file"
    )
    parser.add_argument("--train_file", 
        type=str, default=None, help="A csv or json file containing the training data.",
    )
    parser.add_argument("--validation_file", 
        type=str, default=None, help="A csv or a json file containing the validation data.",
    )
    parser.add_argument("--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument("--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument("--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument("--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--overwrite_cache", 
        action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument("--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenizeation.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`. This argument is also used to override the ``max_length_`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument("--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument("--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument("--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument("--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization)",
    )
    parser.add_argument("--summary_column",
        type=str, default=None, help="The name of the column in the datasets containing the summaries or titles"    
    )
    parser.add_argument("--use_slow_tokenizer",
        action="store_true",
        help="if passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument("--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader."
    )
    parser.add_argument("--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the petential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", 
        type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", 
        type=int, default=3, help="Total number of training epochs to perform")
    parser.add_argument("--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provide, overrides num_train_epochs."
    )
    parser.add_argument("--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumalate before performing a backward/update pass."
    )
    parser.add_argument("--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", 
        type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", 
        type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--output_file",
        type=str, default=None, help="the path to store the final output, which can replace --output_dir")
    parser.add_argument("--seed", 
        type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch",
        choices = MODEL_TYPES,
    )
    parser.add_argument("--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--report_to",
        type=str,
        default="all",
        help=(
            "The integration to report the results and logs to."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument("--split_rate", type=float, default=0.1, help="the ratio of the training to testing data")
    parser.add_argument("--debug", action="store_true", help="using this to test code")

    parser.add_argument("--do_sample", action="store_true", help="decoder algorithm")
    parser.add_argument("--top_k", type=float, default=None, help="when --do_sample passed, this parameter works")
    parser.add_argument("--top_p", type=float, default=None, help="when --do_sample passed, this parameter works")
    parser.add_argument("--temperature", type=float, default=None, help="this parameter control generate diversity")

    # parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    # parser.add_argument(
    #     "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    # )
    # parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    args = parser.parse_args()

    if args.dataset_name is None and args.jsonl_data_file is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extention = args.train_file.split(".")[-1]
    #         assert extention in ["csv", "json"], "`train_file` should be a csv or json file."
    #     if args.validation is not None:
    #         extention = args.validation_file.split(".")[-1]
    #         assert extention in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args