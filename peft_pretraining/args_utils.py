import os
from datetime import datetime

from loguru import logger
def max_train_tokens_to_number(max_train_tokens):
    if type(max_train_tokens) is int:
        return max_train_tokens
    if max_train_tokens.endswith("M"):
        return int(float(max_train_tokens.rstrip("M")) * 1_000_000)
    elif max_train_tokens.endswith("B"):
        return int(float(max_train_tokens.rstrip("B")) * 1_000_000_000)
    else:
        return int(max_train_tokens)


def check_args_torchrun_main(args):

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.max_train_tokens=max_train_tokens_to_number(args.max_train_tokens)
        args.num_training_steps = args.max_train_tokens // (args.total_batch_size*args.max_length)
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    return args
