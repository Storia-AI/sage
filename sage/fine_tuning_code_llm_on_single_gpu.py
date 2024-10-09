import argparse
import os
import flash_attn
import functools
import numpy as np
import random
import pickle
import torch
import wandb

from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder
from huggingface_hub import notebook_login
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from peft import PeftModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

"""
Adapted from https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bigcode/starcoderbase-1b", help="Model checkpoint on the Hugging Face Hub")
parser.add_argument("--dataset", type=str, default="smangrul/hf-stack-v1", help="Dataset on the Hugging Face Hub")
parser.add_argument("--data_column", type=str, default="content", help="Column name containing the code content")
parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--gr_acc_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--num_warmup_steps", type=int, default=30, help="Number of warmup steps")
parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency")
parser.add_argument("--save_freq", type=int, default=100, help="Save frequency")
parser.add_argument("--log_freq", type=int, default=25, help="Logging frequency")
parser.add_argument("--output_dir", type=str, default="peft-starcoder-lora-a100", help="Output directory")
parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
parser.add_argument("--no_fp16", dest="fp16", action="store_false", help="Disable fp16 precision")
parser.add_argument("--fim_rate", type=float, default=0.5, help="FIM rate")
parser.add_argument("--fim_spm_rate", type=float, default=0.5, help="FIM SPM rate")
parser.add_argument("--lora_r", type=int, default=8, help="LoRA r")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
parser.add_argument("--lora_target_modules", type=str, default="c_proj,c_attn,q_attn,c_fc,c_proj", help="LoRA target modules")
parser.add_argument("--use_nested_quant", action="store_true", help="Use nested quantization")
parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="bitsandbytes 4bit compute dtype")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument('--wandb_key', type=str, required=True, help='wandb api key')
parser.add_argument('--hf_token', type=str, required=True, help='huggingface hub token')
parser.add_argument('--run_name', type=str, default='starcoderbase-1b-finetuned', help='run name for wandb')
parser.add_argument('--num_valid_datapoints', type=int, default=1000, help='number of datapoints to use for validation split')
args = parser.parse_args()


def login_to_huggingface_hub(token):
    """
    Log in to the Hugging Face Hub programmatically.
    """
    # Set the token as an environment variable
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    # Set the token in the Hugging Face folder
    HfFolder.save_token(token)
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Verify the token by trying to get user info
    try:
        user_info = api.whoami()
        print(f"Successfully logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Login failed: {str(e)}")


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def get_code_completion(prefix, suffix):
    text = prompt = f"""<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"""
    model.eval()
    outputs = model.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def load_and_test_generation():
    # load the original model first
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=None,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()

    # merge fine-tuned weights with the base model
    peft_model_id = f"runs/{run_name}/{OUTPUT_DIR}"
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.merge_and_unload()
    
    
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id

def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=args.seq_length,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = args.seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = args.seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = args.fim_rate
        self.fim_spm_rate = args.fim_spm_rate
        self.seed = args.seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

if __name__ == "__main__":
    login_to_huggingface_hub(args.hf_token)        
    set_seed(args.seed)
    run_name = args.run_name
    wandb.login(key=args.wandb_key)

    dataset = load_dataset(
        args.dataset,
        data_dir="data",
        split="train",
        streaming=True,
    )

    # Since this is an iterable dataset, we reserve a fixed # of datapoints for the valid split
    valid_data = dataset.take(args.num_valid_datapoints)
    train_data = dataset.skip(args.num_valid_datapoints)
    
    train_data = train_data.shuffle(buffer_size=5000, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)

    train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            content_field=args.data_column,
            fim_rate=args.fim_rate,
            fim_spm_rate=args.fim_spm_rate,
            seed=args.seed,
    )
    eval_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            content_field=args.data_column,
            fim_rate=args.fim_rate,
            fim_spm_rate=args.fim_spm_rate,
            seed=args.seed,
    )

    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('eval_dataset.pkl', 'wb') as f:
        pickle.dump(eval_dataset, f)

    # 4-bit quantization
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=False,
            quantization_config=bnb_config,
            device_map=device_map,
            use_cache=False,  # We will be using gradient checkpointing
            trust_remote_code=True,
            use_flash_attention_2=True,
    )

    # When using a quantized model for training, you need to call the `prepare_model_for_kbit_training()` function to preprocess the quantized model for training.
    model = prepare_model_for_kbit_training(model)

    # Set up lora
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_data.start_iteration = 0
    training_args = TrainingArguments(
        output_dir=f"runs/{args.run_name}/{args.output_dir}",
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gr_acc_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        push_to_hub=True,
        include_tokens_per_second=True,
        run_name=f"runs-{args.run_name}",
        report_to="wandb"
    )


    # As a final step, instantiate the `Trainer` and call the `train` method.   
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )

    print("Training...")
    trainer.train()


    # Finally, you can push the fine-tuned model to your Hub repository to share with your team.
    trainer.push_to_hub()
    wandb.finish()
    
    # TODO (mihail): Add appropriate eval metrics
