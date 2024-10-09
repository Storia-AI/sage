#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning a Code LLM on Custom Code on a single GPU
# 
# _Authored by: [Maria Khalusova](https://github.com/MKhalusova)_
# 
# Publicly available code LLMs such as Codex, StarCoder, and Code Llama are great at generating code that adheres to general programming principles and syntax, but they may not align with an organization's internal conventions, or be aware of proprietary libraries.
# 
# In this notebook, we'll see show how you can fine-tune a code LLM on private code bases to enhance its contextual awareness and improve a model's usefulness to your organization's needs. Since the code LLMs are quite large, fine-tuning them in a traditional manner can be resource-draining. Worry not! We will show how you can optimize fine-tuning to fit on a single GPU.
# 
# 
# ## Dataset
# 
# For this example, we picked the top 10 Hugging Face public repositories on GitHub. We have excluded non-code files from the data, such as images, audio files, presentations, and so on. For Jupyter notebooks, we've kept only cells containing code. The resulting code is stored as a dataset that you can find on the Hugging Face Hub under [`smangrul/hf-stack-v1`](https://huggingface.co/datasets/smangrul/hf-stack-v1). It contains repo id, file path, and file content.
# 
# 
# ## Model
# 
# We'll finetune [`bigcode/starcoderbase-1b`](https://huggingface.co/bigcode/starcoderbase-1b), which is a 1B parameter model trained on 80+ programming languages. This is a gated model, so if you plan to run this notebook with this exact model, you'll need to gain access to it on the model's page. Log in to your Hugging Face account to do so:


import argparse
import os
import flash_attn 
import argparse
import torch
import functools
import numpy as np
import random

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


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bigcode/starcoderbase-1b", help="Model checkpoint on the Hugging Face Hub")
parser.add_argument("--dataset", type=str, default="smangrul/hf-stack-v1", help="Dataset on the Hugging Face Hub")
parser.add_argument("--data_column", type=str, default="content", help="Column name containing the code content")
parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
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
args = parser.parse_args()

run_name = args.run_name


def login_to_huggingface_hub(token):
    """
    Log in to the Hugging Face Hub programmatically.
    
    :param token: Your Hugging Face API token
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


login_to_huggingface_hub(args.hf_token)        
set_seed(SEED)

# Wandb configuration
import wandb
wandb.login()


dataset = load_dataset(
    DATASET,
    data_dir="data",
    split="train",
    streaming=True,
)

valid_data = dataset.take(400)
train_data = dataset.skip(400).take(400)
train_data = train_data.shuffle(buffer_size=5000, seed=SEED)



train_data.info
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)



chars_per_token = chars_token_ratio(train_data, tokenizer, DATA_COLUMN)
# print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")





# Helper function to get token ids of the special tokens for prefix, suffix and middle for FIM transformations.
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


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
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


# Let's define the `ConstantLengthDataset`, an Iterable dataset that will return constant-length chunks of tokens. To do so, we'll read a buffer of text from the original dataset until we hit the size limits and then apply tokenizer to convert the raw text into tokenized inputs. Optionally, we'll perform FIM transformations on some sequences (the proportion of sequences affected is controlled by `fim_rate`).
# Once defined, we can create instances of the `ConstantLengthDataset` from both training and validation data.





# Create an Iterable dataset that returns constant-length chunks of tokens from a stream of text files.

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
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

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


train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
)
eval_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
)

# Dump dataset objs to disk
import pickle
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

with open('eval_dataset.pkl', 'wb') as f:
    pickle.dump(eval_dataset, f)


# ## Prepare the model

# Now that the data is prepared, it's time to load the model! We're going to load the quantized version of the model.
# 
# This will allow us to reduce memory usage, as quantization represents data with fewer bits. We'll use the `bitsandbytes` library to quantize the model, as it has a nice integration with `transformers`. All we need to do is define a `bitsandbytes` config, and then use it when loading the model.
# 
# There are different variants of 4bit quantization, but generally, we recommend using NF4 quantization for better performance (`bnb_4bit_quant_type="nf4"`).
# 
# The `bnb_4bit_use_double_quant` option adds a second quantization after the first one to save an additional 0.4 bits per parameter.
# 
# To learn more about quantization, check out the ["Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA" blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).
# 
# Once defined, pass the config to the `from_pretrained` method to load the quantized version of the model.

# In[ ]:


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

load_in_8bit = False

# 4-bit quantization
compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)

device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=True,
)


# When using a quantized model for training, you need to call the `prepare_model_for_kbit_training()` function to preprocess the quantized model for training.


model = prepare_model_for_kbit_training(model)


# Now that the quantized model is ready, we can set up a LoRA configuration. LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.
# 
# To train a model using LoRA technique, we need to wrap the base model as a `PeftModel`. This involves definign LoRA configuration with `LoraConfig`, and wrapping the original model with `get_peft_model()` using the `LoraConfig`.
# 
# To learn more about LoRA and its parameters, refer to [PEFT documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).



# Set up lora
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES.split(","),
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# As you can see, by applying LoRA technique we will now need to train less than 1% of the parameters.

# ## Train the model

# Now that we have prepared the data, and optimized the model, we are ready to bring everything together to start the training.
# 
# To instantiate a `Trainer`, you need to define the training configuration. The most important is the `TrainingArguments`, which is a class that contains all the attributes to configure the training.
# 
# These are similar to any other kind of model training you may run, so we won't go into detail here.

# In[ ]:


train_data.start_iteration = 0


training_args = TrainingArguments(
    output_dir=f"runs/{run_name}/{OUTPUT_DIR}",
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    max_steps=MAX_STEPS,
    eval_steps=EVAL_FREQ,
    save_steps=SAVE_FREQ,
    logging_steps=LOG_FREQ,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=NUM_WARMUP_STEPS,
    gradient_accumulation_steps=GR_ACC_STEPS,
    gradient_checkpointing=True,
    fp16=FP16,
    bf16=BF16,
    weight_decay=WEIGHT_DECAY,
    push_to_hub=True,
    include_tokens_per_second=True,
    run_name=f"runs-{run_name}",
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


# ## Inference
# 
# Once the model is uploaded to Hub, we can use it for inference. To do so we first initialize the original base model and its tokenizer. Next, we need to merge the fine-duned weights with the base model.

# In[ ]:


from peft import PeftModel
import torch

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


# Now we can use the merged model for inference. For convenience, we'll define a `get_code_completion` - feel free to experiment with text generation parameters!




# Now all we need to do to get code completion is call the `get_code_complete` function and pass the first few lines that we want to be completed as a prefix, and an empty string as a suffix.


# While it is Python syntax, you can see that the original model has no understanding of what a `LoraConfig` should be doing.

# To learn how this kind of fine-tuning compares to full fine-tuning, and how to use a model like this as your copilot in VS Code via Inference Endpoints, or locally, check out the ["Personal Copilot: Train Your Own Coding Assistant" blog post](https://huggingface.co/blog/personal-copilot). This notebook complements the original blog post.
# 
