import bullama.util
from bullama.util import create_model_tokenizer
from datasets import Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import Callable, Mapping
from trl import SFTTrainer
import dataset_prompt

def main():

    model_name = '7b'
    output_dir = 'OUTPUTS'
    model_dir = 'models'
    exp_name = 'bulario_questions'

    model, tokenizer = create_model_tokenizer(model_name, True, False)

    dataset = dataset_prompt.generate_dataset("datasets/bulario_prompts_treino.csv", split = False)

    model = bullama.util.prepare_model(model)

    training_args = TrainingArguments(
        per_device_train_batch_size = 1, #The batch size per GPU (default: 8)
        gradient_accumulation_steps = 1,
        warmup_steps = 0, # Number of steps used for a linear warmup from 0 to learning_rate (default: 0)
        num_train_epochs=3,
    # weight_decay=0.1,
        learning_rate = 2e-4,
        fp16 = True, # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        logging_steps = 1, # Number of update steps between two logs if logging_strategy="steps". Should be an integer or a float in range [0,1). 
        #If smaller than 1, will be interpreted as ratio of total training steps.
        overwrite_output_dir = True,
    # evaluation_strategy = "epoch", #"The evaluation strategy to adopt during training. "
    # save_strategy = "no",
        push_to_hub = False,
        output_dir = output_dir,
        report_to="tensorboard",
        optim="paged_adamw_8bit",
    )

    trainer_sft =  SFTTrainer(
        model = model,
        train_dataset=dataset,
    #    eval_dataset=dataset['valid'],
        peft_config = bullama.util.get_lora_config(),
        dataset_text_field = "text",
        max_seq_length = 2048,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer_sft.train()
    trainer_sft.save_model(f"{model_dir}/{exp_name}")
    tokenizer.save_pretrained(f"{model_dir}/{exp_name}")


if __name__ == "__main__":
    main()