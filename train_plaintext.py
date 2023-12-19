import bullama.util
from bullama.util import create_model_tokenizer
from datasets import Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import Callable, Mapping
from trl import SFTTrainer

def main():

    model_name = '7b'
    output_dir = 'OUTPUTS'
    model_dir = 'models'
    exp_name = 'bulario_plaintext'

    model, tokenizer = create_model_tokenizer(model_name, True, False)
    bulario_df = pd.read_csv('datasets/bulario_plain_text.csv', delimiter=';')
    bulario_df = bulario_df.replace('\n','', regex=True)
    bulario_df = bulario_df.replace('\u200b','', regex=True)


    #dataset = Dataset.from_dict({'text': [''.join(lista)]})
    dataset = Dataset.from_pandas(bulario_df)

    model = bullama.util.prepare_model(model)

    def tokenize(element: Mapping, tokenizer: Callable, context_length: int) -> str:
        inputs = tokenizer(element['texto'], truncation=True, return_overflowing_tokens=True, 
                        return_length=True, max_length=context_length)
        inputs_batch = []
        for length, input_ids in zip(inputs['length'], inputs['input_ids']):
            if length == context_length: # We drop the last input_ids that are shorter than max_length
                inputs_batch.append(input_ids)
        return {"input_ids": inputs_batch}

    tokenized_dataset = dataset.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer, 'context_length': 2048},
                                            remove_columns=dataset.column_names)


    training_args = TrainingArguments(

        per_device_train_batch_size = 1, #The batch size per GPU (default: 8)
        gradient_accumulation_steps = 1,
        warmup_steps = 0, # Number of steps used for a linear warmup from 0 to learning_rate (default: 0)
        num_train_epochs=3,
    # weight_decay=0.1,
        learning_rate = 3e-4,
        #fp16 = True, # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        logging_steps = 1, # Number of update steps between two logs if logging_strategy="steps". Should be an integer or a float in range [0,1). 
        #If smaller than 1, will be interpreted as ratio of total training steps.
        overwrite_output_dir = True,
    # evaluation_strategy = "epoch", #"The evaluation strategy to adopt during training. "
    # save_strategy = "no",
        push_to_hub = False,
        output_dir = output_dir,
        report_to="tensorboard",
    # optim="paged_adamw_8bit",
    )

    trainer_sft =  SFTTrainer(
        model = model,
        train_dataset=dataset,
    #    eval_dataset=dataset['valid'],
        peft_config = bullama.util.get_lora_config(),
        dataset_text_field = "texto",
        max_seq_length = 1024,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer_sft.train()
    trainer_sft.save_model(f"{model_dir}/{exp_name}")
    tokenizer.save_pretrained(f"{model_dir}/{exp_name}")


if __name__ == "__main__":
    main()