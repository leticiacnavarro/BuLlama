import json
from datasets import Dataset
from datasets import DatasetDict

import pandas as pd

def generate_prompt(question, answer) -> str:
    return f""" ### Instruction: Answer the question in portuguese.

### Input: 
{question.strip()}

### Response: 
{answer}
""".strip()

def generate_prompt_2(question, answer) -> str:
    return f"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{question.strip()}
[/INST]

{answer}
""".strip()

def generate_text(data_point):
    return{
        "conversation": data_point["pergunta"],
        "summary": data_point["resposta"],
        "text": generate_prompt_2(data_point["pergunta"], data_point["resposta"])
    }

def process_dataset(data: Dataset):
    return(
        data.shuffle(seed=72)
        .map(generate_text)
        .remove_columns(
            [
                "pergunta",
                "resposta"
            ]
        )
    )

def generate_dataset():
    
    with open("bulas_question.txt", encoding="UTF8") as f:
        a = json.load(f)
    
    my_data = pd.DataFrame(data=a)
  #  subdata = my_data[5300:5350]
    subdata = my_data

    dataset = Dataset.from_pandas(subdata)
    
    test_size = 0.1
    dataset_dict = process_dataset(dataset).train_test_split(test_size=test_size, shuffle=True)
    test_valid = dataset_dict['test'].train_test_split(test_size=0.5)

    final_dataset = DatasetDict({
        'train': dataset_dict['train'],
        'test': test_valid['test'],
        'valid':test_valid['train']
    })

    return final_dataset
