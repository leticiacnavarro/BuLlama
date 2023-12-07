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
    return f"""[INST] <<SYS>>Você é um assistente que responde perguntas sobre remédios. Responda o que for pedido.<</SYS>>
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

def generate_dataset(file, split):
    if(file.endswith('.csv')):
        data_frame = pd.read_csv(file, sep=';').dropna()
        dataset = Dataset.from_pandas(data_frame)
        return dataset
    
    if(file.endswith('.txt')):

        with open(file, encoding="UTF8") as f:
            a = json.load(f)
        
        data_frame = pd.DataFrame(data=a).dropna()
    dataset = Dataset.from_pandas(data_frame)
    if split:
        test_size = 0.1
        dataset_dict = process_dataset(dataset).train_test_split(test_size=test_size, shuffle=True)
        test_valid = dataset_dict['test'].train_test_split(test_size=0.5)

        final_dataset = DatasetDict({
            'train': dataset_dict['train'],
            'test': test_valid['test'],
            'valid':test_valid['train']
        })

        return final_dataset
    else:
        return process_dataset(dataset)
