import argparse
import bullama.util
from bullama.util import create_model_tokenizer, LlamaPipeline
from bullama.inference import QA_Llama
from tqdm import tqdm
import pandas as pd




def evaluate(model_dir, file_questions, local):
    questions = pd.read_csv(file_questions, sep=';')
    questions = questions.sample(frac=0.1, random_state=1)

    model, tokenizer = create_model_tokenizer(model_dir, True, local)
    qa_llama = QA_Llama(model, tokenizer)
    questions['Resposta_predita'] = ''
    
    for index, row in tqdm(questions.iterrows(), total=questions.shape[0]):
        resposta_predita = qa_llama.make_question(questions['Pergunta'][index])
        questions.at[index, 'Resposta_predita'] = resposta_predita
    return questions

def main(type_eval):

    model = ''
    file_in = 'datasets/bulario_questions_teste.csv'
    file_out = ''
    local = True

    if type_eval == 'plain_text':
        model = "models/bulario_plaintext/"
        file_out = 'datasets/evaluate/evaluate_plaintext.csv'

    elif type_eval == 'questions':
        model = "models/bulario_questions/"
        file_out = 'datasets/evaluate/evaluate_questions.csv'

    elif type_eval == 'vanilla':
        local = False
        file_out = 'datasets/evaluate/evaluate_vanilla.csv'
        model = '7b'

    questions = evaluate(model, file_in, local)
    questions.to_csv(file_out, sep=';')

    for index, row in questions.iterrows():
        print(f"Pergunta {index}: {questions['Pergunta'][index]}")
        print(f"Resposta certa {index}: {questions['Resposta'][index]}")
        print(f"Resposta predita {index}: {questions['Resposta_predita'][index]}")
        print("\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("type_eval", help="type of evaluate",
                        type=str)
    args = parser.parse_args()
    main(args.type_eval)