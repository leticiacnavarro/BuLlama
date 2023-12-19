from bullama.gerar_perguntas import GeradorDePerguntasMedicamento
from bullama.bulario import Bulario, BularioType
import pandas as pd

def generate_prompt(question, context, answer) -> str:
    return f"""[INST] <<SYS>> Responda apenas o que for pedido, usando o contexto:<</SYS>>    
    Context: {context.strip()}
    Question: {question.strip()}[/INST]
    Answer: {answer.strip()}""".strip()

def main():
    bulas_questions = Bulario(BularioType.Questions)
    bulas_questions.create_data()

    bulas_questions.save_csv("datasets/bulario_questions.csv")

    bulas_plain_text = Bulario(BularioType.PlainText)
    bulas_plain_text.create_data()

    bulas_plain_text.save_csv("datasets/bulario_plain_text.csv")

    bulario_questions_df = pd.read_csv('datasets/bulario_questions.csv', delimiter=';')
    bulario_plaintext_df = pd.read_csv('datasets/bulario_plain_text.csv', delimiter=';')

    gerador = GeradorDePerguntasMedicamento(bulario_questions_df)
    df_treino, df_teste = gerador.dividir_dataframe()

    df_treino.to_csv('datasets/bulario_questions_treino.csv', sep=';', index=False)
    df_teste.to_csv('datasets/bulario_questions_teste.csv', sep=';', index=False)

    # Carregar os dataframes
    bulario_df = pd.read_csv('datasets/bulario_questions_treino.csv', delimiter=';')
    bulario_plaintext_df = pd.read_csv('datasets/bulario_plain_text.csv', delimiter=';')

    # Criar uma lista de prompts
    prompts = []

    for index, row in bulario_df.iterrows():
    # Iterar sobre as colunas do dataframe bulario_df
        
        contexto = bulario_plaintext_df.loc[bulario_plaintext_df['Nome'] == bulario_df['Nome'][index], 'texto'].values
        if len(contexto) > 0:
            prompt = generate_prompt(bulario_df['Pergunta'][index], contexto[0].replace('\n',''), bulario_df['Resposta'][index])
            prompts.append(prompt)

    # Criar um novo dataframe com a coluna 'text'
    prompts_df = pd.DataFrame({'text': prompts})

    # Salvar o dataframe em um arquivo CSV com delimitador ;
    prompts_df.to_csv('datasets/bulario_prompts_treino.csv', sep=';', index=False)

if __name__ == "__main__":
    main()