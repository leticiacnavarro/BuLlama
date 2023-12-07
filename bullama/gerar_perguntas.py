import pandas as pd
import random

class GeradorDePerguntasMedicamento:
    def __init__(self, df):
        self.df = df
        # Listas de perguntas
        self.perguntas_sobre_remedio_serve = [
            "Qual é a finalidade do medicamento {}?",
            "O que {} pretende tratar?",
            "Qual é o propósito principal de {} na área da saúde?",
            "Em que situações médicas é recomendado o uso de {}?",
            "Para que serve {}?"
        ]

        self.perguntas_sobre_como_usar = [
            "Como usar {}?",
            "Quais são as instruções para o uso correto de {}?",
            "De que maneira {} deve ser administrado?",
            "Quais são as orientações específicas para utilizar {}?",
            "Qual é a posologia recomendada para {}?"
        ]

        self.perguntas_sobre_efeitos_colaterais = [
            "Quais os males que {} pode me causar?",
            "Quais são os possíveis efeitos colaterais associados a {}?",
            "Existem reações adversas conhecidas de {}?",
            "Quais são os potenciais efeitos indesejados de {}?",
            "Quais são os riscos para a saúde relacionados ao uso de {}?"
        ]

        self.perguntas_sobre_contraindicacoes = [
            "Quando não devo usar {}?",
            "Existem situações em que o uso de {} não é recomendado?",
            "Quais são as contraindicações para {}?",
            "Em que circunstâncias eu não deveria utilizar {}?",
            "Há alguma condição específica em que o uso de {} é desaconselhado?"
        ]

        self.perguntas_sobre_mecanismo_de_acao = [
            "Como funciona {}?",
            "Qual é o mecanismo de ação de {}?",
            "Quais são os processos biológicos desencadeados por {}?",
            "Como {} atua no organismo para produzir seus efeitos?",
            "Quais são os princípios do funcionamento de {}?"
        ]


    def gerar_perguntas(self):
        dados = []
        dados_2 = []
        for _, row in self.df.iterrows():
            nome = row["Nome"]
            for coluna, perguntas in zip(self.df.columns[2:], 
                                     [self.perguntas_sobre_remedio_serve, self.perguntas_sobre_como_usar, 
                                      self.perguntas_sobre_efeitos_colaterais, self.perguntas_sobre_contraindicacoes, 
                                      self.perguntas_sobre_mecanismo_de_acao]):
                perguntas = random.choices(perguntas, k=2)
                resposta = row[coluna]
                if not pd.isna(resposta):
                    dados.append({"Nome": nome, "Pergunta": perguntas[0].format(nome), "Resposta": resposta})
                    dados_2.append({"Nome": nome, "Pergunta": perguntas[1].format(nome), "Resposta": resposta})
        return pd.DataFrame(dados), pd.DataFrame(dados_2)

    def dividir_dataframe(self, proporcao=0.1):
        df_treino, df_teste = self.gerar_perguntas()
        df_teste = df_teste.sample(frac=proporcao, random_state=1)
        return df_treino, df_teste