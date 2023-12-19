# BuLlama - Medication Information AI

As bulas de medicamentos são fontes essenciais de informações sobre remédios, sendo crucial para os pacientes lerem esses documentos. Entretanto, estudos indicam que muitos não prestam a devida atenção a essas informações. Por exemplo, uma pesquisa mostrou que apenas uma em cada quatro pessoas lê as bulas dos medicamentos que consome, e um estudo focado na população idosa brasileira revelou que cerca de 46% têm o hábito de ler sempre as bulas. No Brasil, a ANVISA é responsável pela regulamentação das bulas, que são divididas em duas categorias: bulas para pacientes e bulas para profissionais, com informações mais técnicas para estes últimos.

Recentemente, avanços significativos no processamento de linguagem natural, especialmente com o desenvolvimento de Large Language Models (LLMs), como a Llama 2, têm impulsionado o uso de chats baseados nesses modelos. Contudo, incluir informações específicas nesses modelos pode ser desafiador e enfrentar limitações como tempo e custo computacional.


## Usando o código


### Instalando as bibliotecas

Antes de executar o código, instale as bibliotecas requeridas:

```
 pip install -r requirements.txt
```

### Dataset

Para criar os datasets, baixando os dados do https://www.bulario.com/ use esse comando:

```
python create_datasets.py
```

### Treinando

Para treinar com as instruções, use esse comando:

```
python train_questions.py
```
Para treinar com o texto corrido, use esse comando:

```
python train_plaintext.py
```

### Evaluate

Para gerar as respostas de inferência, execute cada um dos comandos. Eles se referem as respostas dos treinos com as intruções, com o texto corrido, e com o modelo sem treino (vanilla):

```
python evaluate.py plain_text
python evaluate.py questions
python evaluate.py vanilla
```


## License

BuLlama is distributed under the [MIT License](LICENSE), which means you can use it for your projects, modify it, and even redistribute it, all subject to the terms of the license.

---

*Please note that BuLlama is for informational purposes only and should not be considered a substitute for professional medical advice. Always consult with a healthcare professional for any medical concerns or decisions.*
