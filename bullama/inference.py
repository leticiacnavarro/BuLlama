import torch

class QA_Llama():
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def summarize(self, model, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    def make_question(self, question: str, context = None):
        if context:
            question = self.generate_prompt_context(question, context)
        else:
            question = self.generate_prompt_2(question)

        summary = self.summarize(self.model, question)
        return summary

    def generate_prompt_2(self, question) -> str:
        return f"""[INST] <<SYS>> Você é um assistente prestativo e direto. Responda apenas o que for pedido. <</SYS>>
    {question.strip()}[/INST]""".strip()

    def generate_prompt_context(self, question, context):
        return f"""[INST] <<SYS>> Responda apenas o que for pedido, considerando o seguinte contexto.<</SYS>>
    ### Context:
    {context.strip()} 
    #### Question:
    {question.strip()}[/INST]""".strip()