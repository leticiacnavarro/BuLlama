import transformers
import torch
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
from torch import float32, nn
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model

import bullama.rag
access_token = "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"


def get_lora_config():
    lora_config = LoraConfig(r = 256, # attention heads
                    lora_alpha = 512, # alpha scaling
                    lora_dropout = 0.05,
                    bias = "all",
                    task_type = "CAUSAL_LM",
                     # set this for CLM or Seq2Seq
    )
    return lora_config

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(float32)

def prepare_model(model):
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(float32)
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    return model

def create_model_tokenizer(model_name, quantizer, local):

    if local:     
        model_id = model_name   
    else:
        model_id = f"meta-llama/Llama-2-{model_name}-chat-hf"

    model = get_model(model_id, quantizer)
    tokenizer = get_tokenizer(model_id=model_id)
    return model, tokenizer

def get_quantizer_4bit():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
def get_quantizer_8bit():
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )    

def get_model(model_id, quantizer):

    if(quantizer):
        quantizer_cfg = get_quantizer_8bit()
    else:
        quantizer_cfg = None


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantizer_cfg,
        token=access_token
    )

    return model     
    
def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
  #  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def get_llama_pipeline(model, tokenizer):
    llama_pipeline = pipeline(
        "text-generation",  # LLM task
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=access_token,
        tokenizer=tokenizer
    )
    return llama_pipeline

def get_response(prompt: str, llama_pipeline, tokenizer) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    with torch.autocast("cuda"):

      sequences = llama_pipeline(
          prompt,
          do_sample=True,
          top_k=10,
          num_return_sequences=2,
          eos_token_id=tokenizer.eos_token_id,
          max_length=512,
      )
      print("Chatbot:", sequences[0]['generated_text'])

class LlamaPipeline():
    def __init__(self, model, tokenizer):
        self.model = model,
        self.tokenizer = tokenizer
        self.pipeline = get_llama_pipeline(model, tokenizer)

    def get_response(self, prompt):
        with torch.autocast("cuda"):

            sequences = self.pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=256,
            )
            print("Chatbot:", sequences[0]['generated_text'])        

class LlamaQA():
    def __init__(self, model: AutoModelForCausalLM, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def summarize(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    def make_question(self, question: str):
        question = self.generate_prompt(question)
        summary = self.summarize(question)
        print(summary)

    def generate_prompt(self, question) -> str:
        return f"""[INST] <<SYS>> Você é um assistente que irá responder perguntas sobre remédios. Responda apenas o que for pedido.<</SYS>>
    {question.strip()}[/INST]""".strip()

class LlamaQA_rag():
    def __init__(self, model: AutoModelForCausalLM, tokenizer, file):
        self.model = model
        self.tokenizer = tokenizer
        self.rag = bullama.rag.RAG(file)


    def summarize(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    def make_question(self, question: str):
        context = self.rag.get_doc(question)
        question = self.generate_prompt(question, context)
        summary = self.summarize(question)
        print(summary)

    def generate_prompt(self, question, context) -> str:
        return f"""[INST] <<SYS>> Você é um assistente que irá responder perguntas sobre remédios. Responda apenas o que for pedido.<</SYS>>
        Usando o seguinte contexto:
        {context.strip()}
    {question.strip()}[/INST]""".strip()