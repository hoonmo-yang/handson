from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "tex-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
)

hf = HuggingFacePipeline.from_model_id(pipeline=pipe)

print(hf.invoke("한국의 수도는 어디야"))
