from transformers import AutoTokenizer
from transformers import TextGenerationPipeline, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("jed351/gpt2_base_zh-hk-lihkg")
model = AutoModelForCausalLM.from_pretrained("jed351/gpt2_base_zh-hk-lihkg")
# try messing around with the parameters
generator = TextGenerationPipeline(model, tokenizer,
                                   max_new_tokens=200,
                                   no_repeat_ngram_size=3, device="cuda:0")
function="扩写200字:"
input_string = function+"科学家发现新型太阳能电池原材料 新型太阳能电池原材料研究论文发表 国际研究团队发现新型太阳能电池原材料 新型太阳能电池原材料的发现 新型太阳能电池原材料将在更多领域得到广泛应用和推广"
output = generator(input_string)
print(output)
string = output[0]['generated_text'].replace(' ', '')
print(string)