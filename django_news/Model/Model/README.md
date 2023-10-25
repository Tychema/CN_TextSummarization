---
license: cc-by-nc-sa-4.0
language:
- zh
pipeline_tag: summarization
tags:
- mT5
- summarization
---

# HeackMT5-ZhSum100k: A Summarization Model for Chinese Texts

This model, `heack/HeackMT5-ZhSum100k`, is a fine-tuned mT5 model for Chinese text summarization tasks. It was trained on a diverse set of Chinese datasets and is able to generate coherent and concise summaries for a wide range of texts.

## Model Details

- Model: mT5
- Language: Chinese
- Training data: Mainly Chinese Financial News Sources, NO BBC or CNN source. Training data contains 100k lines.
- Finetuning epochs: 10

## Evaluation Results

The model achieved the following results:

- ROUGE-1: 56.46
- ROUGE-2: 45.81
- ROUGE-L: 52.98
- ROUGE-Lsum: 20.22

## Usage

Here is how you can use this model for text summarization:

```python
from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("heack/HeackMT5-ZhSum100k")
tokenizer = T5Tokenizer.from_pretrained("heack/HeackMT5-ZhSum100k")

chunk = """
财联社5月22日讯，据平安包头微信公众号消息，近日，包头警方发布一起利用人工智能（AI）实施电信诈骗的典型案例，福州市某科技公司法人代表郭先生10分钟内被骗430万元。
4月20日中午，郭先生的好友突然通过微信视频联系他，自己的朋友在外地竞标，需要430万保证金，且需要公对公账户过账，想要借郭先生公司的账户走账。
基于对好友的信任，加上已经视频聊天核实了身份，郭先生没有核实钱款是否到账，就分两笔把430万转到了好友朋友的银行卡上。郭先生拨打好友电话，才知道被骗。骗子通过智能AI换脸和拟声技术，佯装好友对他实施了诈骗。
值得注意的是，骗子并没有使用一个仿真的好友微信添加郭先生为好友，而是直接用好友微信发起视频聊天，这也是郭先生被骗的原因之一。骗子极有可能通过技术手段盗用了郭先生好友的微信。幸运的是，接到报警后，福州、包头两地警银迅速启动止付机制，成功止付拦截336.84万元，但仍有93.16万元被转移，目前正在全力追缴中。
"""
inputs = tokenizer.encode("summarize: " + chunk, return_tensors='pt', max_length=512, truncation=True)
summary_ids = model.generate(inputs, max_length=150, num_beams=4, length_penalty=1.5, no_repeat_ngram_size=2)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)

包头警方发布一起利用AI实施电信诈骗典型案例:法人代表10分钟内被骗430万元
```

## If you need a longer abbreviation, refer to the following code 如果需要更长的缩略语，参考如下代码：

```python
from transformers import MT5ForConditionalGeneration, T5Tokenizer

model_heack = MT5ForConditionalGeneration.from_pretrained("heack/HeackMT5-ZhSum100k")
tokenizer_heack = T5Tokenizer.from_pretrained("heack/HeackMT5-ZhSum100k")


def _split_text(text, length):
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start > length:
            pos_forward = start + length
            pos_backward = start + length
            pos = start + length
            while (pos_forward < len(text)) and (pos_backward >= 0) and (pos_forward < 20 + pos) and  (pos_backward + 20 > pos) and text[pos_forward] not in {'.', '。','，',','} and text[pos_backward] not in {'.', '。','，',','}:
                pos_forward += 1
                pos_backward -= 1
            if pos_forward - pos >= 20 and pos_backward <= pos - 20:
                pos = start + length
            elif text[pos_backward] in {'.', '。','，',','}:
                pos = pos_backward
            else:
                pos = pos_forward
            chunks.append(text[start:pos+1])
            start = pos + 1
        else:
            chunks.append(text[start:])
            break
    # Combine last chunk with previous one if it's too short
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks

def get_summary_heack(text, each_summary_length=150):
    chunks = _split_text(text, 300)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer_heack.encode("summarize: " + chunk, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model_heack.generate(inputs, max_length=each_summary_length, num_beams=4, length_penalty=1.5, no_repeat_ngram_size=2)
        summary = tokenizer_heack.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)


```

## Credits
This model is trained and maintained by KongYang from Shanghai Jiao Tong University. For any questions, please reach out to me at my WeChat ID: kongyang.

## License
This model is released under the CC BY-NC-SA 4.0 license.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{kongyang2023heackmt5zhsum100k,
    title={HeackMT5-ZhSum100k: A Large-Scale Multilingual Abstractive Summarization for Chinese Texts},
    author={Kong Yang},
    year={2023}
}
