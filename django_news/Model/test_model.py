from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
model_heack = MT5ForConditionalGeneration.from_pretrained("Model").to("cuda:0")
# model_heack =torch.load("Model/pytorch_model.bin")
tokenizer_heack = T5Tokenizer.from_pretrained("heack/HeackMT5-ZhSum100k")

def get_summary_heack(text, each_summary_length=1024):
    chunk = text
    summaries = []
    # for chunk in chunks:
    #     chunk=chunk
    inputs = tokenizer_heack.encode("summarize: " + chunk, return_tensors='pt', max_length=1024, truncation=True).to("cuda:0")
    summary_ids = model_heack.generate(inputs, max_length=each_summary_length, num_beams=10, length_penalty=0.00001, no_repeat_ngram_size=10).to("cuda:0")
    summary = tokenizer_heack.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    # summaries.append(summary)
    # return " ".join(summaries)

if __name__ == '__main__':
    chunk = "随着人们对可再生能源的关注度不断提高，太阳能电池作为一种重要的可再生能源形式，其研究和发展也备受关注。近日，科学家们发现了一种新型太阳能电池原材料，这一发现将有望推动太阳能电池产业的发展。科学家们对这种新型太阳能电池原材料进行了深入研究，发现其具有许多独特的优势。首先，这种材料的制造成本较低，有利于大规模应用和推广。其次，这种材料的光电转换效率较高，可以更好地利用太阳能，提高电池的能量产出。此外，这种材料的耐久性也较强，可以在恶劣的环境条件下稳定运行。该新型太阳能电池原材料的研究论文已发表在著名的国际期刊上。科学家们希望这一发现能够引起更多科研机构和企业对太阳能电池原材料研究的关注，进一步推动太阳能电池产业的发展。据了解，目前太阳能电池产业已经得到了广泛应用，但仍然存在一些问题，如制造成本高、光电转换效率低等。此次发现的这种新型太阳能电池原材料将有望解决这些问题，为太阳能电池产业的发展带来新的机遇。此次发现的这种新型太阳能电池原材料是由一个国际研究团队共同发现的。他们通过实验和计算，发现这种材料具有很好的光电性能和耐久性。同时，这种材料的制造成本较低，有利于大规模生产和使用。该研究团队的发现对于推动太阳能电池产业的发展具有重要的意义。该新型太阳能电池原材料的发现引起了广泛关注。有专家表示，这一发现将有望推动太阳能电池产业的发展，并对于应对气候变化、促进可持续发展具有重要意义。此外，这一发现也为其他可再生能源领域的研究提供了新的思路和方法。这一新型太阳能电池原材料的发现也受到了企业界的关注。一些光伏企业表示，这种新型太阳能电池原材料的制造成本较低，光电转换效率较高，可以帮助企业降低成本并提高产出。他们希望科学家们能够继续深入研究这种新型太阳能电池原材料的性能和应用前景，共同推动太阳能电池产业的发展。总之，科学家们发现的这种新型太阳能电池原材料具有许多独特的优势，包括制造成本低、光电转换效率高、耐久性强等。这一发现将有望推动太阳能电池产业的发展，并为应对气候变化、促进可持续发展作出重要贡献。未来，我们期待着这种新型太阳能电池原材料在更多领域得到广泛应用和推广。"


    SummarizationText=get_summary_heack(chunk)
    print(SummarizationText)