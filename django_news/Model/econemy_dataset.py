import json
from datasets import Dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
tokenizer_heack = T5Tokenizer.from_pretrained("heack/HeackMT5-ZhSum100k")
#经济类起点        print(data[641245])
#经济类终点        print(data[830713])

def get_dataset():
    #CSV 文件路径
    csv_file_path = "dataset/output.csv"
    # 使用 Pandas 读取 CSV 文件并存储为 DataFrame
    df = pd.read_csv(csv_file_path)
    # # 创建 Dataset 对象
    return df
    #return myDataModule(df)


if __name__ == '__main__':
    dataset=get_dataset()
    print(type(dataset))
    print(dataset[0])