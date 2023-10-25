import json
import pandas as pd
def get_dataset():
    # 指定 JSONL 文件的路径
    json_file_path = "dataset\\thucnews_data.json"  # 替换成您的 JSON 文件路径
    i=0
    # 打开文件并逐行读取
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        data = data[641245:830714]
    print(data[0])
    # # 创建 Dataset 对象
    return data

def json_list_to_csv(data):
    df = pd.DataFrame(data)
    df.rename(columns={"title": "target", "content": "text"}, inplace=True)
    print(df.iloc[0])
    df.to_csv("./dataset/output.csv", index=False,encoding="utf_8_sig")


if __name__ == '__main__':
    data=get_dataset()
    json_list_to_csv(data)


