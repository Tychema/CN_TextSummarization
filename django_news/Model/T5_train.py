import sys
import os
import torch
from transformers import AdamW
from rouge import Rouge
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from torch import nn
from transformers import Trainer, TrainingArguments
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path="Model/"
model_heack = MT5ForConditionalGeneration.from_pretrained("heack/HeackMT5-ZhSum100k").to(device)
tokenizer_heack = T5Tokenizer.from_pretrained("heack/HeackMT5-ZhSum100k")



def T5_train(content,sum,b,gradient_accumulation_steps):
    inputs = tokenizer_heack.encode("summarize: " + content, return_tensors='pt', max_length=1024, truncation=True).to(device)
    # summary_ids = model_heack.generate(inputs, max_length=1024, num_beams=10, length_penalty=0.00001,
    #                                    no_repeat_ngram_size=10).to("cuda:0")
    labels=tokenizer_heack.encode(sum, return_tensors='pt', max_length=1024, truncation=True).to(device)
    outputs = model_heack(inputs, labels=labels)
    loss = outputs.loss
    optimizer = AdamW(model_heack.parameters(), lr=1e-4,no_deprecation_warning=True)
    # data1=model_heack.lm_head(outputs.last_hidden_state[:, 0, :])
    # print("content:"+content)
    # print("title:"+sum)
    # # print("summary:"+summary)
    # print(loss)
    # print("before:")
    # for _, param in enumerate(model_heack.named_parameters()):
    #     print(param[0])
    #     print(param[1])
    #     break
    # print(model_heack.lm_head.parameters())
    loss.backward()
    if b % gradient_accumulation_steps == 0:
        optimizer.step()  # 在累积了足够的梯度之后执行梯度更新
        optimizer.zero_grad()
    return loss
    # print("after:")
    # for _, param in enumerate(model_heack.named_parameters()):
    #     print(param[0])
    #     print(param[1])
    #     break

def T5_eval():
    chunk="2021年1月，CCF决定启动新一轮中国计算机学会推荐国际学术会议和期刊目录（以下简称《目录》）调整工作并委托CCF学术工作委员会组织实施。经过前期的充分讨论和论证后，于2021年9月开始正式向各专委会征集调整建议。期间由于疫情反复，最终目录调整的评审会议于2022年10月在北京召开，会议以线上线下相结合的方式，顺利完成本次目录的修订工作。本次目录调整的总体原则是：在既有基础上进行微调，保持宁缺毋滥的原则；领域划分方式保持不变，期刊和会议的推荐类别体系保持不变，在同等情况下增加对国内期刊的支持。本次目录调整工作分为三个阶段完成：提议受理阶段，领域责任专家审议和初审推荐阶段，以及终审核准阶段。根据CCF的授权和工作安排，整个调整工作由CCF学术工作委员会主持并组织CCF相关领域的专家完成。同时，CCF学术工作委员会还负责为初审推荐阶段收集、整理和提供所需要的期刊会议相关数据以及国际上同行的观点与看法并制作了在线查询系统，以及制作了终审会议的详细材料。"
    inputs = tokenizer_heack.encode("summarize: " + chunk, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model_heack.generate(inputs, max_length=1024, num_beams=10, length_penalty=0.00001,
                                       no_repeat_ngram_size=10).to(device)
    summary = tokenizer_heack.decode(summary_ids[0], skip_special_tokens=True)
    criteria = rouge_loss
    # print(outputs.last_hidden_state.size())
    # print(model_heack.encoder.block[0].layer[0].SelfAttention)
    optimizer = AdamW(model_heack.parameters(), lr=1e-6,no_deprecation_warning=True)
    # data1=model_heack.lm_head(outputs.last_hidden_state[:, 0, :])
    loss = torch.tensor([criteria(summary, summary)], requires_grad=True)
    print(loss)

def rouge_loss(hypotheses, references, rouge_types=['rouge-1', 'rouge-2', 'rouge-l']):
    """
    计算ROUGE损失函数

    Args:
        hypotheses (list of str): 模型生成的摘要列表
        references (list of str): 参考摘要列表
        rouge_types (list of str): ROUGE指标的类型，默认包括'rouge-1', 'rouge-2', 'rouge-l'
    Returns:
        loss (float): ROUGE损失值
    """
    rouge = Rouge()

    # 使用ROUGE库计算ROUGE指标
    scores = rouge.get_scores(hypotheses, references, avg=True, ignore_empty=True)

    # 计算损失值（1 - ROUGE得分），这是一个简单的损失函数示例
    loss = 0.0
    for rouge_type in rouge_types:
        loss += 1.0 - scores[rouge_type]['f']

    # 返回平均损失值
    return loss / len(rouge_types)

if __name__ == '__main__':
    dataset=get_dataset()
    # training_args = TrainingArguments(
    #     output_dir='./results',  # output directory 结果输出地址
    #     num_train_epochs=1000,  # total # of training epochs 训练总批次
    #     per_device_train_batch_size=128,  # batch size per device during training 训练批大小
    #     per_device_eval_batch_size=128,  # batch size for evaluation 评估批大小
    #     logging_dir='./logs/rn_log',  # directory for storing logs 日志存储位置
    #     learning_rate=1e-6,  # 学习率
    #     save_steps=False,  # 不保存检查点
    #     remove_unused_columns=False
    # )
    #
    # trainer = Trainer(
    #     model=model_heack,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
    #     args=training_args,  # training arguments, defined above 训练参数
    #     train_dataset=dataset,  # training dataset 训练集
    #     eval_dataset=dataset,  # evaluation dataset 测试集
    #     compute_metrics=rouge_loss
    # )
    #
    # trainer.train()
    # # trainer.evaluate()
    # trainer.save_model(output_dir="./Model")
    # k=0
    # for i in dataset:
    #     if k==0:
    #         T5_train(i["content"],i["title"])
    #         break

    #按照自己的方法train会导致模型读不了
    epoches=1000
    batch_size=128
    e=1
    gradient_accumulation_steps = 4  # 梯度累积的步数
    total_iterations = epoches * batch_size// gradient_accumulation_steps
    # 创建一个进度条对象
    progress_bar = tqdm(total=total_iterations, desc="Processing")
    min_loss = sys.maxsize

    # 假设您已经将数据存储在名为 "dataset" 的 Pandas DataFrame 中
    # 您可以使用 Pandas 的 sample 函数随机抽样
    random_sample = dataset.sample(n=batch_size, random_state=42)
    for i in range(epoches):
        b=1
        for index, row in random_sample.iterrows():
            loss = T5_train(row["text"], row["target"],b,gradient_accumulation_steps)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss)

            progress_bar.update(1)
            if loss < min_loss:
                modelpath = os.path.join(save_path)
                model_heack.save_pretrained(modelpath)
                min_loss=loss
            b=b+1




