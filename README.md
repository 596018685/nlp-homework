文件简单介绍：

reptile.py用于爬虫获得原神中的圣遗物数据和书籍数据

cleandata.py用于对爬虫所得到的数据进行清洗和处理

Datahelp.py在微调大模型时获得数据集

train.py用于对大模型进行微调训练

get_results.py用于根据微调后的大模型，获得百科文档

role_get.py用于获得原神中的角色和武器的具体信息

evaluate/metric.py：用于评估事实性的指标

evaluate/ner.py:命名体识别模型，用于获得生成文本的关键词

evaluate/seqeval_metric.py:在ner模型训练时，用于评估
