from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from role_get import role_data,weapon_data

role_list=["背景故事","性格","角色技能","喜好"]
weapon_list=["背景","资源介绍","获取方法","武器技能"]

def generate_outline_role(title, sections,sentence):
    outline = f"{title}\n"
    outline += f"{sentence}\n"
    for i in range(0, len(role_list)):
        outline += f"{role_list[i]}:{sections[i]}\n"
    return outline

def generate_outline_weapon(title, sections,sentence):
    outline = f"{title}\n"
    outline += f"{sentence}\n"
    for i in range(0, len(weapon_list)):
        outline += f"{weapon_list[i]}:{sections[i]}\n"
    return outline

model_name_or_path = "/root/model/Baichuan2-13B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()  # 以半精度加载原始模型
model = PeftModel.from_pretrained(model, "/root/model/checkpoint/output")  # 加载LoRA模型

data_r=role_data(data_load="/root/data/rool.txt")
data_w=weapon_data(data_load="/root/data/weapon.txt")
search = input("你想在原神这款游戏中了解哪个角色或者武器")
title=""
subsection=[]
if search in data_r:
    title=search+"的人物介绍"
    sentence=model.chat(tokenizer,messages=[{'role':'user','content':search+"的大致介绍"}])
    for context in role_list:
        quesetion="search"+"role_list"+"的介绍"
        subsection.append(model.chat(tokenizer,messages=[{'role':'user','content':quesetion}]))
    print(generate_outline_role(title,subsection,sentence))
elif search in data_w:
    title=search+"的武器介绍"
    sentence=model.chat(tokenizer,messages=[{'role':'user','content':search+"的大致介绍"}])
    for context in weapon_list:
        quesetion="search"+"weapon_list"+"的介绍"
        subsection.append(model.chat(tokenizer,messages=[{'role':'user','content':quesetion}]))
    print(generate_outline_weapon(title,subsection,sentence))