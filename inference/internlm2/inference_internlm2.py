import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import pandas as pd
from tqdm import tqdm
import json
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
tqdm.pandas()

#! TODO: align multitask inference with sft (All T4 ZS + OS)

dataset_path  = "/localfast/xiaowang/LJPtask/Data/Datasets/"
output_folder = "/localfast/xiaowang/LJPtask/Data/icl_output/"

# test data
test_data1_path = dataset_path+"test_data1.csv"
test_data2_path = dataset_path+"test_data2.csv"
test_data3_path = dataset_path+"test_data3.csv"
test_data4_path = dataset_path+"test_data4.csv"
df_test1 = pd.read_csv(test_data1_path)
df_test2 = pd.read_csv(test_data2_path)
df_test3 = pd.read_csv(test_data3_path)
df_test4 = pd.read_csv(test_data4_path)
# one-shot data
one_shot_data_path = "/localfast/xiaowang/LJPtask/Data/one-shot"
one_shot_data1_path = one_shot_data_path+"/one_shot_dataset1.csv"
one_shot_data2_path = one_shot_data_path+"/one_shot_dataset2.csv"
one_shot_data3_path = one_shot_data_path+"/one_shot_dataset3.csv"
one_shot_data4_path = one_shot_data_path+"/one_shot_dataset4.csv"
df_one_shot1 = pd.read_csv(one_shot_data1_path)
df_one_shot2 = pd.read_csv(one_shot_data2_path)
df_one_shot3 = pd.read_csv(one_shot_data3_path)
df_one_shot4 = pd.read_csv(one_shot_data4_path)

# add column specify the model‘s accusation_type
df_test1['accusation_type'] = "single"
df_test2['accusation_type'] = "multiple"
df_test3['accusation_type'] = "single"
df_test4['accusation_type'] = "multiple"


# zero-shot Dataset
def make_prompt(task, defendant, fact, accusation_type):
    """make prompt for inference """
    if   task == 2:
        if accusation_type == "single":
            instruction = f'请你模拟法官依据下面事实和被告人预测被告的罪名（一个），只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪。\n'
        elif accusation_type == "multiple":
            instruction = f'请你模拟法官依据下面事实和被告人预测被告的所有罪名（多个），只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪、XX罪。\n'
    elif task == 3:
        instruction = f"请你模拟法官根据下列事实和被告人预测被告的判决刑期，只按照例如的格式回答，不用解释。例如：判处被告人A有期徒刑X年X个月。\n"
    elif task == 4:
        if accusation_type == "single":
            instruction = f"请你模拟法官根据下列事实和被告人预测被告的罪名（一个）以及最终判决刑期，只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪，判处有期徒刑X年X个月。\n"
        elif accusation_type == "multiple":
            instruction = f"请你模拟法官根据下列事实和被告人预测被告的所有罪名（多个）以及最终判决刑期，只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪、XX罪，判处有期徒刑X年X个月。\n"
    query = f'{instruction}被告人：{defendant}\n事实：{fact}'

    return query


class LJPDataset(Dataset):
    def __init__(self, df, task, model_checkpoint):
        self.data = df
        self.task = task
        self.model_name = model_checkpoint.split("/")[1].replace("-", "_") # for saving

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        defendant = row['defendant']
        fact = row['truncated_fact']
        accusation_type = row['accusation_type']
        prompt  = make_prompt(self.task, defendant, fact, accusation_type)
        return prompt
    
# one-shot Dataset
def format_defendant_accusation(defendant_accusation: str) -> str:
    """Convert format for  SFT dataset curation"""
    return "罪、".join(eval(defendant_accusation))

def format_imprisonment(imprisonment: float)-> str:
    """Convert imprisonment from month to year and month."""
    imprisonment = int(imprisonment)
    year = imprisonment // 12
    month = imprisonment % 12
    if year == 0:
        return f"{month}个月"
    elif month == 0:
        return f"{year}年"
    else:
        return f"{year}年{month}个月"    
    

def make_prompt_fs(task, defendant, fact, fs_defe, fs_fact, fs_impri, fs_accu, accusation_type):
    """make prompt for inference """
    if task == 2:
        if accusation_type == "single":
            instruction = f'请你模拟法官依据下面事实和被告人预测被告的罪名（一个），只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪。\n下面是一个预测被告罪名的例子:\n被告人：{fs_defe}\n事实：{fs_fact}。\n预测：被告人{fs_defe}其行为构成{fs_accu}罪。\n请你参照提示和例子预测下面的罪名：'
        elif accusation_type == "multiple":
            instruction = f'请你模拟法官依据下面事实和被告人预测被告的所有罪名（多个），只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪、XX罪。\n下面是一个例子:\n被告人：{fs_defe}\n事实：{fs_fact}。\n预测：被告人{fs_defe}其行为构成{fs_accu}罪、{fs_accu}罪。\n请你参照提示和例子预测下面的罪名：'
    elif task == 3:
        instruction = f"请你模拟法官根据下列事实和被告人预测被告的判决刑期，只按照例如的格式回答，不用解释。例如：判处被告人A有期徒刑X年X个月。\n下面是一个例子:\n被告人：{fs_defe}\n事实：{fs_fact}。\n判处被告人{fs_defe}有期徒刑{fs_impri}。\n请你参照提示和例子预测下面的刑期："
    elif task == 4:
        if accusation_type == "single":
            instruction = f"请你模拟法官根据下列事实和被告人预测被告的罪名（一个）以及最终判决刑期，只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪，判处有期徒刑X年X个月。\n被告人：{fs_defe}\n事实：{fs_fact}。\n预测：被告人{fs_defe}其行为构成{fs_accu}罪，判处有期徒刑{fs_impri}。\n请你参照提示和例子预测下面的罪名和刑期："
        elif accusation_type == "multiple":
            instruction = f"请你模拟法官根据下列事实和被告人预测被告的所有罪名（多个）以及最终判决刑期，只按照例如的格式回答，不用解释。例如：被告人A其行为构成XX罪、XX罪，判处有期徒刑X年X个月。\n被告人：{fs_defe}\n事实：{fs_fact}。\n预测：被告人{fs_defe}其行为构成{fs_accu}罪，判处有期徒刑{fs_impri}。\n请你参照提示和例子预测下面的罪名和刑期："
    query = f'{instruction}被告人：{defendant}\n事实：{fact}'
    return query


class LJPDatasetOneShot(Dataset):
    def __init__(self, df, one_shot_df, task, model_checkpoint):
        self.data = df
        self.one_shot_df = one_shot_df
        self.task = task
        self.model_checkpoint = model_checkpoint
        self.model_name = model_checkpoint.split("/")[1].replace("-", "_") # for saving

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        one_shot_row = self.one_shot_df.iloc[idx]
        one_shot_defe = one_shot_row['defendant']
        one_shot_fact = one_shot_row['truncated_fact']
        one_shot_accu = format_defendant_accusation(one_shot_row['defendant_accusation'])
        one_shot_impri = format_imprisonment(one_shot_row['imprisonment'])
        
        defendant = row['defendant']
        fact = row['truncated_fact']
        accusation_type = row['accusation_type']
        prompt  = make_prompt_fs(self.task, defendant, fact, one_shot_defe, one_shot_fact, one_shot_impri, one_shot_accu, accusation_type)
        return prompt


# load model
model_checkpoint = "internlm/internlm2-chat-7b-sft"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True,) 

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto", 
    max_memory = {7:"80GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

# ###
# # Zero-shot Inference
# ###

# ## T2: accusation prediction
# task = 2
# batch_size = 1
# test_data1_t2 = LJPDataset(df_test1, 2, model_checkpoint)
# test_data2_t2 = LJPDataset(df_test2, 2, model_checkpoint)
# test_data3_t2 = LJPDataset(df_test3, 2, model_checkpoint)
# test_data4_t2 = LJPDataset(df_test4, 2, model_checkpoint)
# dataloader_test_d1_t2 = DataLoader(test_data1_t2, batch_size=batch_size, shuffle=False) 
# dataloader_test_d2_t2 = DataLoader(test_data2_t2, batch_size=batch_size, shuffle=False) 
# dataloader_test_d3_t2 = DataLoader(test_data3_t2, batch_size=batch_size, shuffle=False) 
# dataloader_test_d4_t2 = DataLoader(test_data4_t2, batch_size=batch_size, shuffle=False)
# task2_dataloader_ls = [dataloader_test_d4_t2, dataloader_test_d3_t2, dataloader_test_d2_t2, dataloader_test_d1_t2]
# dataname_dict = {dataloader_test_d1_t2:"data1", dataloader_test_d2_t2:"data2", dataloader_test_d3_t2:"data3", dataloader_test_d4_t2:"data4"}
# task2_dataloader_ls = [dataloader_test_d4_t2]

# for num,dataloader in enumerate(task2_dataloader_ls):
#     print(f'Dataloader {num} start:\n')
#     output_dict = {}
#     responses = []
#     for step,batch in tqdm(enumerate(dataloader)):
#         with torch.no_grad():
#             try:
#                 response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=100,) #greedy search
#                 torch.cuda.empty_cache()
#             except:
#                 response = "OOM:" + str(step)
#             responses.append(response)
#             output_dict[step] = response
#     # save output_dict to json file
#     output_dir = f'{output_folder}task{str(task)}/{dataname_dict[dataloader]}/'
#     os.makedirs(output_dir, exist_ok=True)
#     with open(output_dir+f'{dataloader.dataset.model_name}_ZS.json', 'w') as f:
#         json.dump(output_dict, f, indent=4, ensure_ascii=False)
#     print(f'{dataloader.dataset.model_name}_ZS.json finished')


# ## T3: imprisonment prediction
# task = 3
# batch_size = 1
# test_data1_t3 = LJPDataset(df_test1, 3, model_checkpoint)
# test_data2_t3 = LJPDataset(df_test2, 3, model_checkpoint)
# test_data3_t3 = LJPDataset(df_test3, 3, model_checkpoint)
# test_data4_t3 = LJPDataset(df_test4, 3, model_checkpoint)
# dataloader_test_d1_t3 = DataLoader(test_data1_t3, batch_size=batch_size, shuffle=False) 
# dataloader_test_d2_t3 = DataLoader(test_data2_t3, batch_size=batch_size, shuffle=False) 
# dataloader_test_d3_t3 = DataLoader(test_data3_t3, batch_size=batch_size, shuffle=False) 
# dataloader_test_d4_t3 = DataLoader(test_data4_t3, batch_size=batch_size, shuffle=False)
# task3_dataloader_ls = [dataloader_test_d4_t3, dataloader_test_d3_t3, dataloader_test_d2_t3, dataloader_test_d1_t3]
# dataname_dict = {dataloader_test_d1_t3:"data1", dataloader_test_d2_t3:"data2", dataloader_test_d3_t3:"data3", dataloader_test_d4_t3:"data4"}
# task3_dataloader_ls = [dataloader_test_d4_t3]

# for num,dataloader in enumerate(task3_dataloader_ls):
#     print(f'Dataloader {num} start:\n')
#     output_dict = {}
#     responses = []
#     for step,batch in tqdm(enumerate(dataloader)):
#         with torch.no_grad():
#             try:
#                 response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1,max_new_tokens=100,) #greedy search
#                 torch.cuda.empty_cache()
#             except:
#                 response = "OOM:" + str(step)
#             responses.append(response)
#             output_dict[step] = response
#     # save output_dict to json file
#     output_dir = f'{output_folder}task{str(task)}/{dataname_dict[dataloader]}/'
#     os.makedirs(output_dir, exist_ok=True)
#     with open(output_dir+f'{dataloader.dataset.model_name}_ZS.json', 'w') as f:
#         json.dump(output_dict, f, indent=4, ensure_ascii=False)
#     print(f'{dataloader.dataset.model_name}_ZS.json finished')

# ### T4: accusation and imprisonment prediction
# task = 4
# batch_size = 1
# test_data1_t4 = LJPDataset(df_test1, 4, model_checkpoint)
# test_data2_t4 = LJPDataset(df_test2, 4, model_checkpoint)
# test_data3_t4 = LJPDataset(df_test3, 4, model_checkpoint)
# test_data4_t4 = LJPDataset(df_test4, 4, model_checkpoint)
# dataloader_test_d1_t4 = DataLoader(test_data1_t4, batch_size=batch_size, shuffle=False)
# dataloader_test_d2_t4 = DataLoader(test_data2_t4, batch_size=batch_size, shuffle=False)
# dataloader_test_d3_t4 = DataLoader(test_data3_t4, batch_size=batch_size, shuffle=False)
# dataloader_test_d4_t4 = DataLoader(test_data4_t4, batch_size=batch_size, shuffle=False)
# task4_dataloader_ls = [dataloader_test_d4_t4, dataloader_test_d3_t4, dataloader_test_d2_t4, dataloader_test_d1_t4]
# dataname_dict = {dataloader_test_d1_t4:"data1", dataloader_test_d2_t4:"data2", dataloader_test_d3_t4:"data3", dataloader_test_d4_t4:"data4"}
# task4_dataloader_ls = [dataloader_test_d4_t4]

# for num,dataloader in enumerate(task4_dataloader_ls):
#     print(f'Dataloader {num} start:\n')
#     output_dict = {}
#     responses = []
#     for step,batch in tqdm(enumerate(dataloader)):
#         with torch.no_grad():
#             try:
#                 response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1,max_new_tokens=100,) #greedy search
#                 torch.cuda.empty_cache()
#             except:
#                 response = "OOM:" + str(step)
#             responses.append(response)
#             output_dict[step] = response
#     # save output_dict to json file
#     output_dir = f'{output_folder}task{str(task)}/{dataname_dict[dataloader]}/'
#     os.makedirs(output_dir, exist_ok=True)
#     with open(output_dir+f'{dataloader.dataset.model_name}_ZS.json', 'w') as f:
#         json.dump(output_dict, f, indent=4, ensure_ascii=False)
#     print(f'{dataloader.dataset.model_name}_ZS.json finished')


###
# One-shot Inference
###
## T2: accusation prediction
task = 2
batch_size = 1
test_data1_t2 = LJPDatasetOneShot(df_test1, df_one_shot1, 2, model_checkpoint)
test_data2_t2 = LJPDatasetOneShot(df_test2, df_one_shot2, 2, model_checkpoint)
test_data3_t2 = LJPDatasetOneShot(df_test3, df_one_shot3, 2, model_checkpoint)
test_data4_t2 = LJPDatasetOneShot(df_test4, df_one_shot4, 2, model_checkpoint)
dataloader_test_d1_t2 = DataLoader(test_data1_t2, batch_size=batch_size, shuffle=False) 
dataloader_test_d2_t2 = DataLoader(test_data2_t2, batch_size=batch_size, shuffle=False) 
dataloader_test_d3_t2 = DataLoader(test_data3_t2, batch_size=batch_size, shuffle=False) 
dataloader_test_d4_t2 = DataLoader(test_data4_t2, batch_size=batch_size, shuffle=False)
task2_dataloader_ls = [dataloader_test_d4_t2, dataloader_test_d3_t2, dataloader_test_d2_t2, dataloader_test_d1_t2]
dataname_dict = {dataloader_test_d1_t2:"data1", dataloader_test_d2_t2:"data2", dataloader_test_d3_t2:"data3", dataloader_test_d4_t2:"data4"}
task2_dataloader_ls = [dataloader_test_d4_t2]

for num,dataloader in enumerate(task2_dataloader_ls):
    print(f'Dataloader {num} start:\n')
    output_dict = {}
    responses = []
    for step,batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            # try:
            #     response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=100,) #greedy search
            #     torch.cuda.empty_cache()
            # except:
            #     response = "OOM:" + str(step)
            response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=100,) #greedy search
            responses.append(response)
            output_dict[step] = response
    # save output_dict to json file
    output_dir = f'{output_folder}task{str(task)}/{dataname_dict[dataloader]}/'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir+f'{dataloader.dataset.model_name}_OS.json', 'w') as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)
    print(f'{dataloader.dataset.model_name}_OS.json finished')


## T3: imprisonment prediction
task = 3
batch_size = 1
test_data1_t3 = LJPDatasetOneShot(df_test1,  df_one_shot1, 3, model_checkpoint)
test_data2_t3 = LJPDatasetOneShot(df_test2, df_one_shot2,3, model_checkpoint)
test_data3_t3 = LJPDatasetOneShot(df_test3, df_one_shot3, 3, model_checkpoint)
test_data4_t3 = LJPDatasetOneShot(df_test4, df_one_shot4, 3, model_checkpoint)
dataloader_test_d1_t3 = DataLoader(test_data1_t3, batch_size=batch_size, shuffle=False) 
dataloader_test_d2_t3 = DataLoader(test_data2_t3, batch_size=batch_size, shuffle=False) 
dataloader_test_d3_t3 = DataLoader(test_data3_t3, batch_size=batch_size, shuffle=False) 
dataloader_test_d4_t3 = DataLoader(test_data4_t3, batch_size=batch_size, shuffle=False)
task3_dataloader_ls = [dataloader_test_d4_t3, dataloader_test_d3_t3, dataloader_test_d2_t3, dataloader_test_d1_t3]
dataname_dict = {dataloader_test_d1_t3:"data1", dataloader_test_d2_t3:"data2", dataloader_test_d3_t3:"data3", dataloader_test_d4_t3:"data4"}
task3_dataloader_ls = [dataloader_test_d4_t3]

for num,dataloader in enumerate(task3_dataloader_ls):
    print(f'Dataloader {num} start:\n')
    output_dict = {}
    responses = []
    for step,batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            # try:
            #     response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=100,) #greedy search
            #     torch.cuda.empty_cache()
            # except:
            #     response = "OOM:" + str(step)
            response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=100,) #greedy search
            responses.append(response)
            output_dict[step] = response
    # save output_dict to json file
    output_dir = f'{output_folder}task{str(task)}/{dataname_dict[dataloader]}/'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir+f'{dataloader.dataset.model_name}_OS.json', 'w') as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)
    print(f'{dataloader.dataset.model_name}_OS.json finished')

# ### T4: accusation and imprisonment prediction
# task = 4
# batch_size = 1
# test_data1_t4 = LJPDatasetOneShot(df_test1, df_one_shot1,4, model_checkpoint)
# test_data2_t4 = LJPDatasetOneShot(df_test2, df_one_shot2, 4,model_checkpoint)
# test_data3_t4 = LJPDatasetOneShot(df_test3, df_one_shot3,4, model_checkpoint)
# test_data4_t4 = LJPDatasetOneShot(df_test4, df_one_shot4, 4,model_checkpoint)
# dataloader_test_d1_t4 = DataLoader(test_data1_t4, batch_size=batch_size, shuffle=False)
# dataloader_test_d2_t4 = DataLoader(test_data2_t4, batch_size=batch_size, shuffle=False)
# dataloader_test_d3_t4 = DataLoader(test_data3_t4, batch_size=batch_size, shuffle=False)
# dataloader_test_d4_t4 = DataLoader(test_data4_t4, batch_size=batch_size, shuffle=False)
# task4_dataloader_ls = [dataloader_test_d4_t4, dataloader_test_d3_t4, dataloader_test_d2_t4, dataloader_test_d1_t4]
# dataname_dict = {dataloader_test_d1_t4:"data1", dataloader_test_d2_t4:"data2", dataloader_test_d3_t4:"data3", dataloader_test_d4_t4:"data4"}
# task4_dataloader_ls = [dataloader_test_d4_t4]

# for num,dataloader in enumerate(task4_dataloader_ls):
#     print(f'Dataloader {num} start:\n')
#     output_dict = {}
#     responses = []
#     for step,batch in tqdm(enumerate(dataloader)):
#         with torch.no_grad():
#             try:
#                 response, history = model.chat(tokenizer, batch, history=[], do_sample=False, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=100,) #greedy search
#                 torch.cuda.empty_cache()
#             except:
#                 response = "OOM:" + str(step)
#             responses.append(response)
#             output_dict[step] = response
#     # save output_dict to json file
#     output_dir = f'{output_folder}task{str(task)}/{dataname_dict[dataloader]}/'
#     os.makedirs(output_dir, exist_ok=True)
#     with open(output_dir+f'{dataloader.dataset.model_name}_OS.json', 'w') as f:
#         json.dump(output_dict, f, indent=4, ensure_ascii=False)
#     print(f'{dataloader.dataset.model_name}_OS.json finished')