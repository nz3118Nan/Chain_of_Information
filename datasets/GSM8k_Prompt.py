import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import re

# progress bar
from tqdm import tqdm
from random import sample
import random
import os

GSM8K_test_dir = "datasets/grade-school-math/grade_school_math/data/test.jsonl"
GSM8K_train_dir = "datasets/grade-school-math/grade_school_math/data/train.jsonl"

GSM8K_test_df = pd.read_json(GSM8K_test_dir , lines = True)
GSM8K_train_df = pd.read_json(GSM8K_train_dir , lines = True)

GSM8K_test_df['answer_clean'] = "null"
GSM8K_train_df['answer_clean'] = "null"

for i in range(len(GSM8K_test_df)):
    rea_i = GSM8K_test_df["answer"][i]
    ans_i = rea_i.split("\n#### ")[1]
    ans_i = ans_i.replace(",","")
    GSM8K_test_df["answer_clean"][i] = ans_i

for i in range(len(GSM8K_train_df)):
    rea_i = GSM8K_train_df["answer"][i]
    ans_i = rea_i.split("\n#### ")[1]
    ans_i = ans_i.replace(",","")
    GSM8K_train_df["answer_clean"][i] = ans_i

# stardard prompt and add reason process 
def n_shot_prompt_generator(n, question, datasets, standard_index = True):                                                            
    prompt = ''
    if standard_index: 

        prompt = "[Instruction]: Please give the solution answer of the [Question] given, based on the answer of formation in [Examples].  \n\n[Examples]: \n"
        for i in range(n):
            prompt += "Question[n]: [Q[n]] \nAnswer[n]: [A[n]] \n\n".replace('[n]',str(i+1))
        
        prompt += "[Question]: \n[Q[n]] \nAnswer:  \n".replace('[n]',str(n+1))
        prompt = prompt.replace("[Q"+str(n+1)+"]",question)

        sample_df = datasets.sample(n)
        
        for i in range(n):
            prompt = prompt.replace("[Q"+str(i+1)+"]", list(sample_df["question"])[i])
            prompt = prompt.replace("[A"+str(i+1)+"]", list(sample_df["answer_clean"])[i])
    else: 
        prompt = "[Instruction]: Please give the solution answer of the [Question] given, based on the answer of formation in [Examples].  \n\n[Examples]: \n"
        for i in range(n):
            prompt += "Question[n]: [Q[n]] \n\nReasoning[n]: [R[n]]\nAnswer[n]: [A[n]] \n\n".replace('[n]',str(i+1))
        
        prompt += "[Question]: \n[Q[n]]\nReasoning: \nAnswer: ".replace('[n]',str(n+1))
        prompt = prompt.replace("[Q"+str(n+1)+"]",question)

        sample_df = datasets.sample(n)
        
        for i in range(n):
            prompt = prompt.replace("[Q"+str(i+1)+"]", list(sample_df["question"])[i])
            reason_i = list(sample_df["answer"])[i].split("#### ")[0]
            prompt = prompt.replace("[A"+str(i+1)+"]", list(sample_df["answer_clean"])[i])
            prompt = prompt.replace("[R"+str(i+1)+"]", reason_i)  
             
    return prompt

# print(n_shot_prompt_generator(3,GSM8K_test_df["question"][0], GSM8K_train_df))

OPENAI_API_KEY = "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"

reason_step_list = []

reason_step_num_list = []

for i in range(len(GSM8K_train_df)):
    reason_step_num_list.append(len(GSM8K_train_df["answer"][i].split("\n"))-1)
    reason_step_list.append(GSM8K_train_df["answer"][i].split("\n")[0:reason_step_num_list[i]])

GSM8K_train_df_reason = GSM8K_train_df.copy()
GSM8K_train_df_reason["reason_step_num"] = reason_step_num_list
GSM8K_train_df_reason["reason_step_list"] = reason_step_list


# variant prompt generator GSM8K reason
def n_shot_prompt_generator_GSM8K_reason(n, n_step, question, datasets, 
                                            standard_index = True, 
                                            reason_step = None, 
                                            answer_index = True):                                                            
    prompt = ''
    datasets_sub = datasets[datasets["reason_step_num"] == n_step]
    sample_df = datasets_sub.sample(n).reset_index(drop=True)
    if standard_index: 
        prompt = "Give the answer of the final question, based on format give in the examples.\n\n"
        for i in range(n):
            prompt += "Question: [Q] \nAnswer: The answer is ###{[A]}\n\n".replace("[Q]", 
                                                        list(sample_df["question"])[i]).replace("[A]", 
                                                        list(sample_df["answer_clean"])[i])
        
        prompt += "Question: [Q[n]] \nAnswer:  \n".replace('[n]',str(n+1))
        prompt = prompt.replace("[Q"+str(n+1)+"]",question)
        sample_df = datasets.sample(n)

        for i in range(n):
            prompt = prompt.replace("[Q"+str(i+1)+"]", list(sample_df["question"])[i])
            prompt = prompt.replace("[A"+str(i+1)+"]", list(sample_df["answer_clean"])[i])
    else: 
        prompt = "Give the answer of the final question, based on format give in the examples.\n\n"
        for i in range(n):
            prompt = prompt.replace("[Q"+str(i+1)+"]", list(sample_df["question"])[i])
            if reason_step != None:
                reason_i = ""
                for j in range(reason_step):
                    reason_i = reason_i + "Step " + str(j+1) + "  " + sample_df["reason_step_list"][i][j] + "\n"
            else:
                reason_i = list(sample_df["answer"])[i].split("#### ")[0]
            if answer_index:
                prompt += "Question: [Q] \nAnswer: \n[R]The answer is ###{[A]} \n\n".replace('[Q]', 
                                            list(sample_df["question"])[i]).replace('[R]',
                                            reason_i).replace('[A]', 
                                            list(sample_df["answer_clean"])[i])
            else:
                prompt += "Question: [Q] \nAnswer: \n[R]\n\n".replace('[Q]', 
                                            list(sample_df["question"])[i]).replace('[R]',
                                            reason_i)       


        prompt += "Question: [Q[n]]\nAnswer: ".replace('[n]',str(n+1))
        prompt = prompt.replace("[Q"+str(n+1)+"]",question)           
  
    return prompt

# print(n_shot_prompt_generator_GSM8K_reason(4,3,GSM8K_test_df["question"][0], GSM8K_train_df_reason,False,3,True))