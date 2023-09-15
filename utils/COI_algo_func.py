# function generate token distribution table df_result_majority
def entropy_df_generation(dic_result,n_round = 3):
    df = pd.DataFrame(columns=["token", 'logprob'])
    for i in range(n_round):
        result_i = dic_result[i]["choices"]
        if '{' in  result_i[0]["logprobs"]["tokens"]:
            index = result_i[0]["logprobs"]["tokens"].index('{')
        else:
            index = result_i[0]["logprobs"]["tokens"].index(' {')
        sample_token = result_i[0]["logprobs"]["tokens"][index+1]
        sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
        df.loc[len(df)] = [sample_token, sample_token_logprob]  
        log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]
        for key in list(log_list.keys()):
            df.loc[len(df)] = [key, log_list[key]]
        df["token"] = [i.replace(" ","") for i in df["token"]]

    df_result = pd.DataFrame(columns=["token", 'logprob'])
    for token in np.unique(df["token"]):
        df_result.loc[len(df_result)] = [token, np.mean(df[df["token"] == token]["logprob"])]
    df_result["prob"] = np.exp(df_result["logprob"])
    df_result["prob"] = df_result["prob"]/np.sum(df_result["prob"])
    df_result["logprob"] = np.log(df_result["prob"])
    df_result = df_result.sort_values(by = "prob", ascending = False).reset_index(drop=True)
    df_result["cumprob"] = np.cumsum(df_result["prob"])
    df_result_majority = df_result[df_result["cumprob"] < 1] 
    df_result_majority = df_result_majority[[i.replace(' ','').isnumeric() for i in df_result_majority["token"]]].reset_index(drop=True)
    df_result_majority["prob"] = np.exp(df_result_majority["logprob"])
    df_result_majority["prob"] = df_result_majority["prob"]/np.sum(df_result_majority["prob"])
    df_result_majority["cumprob"] = np.cumsum(df_result_majority["prob"])
    return df_result_majority

# openai api functions to conduct early stopping and short generation
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY
# demo cot 4-shot ICL GSM8K testset Q10 
def Davinci_openai_stop(prompt, stop_index):
    response = openai.Completion.create(
    model="text-davinci-002",prompt = prompt, temperature = 1,
    max_tokens=1024, top_p=1, frequency_penalty=0,
    presence_penalty=0,logprobs = 5,
    stop = stop_index)
    return response
def Davinci_openai(prompt):
    response = openai.Completion.create(
    model="text-davinci-002",prompt = prompt, temperature = 1,
    max_tokens=32, top_p=1, frequency_penalty=0,
    presence_penalty=0,logprobs = 5)
    return response

# function to generate next step reasoning step
def answer_diversity(ans_prompt,n_round = 3):
    count = 0 
    dic_result = {}
    while count < n_round:
        index_front  = -100
        index_end = 100 
        ans_generation = Davinci_openai(ans_prompt)
        if ("{" in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_front =  ans_generation["choices"][0]["logprobs"]["tokens"].index("{")
        if (" {" in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_front =  ans_generation["choices"][0]["logprobs"]["tokens"].index(" {")

        if ("}" in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_end =  ans_generation["choices"][0]["logprobs"]["tokens"].index("}")
        if ("} " in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_end =  ans_generation["choices"][0]["logprobs"]["tokens"].index("} ")
        
        if index_end - index_front == 2: 
            dic_result[count] = ans_generation
            count += 1 
    return dic_result

