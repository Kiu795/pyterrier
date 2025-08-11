from concurrent.futures import ThreadPoolExecutor
import pyterrier as pt
import pyterrier_alpha as pta
import pandas as pd
from typing import List
import re
from pyterrier_rag import HuggingFaceBackend
import torch
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

import transformers



class AgenticRAG(pt.Transformer):
    
    def __init__(
        self,
        retriever : pt.Transformer, 
        generator :None, #HuggingFaceBackend when using o1, None when using r1
        prompt:str = None,
        temperature:float = 0.7,
        top_k:int = 5,
        top_p:float = 0.95,
        max_turn:int = 10,
        model_id:str = None,
        max_tokens:int = None, #max tokens for the generator when using r1searcher
        tokenizer:str = None,
        model_kw_args:dict = {},
        start_search_tag:str = None,
        end_search_tag:str = None,
        start_results_tag:str = None,
        end_results_tag:str = None,
        **kwargs
        ):
        """_summary_

        Args:
            retriever (pt.Transformer): _description_
            generator (None): _description_
            Nonewhenusingr1prompt (str, optional): _description_. Defaults to None.
            temperature (float, optional): _description_. Defaults to 0.7.
            top_k (int, optional): _description_. Defaults to 5.
            top_p (float, optional): _description_. Defaults to 0.95.
            max_turn (int, optional): _description_. Defaults to 10.
            model_id (str, optional): _description_. Defaults to None.
            max_tokens (int, optional): _description_. Defaults to None.
            model_kw_args (dict, optional): _description_. Defaults to {}.
            start_search_tag (str, optional): _description_. Defaults to None.
            end_search_tag (str, optional): _description_. Defaults to None.
            start_results_tag (str, optional): _description_. Defaults to None.
            end_results_tag (str, optional): _description_. Defaults to None.
        """

        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.prompt = prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_turn = max_turn
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model_kw_args = model_kw_args if model_kw_args else {}
        self.kwargs = kwargs

        # todo: add model_id and tokenizer

        # implement in subclasses
        self.start_search_tag = start_search_tag
        self.end_search_tag = end_search_tag
        self.start_results_tag = start_results_tag
        self.end_results_tag = end_results_tag
        self.kwargs = kwargs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # BACKUP　plan
        #return pd.concat([self.transform_one(query_df) for query_df in ])
        
        state_active_queries = []
        for _, row in df.iterrows():
            state = {
                'qid': row['qid'],
                'query': row['query'],
                'context': self.prompt.format(question=row["query"]) if self.prompt else row["query"],
                'search_history': [],
            }
            state_active_queries.append(state)
        state_finished_queries = []

        for turn in range(self.max_turn):
            if not state_active_queries:
                break
            #1. call the LLM for each query still active
            # outputs = self.generate([q['context'] for q in state_active_queries])

            # 1. 批量调用LLM生成
            outputs = self.generate([q['context'] for q in state_active_queries])  # outputs: List[str]
            #2. check for answer in each of the :
            # if we see the question has been answered:
                # extract the answer
                # remove this query from state_active, add to state_finished list
            # 2. 检查每个输出是否已经有答案
            # 这里的batch_answers是使用check_answers方法提取出的列表，带有序号
            batch_answers = self.check_answers(outputs)  # List[answer or None]
            pending_queries = []
            pending_search_str = []
            for i, answer in enumerate(batch_answers):
                if answer is not None:
                    finished_query = state_active_queries[i]
                    finished_query['qanswer'] = answer
                    finished_query['output'] = outputs[i]
                    state_finished_queries.append(finished_query)
                else:
                #如果还没有答案，则需要检索。
                    pending_queries.append(state_active_queries[i])
                    pending_search_str.append(self.get_search_query(outputs[i]))  # 提取检索query

            #3. check for retrieve requirements in each of the outputs
            # build up BATCH of queries (df) to execute
            #3a. check for outputs with no ansewr and no retrieval - that is error condition
            for i, q in enumerate(pending_queries):
                if pending_search_str[i] is None or pending_search_str[i] == "":
                    # 标记为异常，直接结束
                    q['qanswer'] = None
                    q['output'] = None
                    state_finished_queries.append(q)
            # 只保留需要检索的
            pending_queries = [q for i, q in enumerate(pending_queries) if pending_search_str[i] is not None and pending_search_str[i] != ""]
            pending_search_str = [q for q in pending_search_str if q is not None and q != ""]

            # 4. 执行批量检索
            #4. exectute queries
            # all_results = (self.retriever % self.top_k)(batch_queries)
            if pending_search_str:
                all_results = (self.retriever % self.top_k).search(pending_search_str)
            else:
                all_results = []
            # replace state_active_queries with pending_queries
            # 5. 将检索结果加入到下轮context
            for i, q in enumerate(pending_queries):
                if pending_search_str:
                    docs_str = self.format_docs(all_results[i]) if len(all_results) > i else ""
                    q['context'] += self.wrap_search_results(docs_str)
                    q['search_history'].append(pending_search_str[i])
            state_active_queries = pending_queries

            # 6. if no queries left in state_active, then break
            # 6. 如果没有剩余active，提前结束
            if not state_active_queries:
                break

        # 7. 合并所有已完成的和剩余未完成的
        # combine state_finished into results_df, and anything left in state_active that
        results = state_finished_queries + state_active_queries
        return pd.DataFrame(results)

    # def transform_backup(self, df: pd.DataFrame) -> pd.DataFrame:
    #     state_active_queries = []
    #     for _, row in df.iterrows():
    #         state = {
    #             'qid': row['qid'],
    #             'query': row['query'],
    #             'context': self.prompt.format(question=row["query"]) if self.prompt else row["query"],
    #             'search_history': [],
    #         }
    #         state_active_queries.append(state)
    #     state_finished_queries = []
    #     for turn in range(self.max_turn):
    #         outputs = self.generate([q['context'] for q in state_active_queries])
    #         batch_answers = self.check_answers(outputs)
    #         batch_querying = []
    #         batch_queries = []
    #         for i, answer in enumerate(batch_answers):
    #             if answer is not None:
    #                 finished_query = state_active_queries[i]
    #                 finished_query['qanswer'] = answer
    #                 finished_query['output'] = outputs[i]
    #                 state_finished_queries.append(finished_query)
    #             else:
    #                 batch_querying.append(state_active_queries[i])
    #                 batch_queries.append(self.get_search_query(outputs[i]))
    #         for i, q in enumerate(batch_querying):
    #             if batch_queries[i] is None or batch_queries[i] == "":
    #                 q['qanswer'] = None
    #                 q['output'] = outputs[i]
    #                 q['error'] = "No answer and no retrieval request"
    #                 state_finished_queries.append(q)
    #         batch_querying = [q for i, q in enumerate(batch_querying) if batch_queries[i] is not None and batch_queries[i] != ""]
    #         batch_queries = [q for q in batch_queries if q is not None and q != ""]
    #         if batch_queries:
    #             all_results = (self.retriever % self.top_k).search(batch_queries)
    #         else:
    #             all_results = []
    #         for i, q in enumerate(batch_querying):
    #             if batch_queries:
    #                 docs_str = self.format_docs(all_results[i]) if len(all_results) > i else ""
    #                 q['context'] += self.wrap_search_results(docs_str)
    #                 q['search_history'].append(batch_queries[i])
    #         state_active_queries = batch_querying
    #         if not state_active_queries:
    #             break
    #     results = state_finished_queries + state_active_queries
    #     return pd.DataFrame(results)
    
    def check_answers(self, model_outputs: List[str]) -> List[str]:
        results = []
        for output in model_outputs:
            answer = self.format_answers(output)
            if answer == "no answer found":
                results.append(None)
            else:
                results.append(answer)
        return results
    
    # here transform_one is used for one-row-query
    # not tested, just for concepts preparation
    def transform_one(self, sample):

        # 处理单条样本(如一行DataFrame),返回结构化结果字典。
        context = self.prompt.format(question=sample["query"]) if self.prompt else sample["query"]
        history = []
        for turn in range(self.max_turn):
            output = self.generate(context)
            history.append(output)
            query = self.get_search_query(output)
            if query:
                docs = self.retriever.search(query, qid=sample["qid"])
                docs_str = self.format_docs(docs)
                context += self.wrap_search_results(docs_str)
            # 终止条件：判断是否需要继续搜索，以及是否遇到特殊终止条件
            if self.is_finished(output):
                break
        return {
            "qid": sample["qid"],
            "query": sample["query"],
            "output": output,
            "history": history,
            "answer": self.format_answers(output)
        }

    # same as the last one, not tested, just for concepts preparation
    def parallel_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        # 并行处理所有输入样本，每个样本调用 self.transform_one(row)。
        # 返回所有结果组成的 DataFrame。
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.transform_one, row) for _, row in df.iterrows()]
            for future in as_completed(futures):
                results.append(future.result())
        return pd.DataFrame(results)

    # get prompt
    def get_prompt(self, context:str):
        raise NotImplementedError

    def generate(self, context:List[str]) -> List[str]: 
        raise NotImplementedError
        
    #get search query from the output, can be similar among different models
    def get_search_query(self, output:str) -> str:
        query = output.split(self.start_search_tag)[1].split(self.end_search_tag)[0]
        query = query.replace('"',"").replace("'","").replace("\t"," ").replace("...","")
        return query

    # 格式化检索文档
    def format_docs(self, docs: pd.DataFrame):
        return "\n".join([doc["text"] for doc in docs])
        
    # 包装检索结果
    def wrap_search_results(self, docs_str: str):
        return f"{self.start_results_tag}{docs_str}{self.end_results_tag}"

    # 格式化检索结果
    def format_answers(output: str) -> str:

        match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer

        match = re.search(r"\\boxed\{(.*)\}", output)
        if match:
            answer = match.group(1).strip()
            if answer:
                # 进一步提取 \text{...} 结构
                inner_match = re.search(r"\\text\{(.*)\}", answer)
                if inner_match:
                    return inner_match.group(1).strip("()")
                return answer

        if "</think>" in output:
            after_think = output.split("</think>", 1)[1].strip()
            if after_think:
                return after_think

        #其它自定义格式
        # if ...
        return "no answer found"

    #终止条件，子类实现
    def is_finished(self, output:str) -> bool:
        
        if "<answer>" in output:
        # 如果需要严格匹配 stop_reason
            if output.outputs[0].stop_reason:
                return output.outputs[0].stop_reason.strip() == "</answer>"
            return True
        
        if re.search(r"\\boxed\{.*?\}", output):
            return True

        return False
    

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False
        
class SearchR1(AgenticRAG):
    def __init__(self,
        retriever,
        generator = None,
        temperature = 0.7,
        top_k = 5,
        max_turn = 10,
        max_tokens = 512,
        model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        verbose = False,
        **kwargs
    ):
        super().__init__(
            retriever = retriever,
            generator = generator,
            prompt = self.get_prompt(),
            temperature = temperature,
            top_k = top_k,
            max_turn = max_turn,
            model_id = model_id,
            tokenizer = AutoTokenizer.from_pretrained(model_id),
            max_tokens = max_tokens,
            start_search_tag="<search>",
            end_search_tag="</search>",
            start_results_tag="<information>",
            end_results_tag="</information>",
            **kwargs
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, self.tokenizer)])
        self.curr_eos = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token, "<|im_end|>"]) #[151645, 151643] # for Qwen2.5 series models
        self.retrieval_top_k = top_k

        self.llm = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    def get_prompt(self):
        prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of China?
User:{question}
Assistant: <think>"""
        return prompt

    def generate(self, contexts:List[str]) -> List[str]:
        input_ids = self.tokenizer.encode(contexts, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_length = input_ids.shape[-1] 
        token_ids = self.llm.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=self.stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.temperature
        )
        token_ids = token_ids[:, prompt_length:] 
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        
        # #TODO　check what 0 is 
        # if outputs[0][-1].item() in self.curr_eos:
        #     generated_tokens = outputs[0][input_ids.shape[1]:]
        #     output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        #     return output_text
        raise ValueError("no eos token found in %s", str(outputs))
        #return outputs
        
class R1Searcher(AgenticRAG):
    # prompt_type 说明：
    # v0: 单步推理，遇到不确定知识时用 <|begin_of_query|>关键词<|end_of_query|> 检索，适合一般问答。
    # v1: 多步/子问题推理，问题会被拆分为子问题，每个子问题用 <|begin_of_query|>kw1\tkw2<|end_of_query|> 检索，关键词用制表符分隔，适合多跳/复杂问答。
    # v2: 多步推理+关键词检索，检索query只允许关键词列表（用\t分隔），不允许完整句子，适合只用关键词检索的场景。
    # v3: 判断类推理，专为yes/no问题设计，推理后答案必须是yes或no，检索方式同v0，适合判断类问答。
    def __init__(self,
        retriever, 
        generator = None,
        temperature = 0.7,
        top_k = 5,
        top_p = 0.95,
        max_turn = 10,
        max_tokens = 512,
        model_id = "XXsongLALA/Qwen-2.5-7B-base-RAG-RL",
        model_kw_args = {'tensor_parallel_size' : 1, 'gpu_memory_utilization' : 0.95},
        prompt_type = 'v1', #prompt for the agent
        verbose = False,
        **kwargs
    ):
        super().__init__(
            retriever = retriever,
            generator = generator,
            prompt = self.get_prompt(prompt_type),
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            max_turn = max_turn,
            model_id = model_id,
            tokenizer = AutoTokenizer.from_pretrained(model_id),
            max_tokens = max_tokens,
            model_kw_args=model_kw_args,
            start_search_tag="<|begin_of_query|>",
            end_search_tag="<|end_of_query|>",
            start_results_tag="<|begin_of_documents|>",
            end_results_tag="<|end_of_documents|>",
            **kwargs
        )
        self.prompt_type = prompt_type
        self.verbose = verbose
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_id, trust_remote_code=True, **model_kw_args)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def get_prompt(self, prompt_type:str):
        if prompt_type == 'v0':
            return """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

        elif prompt_type == 'v1':
            return """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
**The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

        elif prompt_type == 'v2':
            return """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

        elif prompt_type == 'v3':
            return """The User asks a **Judgment question**, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""

    def generate(self, context:str) -> str:
        output = self.llm.generate([context], self.sampling_params, use_tqdm = self.verbose)[0]
        return output.outputs[0].text

# class O1Analyser(pt.Transformer):
    
#     def __init__(backend):

    
#     @pta.transform.by_query(add_ranks=False)
#     def transform(df_one_query):
#         prompt ="""**Task Instruction:**
#             You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

#             **Guidelines:**

#             1. **Analyze the Searched Web Pages:**
#             - Carefully review the content of each searched web page.
#             - Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

#             2. **Extract Relevant Information:**
#             - Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
#             - Ensure that the extracted information is accurate and relevant.

#             3. **Output Format:**
#             - **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
#             **Final Information**

#             [Helpful information]

#             - **If the web pages do not provide any helpful information for current search query:** Output the following text.

#             **Final Information**

#             No helpful information found.

#             **Inputs:**
#             - **Previous Reasoning Steps:**  
#             {prev_reasoning}

#             - **Current Search Query:**  
#             {search_query}

#             - **Searched Web Pages:**  
#             {document}

#             Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
#             """
#         # todo fill in prompt using df_one_query
#         self.backend.generate([prompt])
#         # todo return a string
        