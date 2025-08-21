# from concurrent.futures import ThreadPoolExecutor
import pyterrier as pt
import os
import pyterrier_alpha as pta
import pandas as pd
from typing import List, Optional
import re
from pyterrier_rag import HuggingFaceBackend
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

try:
    from vllm import LLM, SamplingParams
    _VLLM_OK = True
except Exception:
    _VLLM_OK = False



class AgenticRAG(pt.Transformer):
    
    def __init__(
        self,
        retriever : pt.Transformer, 
        generator :None, #HuggingFaceBackend when using o1, None when using r1
        prompt:str = "{question}",
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
        self.max_tokens = max_tokens
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
        
        state_active_queries : List[Dict[str,Any]] = []
        for _, row in df.iterrows():
            state = {
                'qid': str(row['qid']),
                'query': row['query'],
                'context': self.prompt.format(question=row["query"]) if self.prompt else row["query"],
                'search_history': [],
                'search_iterations' : 0,
                'qanswer' : None,
                'output' : '',
                'stop_reason' : None,
            }
            state_active_queries.append(state)
        state_finished_queries : List[Dict[str,Any]] = []

        for turn in range(self.max_turn):
            # print(turn)
            if not state_active_queries:
                break

            # 1. 批量调用LLM生成
            #1. call the LLM for each query still active

            outputs : List[str] = self.generate([q['context'] for q in state_active_queries])  # outputs: List[str]
            #2. check for answer in each of the :
            # if we see the question has been answered:
                # extract the answer
                # remove this query from state_active, add to state_finished list
            # 2. 检查每个输出是否已经有答案
            # 这里的batch_answers是使用check_answers方法提取出的列表，带有序号
            batch_answers : List[str|None] = self.check_answers(outputs)  # List[answer or None]
            # one for every active query
            assert len(batch_answers) == len(state_active_queries)
            pending_queries : List[Dict[str,Any]] = []

            for i, answer in enumerate(batch_answers):
                this_query_state = state_active_queries[i]
                this_query_state['output'] += outputs[i]

                # 1) 严格判定是否已经有答案（避免误判首句为答案）
                ans = self.format_answers(outputs[i], strict=True)
                if ans != "no answer found":
                    this_query_state['qanswer'] = ans
                    this_query_state['stop_reason'] = 'Got answer'
                    state_finished_queries.append(this_query_state)
                    continue

                # 2) 抽取搜索词；没有就结束本条，避免空转
                next_search = self.get_search_query(outputs[i])
                if not next_search:
                    this_query_state['stop_reason'] = 'No answer, no search'
                    state_finished_queries.append(this_query_state)
                    continue

                # 记录并进入下一阶段检索
                this_query_state['search_history'].append(next_search)
                this_query_state['search_iterations'] += 1
                pending_queries.append(this_query_state)
            state_active_queries = pending_queries

            if len(pending_queries) == 0:
                break

            # 4. 执行批量检索
            #4. exectute queries
            # all_results = (self.retriever % self.top_k)(batch_queries)
            batch_queries = pd.DataFrame({
                "qid": [ f"{q['qid']}-{len(q['search_history'])}"  for q in pending_queries ],
                "query": [ q['search_history'][-1] for q in pending_queries ]
            })

            # batch_queries["qid"] = batch_queries["qid"].astype(str)
            batch_results = (self.retriever % self.top_k).transform(batch_queries)
            
            # replace state_active_queries with pending_queries
            # 5. 将检索结果加入到下轮context
            next_state_active_queries = []
            for i, q in enumerate(pending_queries):
                batch_results["qid"] = batch_results["qid"].astype(str)
                this_q_results = batch_results[batch_results.qid.str.startswith(q['qid'] + "-")]
                if len(this_q_results):
                    docs_str = self.format_docs(this_q_results)
                    q['context'] += self.wrap_search_results(docs_str)
                    next_state_active_queries.append(q)
                else:
                    q['stop_reason'] = 'No retrieval results'
                    state_finished_queries.append(q)
                    continue

            state_active_queries = next_state_active_queries

        # any still active queries must have had no answer after self.max_turns    
        if state_active_queries:
            for q in state_active_queries:
                if not q.get("stop_reason"):
                    q["stop_reason"] = "No answer after max turns"


        # 7. 合并所有已完成的和剩余未完成的
        # combine state_finished into results_df, and anything left in state_active that
        results = state_finished_queries + state_active_queries
        return pd.DataFrame(results)
    
    def check_answers(self, model_outputs: List[str]) -> List[str]:
        results = []
        for output in model_outputs:
            answer = self.format_answers(output, strict=True)
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

    def generate(self, context:List[str]) -> List[str]: 
        raise NotImplementedError
        
    #get search query from the output, can be similar among different models
    def get_search_query(self, output: str) -> Optional[str]:
        if output is None:
            return None
        start_tag = self.start_search_tag
        end_tag = self.end_search_tag

        start_idx = output.find(start_tag)
        if start_idx == -1:
            return None
        start_idx += len(start_tag)
        end_idx = output.find(end_tag, start_idx)
        if end_idx == -1:
            return None

        query = output[start_idx:end_idx].strip()
        if not query:
            return None

        query = (
            query.replace('"', "")
                 .replace("'", "")
                 .replace("\t", " ")
                 .replace("...", "")
                 .strip()
        )

        # 规范化空白
        query = re.sub(r"\s+", " ", query) if query else ""
        return query if query else None
    
    def format_docs(self, docs: pd.DataFrame) -> str:
    # 空值兜底
        if docs is None:
            return ""

        # 如果是 DataFrame
        if isinstance(docs, pd.DataFrame):
            if len(docs) == 0:
                return ""
            # 优先常见文本列
            for col in ["text", "body", "raw", "contents", "title"]:
                if col in docs.columns:
                    return "\n".join(docs[col].astype(str).tolist())
            # 没有文本列时，拼一些可读字段便于排查
            meta_cols = [c for c in ["docno", "docid", "rank", "score"] if c in docs.columns]
            if meta_cols:
                return "\n".join(
                    docs[meta_cols].astype(str).agg(" | ".join, axis=1).tolist()
                )
            # 实在没有就返回整行字符串
            return "\n".join(docs.astype(str).agg(" | ".join, axis=1).tolist())


        
    # 包装检索结果
    def wrap_search_results(self, docs_str: str):
        return f"{self.start_results_tag}{docs_str}{self.end_results_tag}"

    # 格式化检索结果
    def format_answers(self, output: str, strict: bool = False) -> str:
        # 1) 显式 <answer> 标签
        match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer

        # 2) LaTeX 风格 \\boxed{...}（可含 \\text{...}）
        match = re.search(r"\\boxed\{(.*)\}", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                inner_match = re.search(r"\\text\{(.*)\}", answer, re.DOTALL)
                if inner_match:
                    return inner_match.group(1).strip("() ")
                return answer

        # 3) 英/中 Final Answer/答案 提示样式（大小写、空格、冒号兼容）
        patterns = [
            r"(?i)final\s*answer\s*[:]\s*(.+)",
            r"(?i)answer\s*[:]\s*(.+)",
            r"(?:最终答案|答案)\s*[:]\s*(.+)",
        ]
        for p in patterns:
            m = re.search(p, output, flags=re.IGNORECASE | re.DOTALL)
            if m:
                candidate = m.group(1).strip()
                if candidate:
                    sentence = re.split(r"[\n。!?\.]", candidate, maxsplit=1)[0].strip()
                    if sentence:
                        return sentence

        # 4) 有思考标签时，取 </think> 之后首句作为兜底
        if not strict:
            if "</think>" in output:
                after_think = output.split("</think>", 1)[1].strip()
                if after_think:
                    sentence = re.split(r"[\n。!?\.]", after_think, maxsplit=1)[0].strip()
                    if sentence:
                        return sentence

            # 5) 纯文本兜底：取首个非空行/首句
            clean = (output or "").strip()
            if clean:
                sentence = re.split(r"[\n。!?\.]", clean, maxsplit=1)[0].strip()
                if sentence:
                    return sentence

        return "no answer found"

    #终止条件，子类实现
    def is_finished(self, output:str) -> bool:
        # 基于字符串的简单完成判断
        if "<answer>" in output:
            return True
        
        if re.search(r"\\boxed\{.*?\}", output):
            return True

        # Final Answer/答案 样式
        if re.search(r"(?i)final\s*answer\s*[:]", output):
            return True
        if re.search(r"(?i)\banswer\s*[:]", output):
            return True
        if re.search(r"(最终答案|答案)\s*[:]", output):
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
        top_k = 8,
        max_turn = 10,
        max_tokens = 512,
        model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        verbose = False,
        **kwargs
    ):
        super().__init__(
            retriever = retriever,
            generator = generator,
            prompt = self._get_prompt(),
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

        # 确保 tokenizer 具备 pad_token，避免批量 padding 或 generate 报错
        if self.tokenizer.pad_token is None:
            # 保证批量 padding 可用，且 HF generate() 不会报错
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(self.target_sequences, self.tokenizer)])
        self.curr_eos = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token, "<|im_end|>"]) #[151645, 151643] # for Qwen2.5 series models
        self.retrieval_top_k = top_k

        self.llm = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    def _get_prompt(self):
        prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of China?
User:{question}
Assistant: <think>"""
        return prompt

    def generate(self, contexts: List[str]) -> List[str]:
        """
        批量生成：
        - 使用 tokenizer 批量编码（padding + truncation）
        - 生成时使用 stopping_criteria（在 __init__ 已设置）
        - 只解码新增段（按每个样本的 prompt 长度裁切）
        """
        # 批量编码（注意：不能用 encode 处理 List[str]）
        encoded = self.tokenizer(
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

        # 送到设备
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens or 512,
                do_sample=True,
                temperature=self.temperature,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 仅保留“新生成”的 tokens：按每条样本的 prompt 实际长度裁切
        if attention_mask is None:
            prompt_lens = [input_ids.shape[1]] * input_ids.shape[0]
        else:
            # attention_mask 中 1 的数量即为有效 prompt 长度
            prompt_lens = attention_mask.sum(dim=1).tolist()

        texts: List[str] = []
        for i in range(outputs.size(0)):
            new_tokens = outputs[i, prompt_lens[i]:]
            texts.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        return texts
        
class R1Searcher(AgenticRAG):
    # prompt_type 说明：
    # v0: 单步推理，遇到不确定知识时用 <|begin_of_query|>关键词<|end_of_query|> 检索，适合一般问答。
    # v1: 多步/子问题推理，问题会被拆分为子问题，每个子问题用 <|begin_of_query|>kw1\tkw2<|end_of_query|> 检索，关键词用制表符分隔，适合多跳/复杂问答。
    # v2: 多步推理+关键词检索，检索query只允许关键词列表（用\t分隔），不允许完整句子，适合只用关键词检索的场景。
    # v3: 判断类推理，专为yes/no问题设计，推理后答案必须是yes或no，检索方式同v0，适合判断类问答。
    def __init__(self, 
             retriever,
             generator=None,
             temperature=0.3,
             top_k=8,
             top_p=0.95,
             max_turn=6,
             max_tokens=512,
             model_id="XXsongLALA/Qwen-2.5-7B-base-RAG-RL",   # 建议先用公开模型
             model_kw_args=None,
             prompt_type='v1',
             verbose=True,
             use_vllm=True,                         # 需要时可在实例化时传 False 直接走 transformers
             hf_token=None,
             **kwargs):
        # 父类里不再用 model_id 初始化生成后端；由本类接管
        super().__init__(retriever=retriever, generator=None, temperature=temperature,
                         top_k=top_k, top_p=top_p, max_turn=max_turn,
                         max_tokens=max_tokens, model_id=None, model_kw_args={},
                         prompt=self.get_prompt(prompt_type)
                         )

        
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        self._backend = None   # "vllm" | "hf"
        self.tokenizer = None
        self.device = None
        self.prompt_type = prompt_type
        self.start_search_tag = "<|begin_of_query|>"
        self.end_search_tag = "<|end_of_query|>"
        self.start_results_tag = "<|begin_of_documents|>"
        self.end_results_tag = "<|end_of_documents|>"

        # 基础配置与提示/标记
        # 与 R1Searcher 的查询标签一致（而非 </search>）
        self.target_sequences = ["<|end_of_query|>", " <|end_of_query|>", "<|end_of_query|>\n", " <|end_of_query|>\n", "<|end_of_query|>\n\n", " <|end_of_query|>\n\n"]

        # —— 优先尝试 vLLM（更快），失败自动回退 ——
        if use_vllm and _VLLM_OK:
            try:
                # 多进程与日志的稳态设置（notebook/本地更稳）
                os.environ.setdefault("VLLM_WORKER_MULTIPROCESSING_METHOD", "spawn")
                os.environ.setdefault("VLLM_USE_V1", "1")      # 如需 v1 再改回 "1"
                os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

                mk = dict(
                    trust_remote_code=True,
                    hf_token=self.hf_token,
                    dtype="bfloat16",                 # 或 "float16"
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.95,      # 降预分配，减少 Engine 启动失败
                    max_model_len=1024,               # 降 KV cache 预分配
                    enforce_eager=True,               # 初始化更稳
                )
                if model_kw_args:
                    mk.update(model_kw_args)

                self.llm = LLM(model=self.model_id, **mk)
                self.sampling_params = SamplingParams(
                    temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                    stop = ["<|im_end|>", self.end_search_tag, "<|endoftext|>", "</answer>"]
                    #stop = ["<|im_end|>", "<|endoftext|>"]
                    
                    # stop = self.target_sequences
                )
                self._backend = "vllm"
                if self.verbose:
                    print(f"[R1Searcher] vLLM backend ready: {self.model_id}")
            except Exception as e:
                if self.verbose:
                    print(f"[R1Searcher] vLLM init failed, fallback to transformers: {e}")

        # —— 回退：transformers（最稳的保底路径） ——
        if self._backend is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_id, token=self.hf_token, torch_dtype="auto", device_map="auto"
            )
            self.device = next(self.llm.parameters()).device
            self._backend = "hf"
            if self.verbose:
                print(f"[R1Searcher] transformers backend ready: {self.model_id} on {self.device}")
            self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(self.target_sequences, self.tokenizer)])

    def get_prompt(self, prompt_type:str):
        if prompt_type == 'v0':
            return """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

        elif prompt_type == 'v1':
            return """The User asks a question, and the Assistant solves it.
Use these tags ONLY: <think>...</think>, <|begin_of_query|>...<|end_of_query|>, <|begin_of_documents|>...<|end_of_documents|>, <answer>...</answer>.
General protocol:
1) Inside <think>, decompose the question if needed and decide what information is missing.
2) When external knowledge is needed, output EXACTLY one line:
   <|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>
   - Include the core entity/subject and the essential property/constraint keywords.
   - Add common aliases/synonyms (English and/or Chinese) when helpful.
   - Immediately STOP after <|end_of_query|>. Do NOT output anything else until <|begin_of_documents|> is provided.
3) After I return <|begin_of_documents|> ... <|end_of_documents|>, resume <think> to extract the needed facts:
   - Prefer explicit statements that directly support the requirement.
   - If evidence is insufficient or off-topic, refine keywords and SEARCH again.
4) Only when there is clear supporting evidence in <|begin_of_documents|> ... <|end_of_documents|>, output:
   <answer> final answer here </answer>

Output rules:
- Keep <think> concise; do not reveal chain-of-thought beyond the tag.
- Do NOT output <answer> until evidence from <|begin_of_documents|> is found.
- If still uncertain after several searches, continue searching; do not guess.
- Do NOT output <answer> until a clear supporting statement is found in <|begin_of_documents|>.
- If the retrieved information does not directly answer the question, refine the keywords and <|begin_of_query|>...<|end_of_query|> again.
User:{question}
Assistant: <think>
"""

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


    def generate(self, contexts: List[str]) -> List[str]:
        # vLLM 后端：原生支持批量
        if self._backend == "vllm":
            results = self.llm.generate(contexts, self.sampling_params, use_tqdm=self.verbose)
            texts: List[str] = []
            for r in results:
                first = r.outputs[0] if getattr(r, "outputs", None) else None
                generated_text = getattr(first, "text", "") if first else ""

                normalized = generated_text or ""
                last_begin_query = normalized.rfind("<|begin_of_query|>")
                last_answer = normalized.rfind("<answer>")

                if last_begin_query != -1 and (last_answer == -1 or last_begin_query > last_answer):
                    if normalized.rfind("<|end_of_query|>", last_begin_query) == -1:
                        normalized = normalized.rstrip() + " <|end_of_query|>"
                elif last_answer != -1:
                    if normalized.rfind("</answer>", last_answer) == -1:
                        normalized = normalized.rstrip() + " </answer>"

                texts.append(normalized)

            return texts

        # transformers 回退：批量 tokenize 与生成
        assert self.tokenizer is not None and self.device is not None, "HF backend not initialized"
        encoded = self.tokenizer(
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            generated = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.temperature,
                stopping_criteria=self.stopping_criteria,
                top_p=self.top_p,
                max_new_tokens=self.max_tokens or 1024,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 仅解码新生成段，按每条样本原始长度裁切
        if attention_mask is None:
            prompt_lengths = [input_ids.shape[1]] * input_ids.shape[0]
        else:
            prompt_lengths = attention_mask.sum(dim=1).tolist()

        texts: List[str] = []
        for i in range(generated.size(0)):
            new_tokens = generated[i, prompt_lengths[i]:]
            texts.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return texts

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
        