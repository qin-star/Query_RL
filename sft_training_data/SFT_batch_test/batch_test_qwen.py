import pandas as pd
import json
import requests
from typing import List, Dict, Any
import time
import os
from datetime import datetime

# 配置变量
MODEL_URL = "http://10.72.1.16:36784/v1/chat/completions"
API_KEY = "sk-xxxx"  # 请替换为实际的API密钥
MODEL_NAME = "Qwen3-8B-SFT"

class QwenBatchTester:
    def __init__(self, model_url: str = MODEL_URL, api_key: str = API_KEY, model_name: str = MODEL_NAME):
        """
        初始化批量测试器
        
        Args:
            model_url: 模型API地址
            api_key: API密钥
            model_name: 模型名称
        """
        self.model_url = model_url
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 系统提示词，根据test.py中的设置
        self.system_prompt = """你是教培行业的对话理解与用户洞察专家，具备多年公考教培销售经验，擅长用户画像提取、上下文理解与问题改写。
你能够准确识别用户对话中的真实意图，尤其擅长处理模糊表达、反问句、错别字、情绪性表达等，
并将其转化为语义完整、检索目标明确、知识库能命中的清晰问题表达。

你的任务是基于用户提供的对话历史，依次完成三个任务：
1. 提取用户画像（user_profile）
2. 提炼历史上下文摘要（history_summary）
3. 对当前用户输入进行 query 改写（rewritten_query）

请严格按照用户指令中的要求和规则进行分析和输出。"""
        
        # 用户指令模板，根据test.py中的设置
        self.user_instruction_template = """请基于以下对话信息，依次完成三个任务：

## 输入信息

### 历史对话内容：
{history_chat}

### 用户当前输入：
{current_query}

## 任务要求

### 任务1：提取用户画像（user_profile）
总结用户身份背景、考试目标、当前备考阶段与关注重点，可参考以下维度：
- 年龄段或身份（如应届生、在职等）
- 目标考试类型（如公务员、事业编等）
- 当前备考阶段
- 是否有培训或复习经验
- 当前关注重点（如报名时间、课程内容、面试准备等）

> 如信息不足时，请结合上下文合理推理；如仍无法判断，可略写或留空。

### 任务2：提炼历史上下文摘要（history_summary）
请根据历史对话，提取出对当前轮对话最有帮助的核心信息，内容包括但不限于：
- 目标考试类型（如事业编、公务员等）
- 当前备考状态或用户疑问
- 用户兴趣方向（如课程、面试技巧等）
- 销售老师引导点或课程推荐记录
- 用户关注的问题趋势或反复提及内容

> 如信息不足，可不强行总结，但请尽可能压缩历史对话为有效摘要。

### 任务3：Query改写（rewritten_query）
请基于用户当前输入，结合上述用户画像和历史摘要进行改写。

#### 改写规则

**表达补全与修改：**
1. 明确用户提问中未指明的背景信息（如考试类型、公务员笔试或面试、报名流程等），请结合上下文主动补全
2. 若用户当前轮对话内容存在语病、错别字或表达不清，可基于上下文合理修改
3. 去掉语气词，例如"哦哦"、"嗯嗯"等，但不得改变原始语义
4. 保留用户原始意图，不做语义扭曲或主观判断
5. 严禁凭空添加年份、月份或具体时间信息，除非历史对话中已明确提到具体时间

**意图还原与重构：**
6. 对模糊表达结合上下文补全成清晰意图
7. 对于课程/考试类问题，重构为明确的目标性问题，如"该课程是否适用于该考试"或"课程内容是否覆盖考试核心知识点"
8. 情绪性或碎片表达，应转化为具有检索价值的问题
9. 若用户表露出对模块掌握、做题时间、考试压力等困扰，请重写为策略性建议或技巧性问题，如"如何安排科学的答题顺序""资料分析模块有哪些解题技巧"
10. 若包含多个问题，拆分为不超过三条独立问句，按重要性排序
11. 若历史对话信息中没有充足上下文信息或上下文意图不明显，则不要进行改写，不允许自己发挥
12. 若销售问客户手机号和考试目标，请不要改写
13. 若客户单纯回复"好的""嗯嗯""收到"等，请不要改写

**表达风格控制：**
14. 保持语言自然流畅，贴近教培行业用户表达习惯
15. 改写内容须与上下文保持逻辑连贯，避免信息跳跃
16. 不得加入用户未表达的内容、不得制造意图或虚构信息

## 输出格式

请严格按照以下 JSON 格式输出（三个字段缺一不可）：

```json
{{
  "user_profile": "用一句话概括用户的画像信息",
  "history_summary": "用一句话总结历史对话中对当前问题最有帮助的信息",
  "rewritten_query": "用一句话表达用户当前输入的清晰检索问题，语言自然、语义完整"
}}
```"""

    def read_training_data(self, file_path: str) -> pd.DataFrame:
        """
        读取训练数据Excel文件
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            包含训练数据的DataFrame
        """
        try:
            df = pd.read_excel(file_path)
            print(f"成功读取训练数据，共 {len(df)} 行，{len(df.columns)} 列")
            print(f"列名: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return pd.DataFrame()

    def extract_current_query_from_history(self, history_chat: str) -> str:
        """
        从历史对话中提取最后一轮用户输入作为当前查询
        这里可以根据实际的对话格式进行调整
        """
        if not history_chat or pd.isna(history_chat):
            return ""
        
        # 简单的提取逻辑：假设对话格式是可以解析的
        # 你可能需要根据实际的对话格式调整这个逻辑
        lines = str(history_chat).strip().split('\n')
        
        # 查找最后一个用户输入
        for line in reversed(lines):
            line = line.strip()
            if line and ('用户:' in line or 'user:' in line or '客户:' in line):
                # 提取用户输入内容
                if ':' in line:
                    return line.split(':', 1)[1].strip()
        
        # 如果没有找到明确的用户输入标识，返回最后一行非空内容
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line
        
        return ""

    def build_test_prompt(self, history_chat: str, current_query: str = None) -> List[Dict[str, str]]:
        """
        构建测试提示词
        
        Args:
            history_chat: 历史对话内容
            current_query: 当前用户查询（如果为空，会从历史对话中提取）
            
        Returns:
            构建好的消息列表
        """
        if not current_query:
            current_query = self.extract_current_query_from_history(history_chat)
        
        user_content = self.user_instruction_template.format(
            history_chat=history_chat or "无历史对话",
            current_query=current_query or "无当前输入"
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        return messages

    def request_model(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """
        请求模型生成响应
        
        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            
        Returns:
            模型生成的响应文本
        """
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.model_url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"响应格式异常: {result}")
                        return "响应格式异常"
                else:
                    print(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
                    
            except Exception as e:
                print(f"请求模型失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                    
        return "请求失败"

    def batch_test_model(self, input_file: str, output_file: str, start_row: int = 0, end_row: int = None,
                         history_column: str = "history_chat", current_query_column: str = None,
                         save_interval: int = 10):
        """
        批量测试模型
        
        Args:
            input_file: 输入Excel文件路径
            output_file: 输出Excel文件路径
            start_row: 开始处理的行号
            end_row: 结束处理的行号（None表示处理到最后一行）
            history_column: 历史对话列名
            current_query_column: 当前查询列名（可选）
            save_interval: 实时保存间隔（处理多少行后保存一次，默认10行）
        """
        # 读取训练数据
        df = self.read_training_data(input_file)
        if df.empty:
            return
            
        # 检查必要的列是否存在
        if history_column not in df.columns:
            print(f"未找到列 '{history_column}'，可用列: {list(df.columns)}")
            return
            
        # 确定处理范围
        if end_row is None:
            end_row = len(df)
        else:
            end_row = min(end_row, len(df))
            
        print(f"将处理第 {start_row} 行到第 {end_row-1} 行的数据")
        
        # 创建输出DataFrame的副本
        output_df = df.copy()
        
        # 添加新的列用于存储模型输出，使用-8B后缀避免覆盖原有列
        model_output_col = "model_output-8B"
        processing_time_col = "processing_time-8B"
        user_profile_col = "user_profile-8B"
        history_summary_col = "history_summary-8B"
        rewritten_query_col = "rewritten_query-8B"
        
        if model_output_col not in output_df.columns:
            output_df[model_output_col] = ""
        if processing_time_col not in output_df.columns:
            output_df[processing_time_col] = ""
        if user_profile_col not in output_df.columns:
            output_df[user_profile_col] = ""
        if history_summary_col not in output_df.columns:
            output_df[history_summary_col] = ""
        if rewritten_query_col not in output_df.columns:
            output_df[rewritten_query_col] = ""
        
        # 为输出文件添加时间后缀
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(output_file)
        timestamped_output_file = f"{name}_{timestamp}{ext}"
        
        # 程序开始时立即创建输出文件并写入完整的表头结构
        print(f"创建输出文件: {timestamped_output_file}")
        try:
            # 创建包含完整结构的DataFrame（原始数据 + 新列）
            output_df.to_excel(timestamped_output_file, index=False)
            print(f"输出文件创建成功，已写入完整表头结构，共 {len(output_df.columns)} 列")
            print(f"列名: {list(output_df.columns)}")
        except Exception as e:
            print(f"创建输出文件失败: {e}")
            return
        
        # 批量处理
        success_count = 0
        total_count = end_row - start_row
        last_save_row = start_row
        
        print(f"开始批量处理，共 {total_count} 行数据...")
        
        for i in range(start_row, end_row):
            try:
                # 获取历史对话内容
                history_chat = df.iloc[i][history_column]
                
                # 获取当前查询（如果有指定列）
                current_query = None
                if current_query_column and current_query_column in df.columns:
                    current_query = df.iloc[i][current_query_column]
                
                if pd.isna(history_chat) or history_chat == "":
                    print(f"第 {i+1} 行数据为空，跳过")
                    output_df.at[i, model_output_col] = "输入数据为空"
                    output_df.at[i, processing_time_col] = "0"
                    continue
                
                print(f"处理第 {i+1}/{total_count} 行数据...")
                
                # 构建提示词
                messages = self.build_test_prompt(str(history_chat), current_query)
                
                # 记录开始时间
                start_time = time.time()
                
                # 请求模型
                model_output = self.request_model(messages)
                
                # 记录结束时间
                end_time = time.time()
                processing_time = f"{end_time - start_time:.2f}s"
                
                # 尝试解析JSON输出
                try:
                    # 检查输出是否为空
                    if not model_output or model_output.strip() == "":
                        raise ValueError("模型输出为空")
                    
                    # 清理可能的格式问题，移除可能的代码块标记
                    cleaned_output = model_output.strip()
                    if cleaned_output.startswith("```json"):
                        cleaned_output = cleaned_output[7:]
                    if cleaned_output.endswith("```"):
                        cleaned_output = cleaned_output[:-3]
                    cleaned_output = cleaned_output.strip()
                    
                    # 尝试查找JSON部分（处理可能包含额外文本的情况）
                    json_start = cleaned_output.find('{')
                    json_end = cleaned_output.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_part = cleaned_output[json_start:json_end]
                        output_json = json.loads(json_part)
                        user_profile = output_json.get("user_profile", "")
                        history_summary = output_json.get("history_summary", "")
                        rewritten_query = output_json.get("rewritten_query", "")
                    else:
                        raise ValueError("未找到有效的JSON格式")
                    
                    print(f"JSON解析成功:")
                    print(f"  用户画像: {user_profile}")
                    print(f"  历史摘要: {history_summary}")
                    print(f"  重写查询: {rewritten_query}")
                    
                except Exception as e:
                    # 如果解析失败，记录错误并将整个输出作为model_output
                    print(f"JSON解析失败: {e}")
                    print(f"原始输出: {model_output}")
                    user_profile = "解析失败"
                    history_summary = "解析失败"
                    rewritten_query = model_output
                
                # 保存结果到DataFrame
                output_df.at[i, model_output_col] = model_output
                output_df.at[i, processing_time_col] = processing_time
                output_df.at[i, user_profile_col] = user_profile
                output_df.at[i, history_summary_col] = history_summary
                output_df.at[i, rewritten_query_col] = rewritten_query
                
                # 每次处理完一行立即保存到Excel
                try:
                    output_df.to_excel(timestamped_output_file, index=False)
                    print(f"✅ 第 {i+1} 行数据已保存到: {timestamped_output_file}")
                except Exception as e:
                    print(f"❌ 保存第 {i+1} 行失败: {e}")
                
                print(f"第 {i+1} 行处理完成，耗时: {processing_time}")
                print(f"输入: {str(history_chat)}...")
                print(f"输出: {model_output}...")
                print("-" * 50)
                
                success_count += 1
                
                # 添加延迟避免过快请求
                time.sleep(1)
                
            except Exception as e:
                print(f"处理第 {i+1} 行时出错: {e}")
                output_df.at[i, model_output_col] = f"处理错误: {e}"
                output_df.at[i, processing_time_col] = "错误"
                output_df.at[i, user_profile_col] = ""
                output_df.at[i, history_summary_col] = ""
                output_df.at[i, rewritten_query_col] = ""
                
                # 即使出错也要保存当前状态
                try:
                    output_df.to_excel(timestamped_output_file, index=False)
                    print(f"✅ 第 {i+1} 行错误状态已保存到: {timestamped_output_file}")
                except Exception as save_e:
                    print(f"❌ 保存第 {i+1} 行错误状态失败: {save_e}")
        
        # 最终确认保存
        try:
            output_df.to_excel(timestamped_output_file, index=False)
            print(f"🎉 所有数据已最终保存到: {timestamped_output_file}")
            print(f"✅ 成功处理 {success_count}/{total_count} 条数据")
        except Exception as e:
            print(f"❌ 最终保存失败: {e}")

    def test_single_sample(self, context: str) -> str:
        """
        测试单个样本
        
        Args:
            context: 上下文信息
            
        Returns:
            模型生成的响应
        """
        messages = self.build_test_prompt(context)
        return self.request_model(messages)


def main():
    """
    主函数
    """
    # 创建测试器实例
    tester = QwenBatchTester()
    
    # 输入输出文件路径
    input_file = "/home/jovyan2/query_rl/sft_training_data/data/sft/chengla_v2/橙啦-query_RL_训练集.xlsx"
    output_file = "/home/jovyan2/query_rl/sft_training_data/data/sft/chengla_v2/Test_data/qwen_batch_test_results.xlsx"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    print("开始批量测试Qwen-8B模型...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 执行批量测试 - 处理整个数据集
    # 如果需要测试部分数据，可以指定start_row和end_row参数
    tester.batch_test_model(input_file, output_file, start_row=0, end_row=None)  # 处理全部数据
    
    print("批量测试完成！")


if __name__ == "__main__":
    main()