"""
将橙啦训练数据集转换为SFT训练格式（多任务输出版本）

任务：从对话历史中同时生成 user_profile + history_summary + rewritten_query

输入: 橙啦-query_RL_训练集.xlsx
输出: JSONL格式的训练数据（train/val/test split）
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import List, Dict

class SFTDataConverterV2:
    """SFT训练数据转换器 - 多任务输出版本"""
    
    def __init__(self, tenant_id: str = "chengla"):
        self.tenant_id = tenant_id
        
        # 系统prompt - 只包含角色定位和通用指令，不包含数据占位符
        self.system_prompt = """你是教培行业的对话理解与用户洞察专家，具备多年公考教培销售经验，擅长用户画像提取、上下文理解与问题改写。
你能够准确识别用户对话中的真实意图，尤其擅长处理模糊表达、反问句、错别字、情绪性表达等，
并将其转化为语义完整、检索目标明确、知识库能命中的清晰问题表达。

你的任务是基于用户提供的对话历史，依次完成三个任务：
1. 提取用户画像（user_profile）
2. 提炼历史上下文摘要（history_summary）
3. 对当前用户输入进行 query 改写（rewritten_query）

请严格按照用户指令中的要求和规则进行分析和输出。"""
        
        # 用户指令模板 - 包含完整的任务说明、规则和输出格式
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
    
    def convert_to_messages_format(
        self, 
        context: str, 
        user_profile: str,
        history_summary: str,
        rewritten_query: str
    ) -> Dict:
        """转换为messages格式
        
        输入：对话历史
        输出：JSON格式的三个字段
        """
        
        # 拆分对话历史：提取最后一条客户消息作为当前query
        lines = context.strip().split('\n')
        
        # 从后往前找最后一条[客户]消息
        current_query = ""
        history_chat = ""
        
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('[客户]'):
                # 找到最后一条客户消息
                if '：' in line:
                    current_query = line.split('：', 1)[1].strip()
                else:
                    current_query = line
                
                # 其余部分作为历史对话
                history_chat = '\n'.join(lines[:i]) if i > 0 else ""
                break
        
        # 如果没有找到[客户]消息，将最后一行作为query，前面的作为历史
        if not current_query and lines:
            current_query = lines[-1]
            history_chat = '\n'.join(lines[:-1]) if len(lines) > 1 else ""
        
        # 使用模板填充实际数据，构建完整的用户指令
        user_content = self.user_instruction_template.format(
            history_chat=history_chat if history_chat else "（无历史对话）",
            current_query=current_query
        )
        
        # 助手输出：JSON格式
        assistant_output = {
            "user_profile": user_profile.strip() if pd.notna(user_profile) else "",
            "history_summary": history_summary.strip() if pd.notna(history_summary) else "",
            "rewritten_query": rewritten_query.strip() if pd.notna(rewritten_query) else ""
        }
        
        # 转换为JSON字符串（格式化输出）
        json_output = json.dumps(assistant_output, ensure_ascii=False, indent=2)
        
        # 在输出前添加<think>标记，保持模型推理能力
        assistant_content = f"<think>\n\n</think>\n\n{json_output}"
        
        # 构建messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
        
        return messages
    
    def extract_last_customer_query(self, context: str) -> str:
        """从对话上下文中提取最后一条客户消息（用于metadata）"""
        lines = context.strip().split('\n')
        
        # 从后往前找最后一条[客户]消息
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('[客户]'):
                # 提取时间戳后的内容
                if '：' in line:
                    query = line.split('：', 1)[1].strip()
                    return query
        
        # 如果没找到，返回最后一行（去除前缀）
        last_line = lines[-1].strip() if lines else ""
        if '：' in last_line:
            return last_line.split('：', 1)[1].strip()
        return last_line
    
    def convert_excel_to_jsonl(
        self,
        excel_path: str,
        output_dir: str = "data/sft/chengla_v2",
        quality_filter: bool = True
    ):
        """将Excel转换为JSONL训练格式
        
        Args:
            excel_path: Excel文件路径
            output_dir: 输出目录
            quality_filter: 是否进行质量过滤
        """
        
        print("=" * 80)
        print("开始转换SFT训练数据（多任务输出版本）")
        print("=" * 80)
        print("\n📋 任务说明：")
        print("  输入：对话历史")
        print("  输出：{user_profile, history_summary, rewritten_query}")
        
        # 读取Excel
        df = pd.read_excel(excel_path)
        print(f"\n📚 读取数据集: {len(df)} 条")
        
        # 数据清洗和质量过滤
        original_count = len(df)
        
        # 1. 移除空值 - 三个目标字段都不能为空
        required_cols = ['rewritten_query', 'user_profile', 'history_summary']
        for col in required_cols:
            df = df.dropna(subset=[col])
            df = df[df[col].str.strip() != '']
        
        print(f"✅ 移除空值样本: {original_count - len(df)} 条，剩余 {len(df)} 条")
        
        if quality_filter:
            # 2. 长度检查
            initial_len = len(df)
            
            # rewritten_query: 5-200字符
            df = df[df['rewritten_query'].str.len() >= 5]
            df = df[df['rewritten_query'].str.len() <= 200]
            
            # user_profile: 10-300字符
            df = df[df['user_profile'].str.len() >= 10]
            df = df[df['user_profile'].str.len() <= 300]
            
            # history_summary: 10-300字符
            df = df[df['history_summary'].str.len() >= 10]
            df = df[df['history_summary'].str.len() <= 300]
            
            print(f"✅ 长度过滤后剩余: {len(df)} 条")
            
            # 3. 确保对话历史不为空
            df = df[df['最终传参上下文'].str.len() >= 20]
            print(f"✅ 对话历史过滤后剩余: {len(df)} 条")
        
        # 转换为训练格式
        training_samples = []
        
        for idx, row in df.iterrows():
            try:
                # 提取原始query（用于metadata）
                original_query = self.extract_last_customer_query(row['最终传参上下文'])
                
                # 转换为messages格式
                messages = self.convert_to_messages_format(
                    context=row['最终传参上下文'],
                    user_profile=row['user_profile'],
                    history_summary=row['history_summary'],
                    rewritten_query=row['rewritten_query']
                )
                
                sample = {
                    "messages": messages,
                    "metadata": {
                        "source": "chengla_rl_dataset",
                        "tenant_id": self.tenant_id,
                        "sample_id": f"chengla_v2_{idx}",
                        "original_query": original_query,
                        "task_type": "multi_output"  # 标记为多任务输出
                    }
                }
                
                training_samples.append(sample)
                
            except Exception as e:
                print(f"⚠️  处理第{idx}行时出错: {e}")
                continue
        
        print(f"\n✅ 成功转换: {len(training_samples)} 条样本")
        
        # 划分训练集、验证集、测试集
        self.split_and_save(training_samples, output_dir)
        
        # 保存样本示例
        self.save_sample_examples(training_samples[:3], output_dir)
        
        return training_samples
    
    def split_and_save(
        self,
        samples: List[Dict],
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ):
        """划分数据集并保存"""
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 打乱数据
        random.shuffle(samples)
        
        # 划分
        n = len(samples)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        print("\n" + "=" * 80)
        print("数据集划分")
        print("=" * 80)
        print(f"训练集: {len(train_samples)} 条 ({len(train_samples)/n*100:.1f}%)")
        print(f"验证集: {len(val_samples)} 条 ({len(val_samples)/n*100:.1f}%)")
        print(f"测试集: {len(test_samples)} 条 ({len(test_samples)/n*100:.1f}%)")
        
        # 保存为JSONL格式
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, split_data in splits.items():
            output_file = output_path / f"{split_name}_latest.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"💾 已保存: {output_file}")
        
        # 生成统计报告
        self.generate_stats_report(splits, output_path)
    
    def save_sample_examples(self, samples: List[Dict], output_path: Path):
        """保存样本示例供查看"""
        
        example_file = Path(output_path) / "sample_examples.json"
        
        with open(example_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 样本示例已保存: {example_file}")
        
        # 在控制台显示第一个样本
        if samples:
            print("\n" + "=" * 80)
            print("样本示例（第1条）")
            print("=" * 80)
            
            sample = samples[0]
            
            print("\n【系统Prompt】")
            print("-" * 80)
            print(sample['messages'][0]['content'][:300] + "...")
            
            print("\n【用户输入】")
            print("-" * 80)
            user_msg = sample['messages'][1]['content']
            # 只显示前500字符
            if len(user_msg) > 500:
                print(user_msg[:500] + "...")
            else:
                print(user_msg)
            
            print("\n【模型输出】")
            print("-" * 80)
            print(sample['messages'][2]['content'])
    
    def generate_stats_report(self, splits: Dict, output_path: Path):
        """生成统计报告"""
        
        report = {
            "tenant_id": self.tenant_id,
            "dataset_name": "chengla_query_rewrite_sft_v2",
            "task_type": "multi_output",
            "output_fields": ["user_profile", "history_summary", "rewritten_query"],
            "total_samples": sum(len(split) for split in splits.values()),
            "splits": {}
        }
        
        for split_name, split_data in splits.items():
            # 解析JSON输出统计长度
            user_profile_lengths = []
            history_summary_lengths = []
            rewritten_query_lengths = []
            
            for sample in split_data:
                try:
                    # 解析assistant的JSON输出（需要去掉<think>标记）
                    assistant_content = sample['messages'][2]['content']
                    
                    # 提取JSON部分（去掉<think>\n\n</think>\n\n前缀）
                    if assistant_content.startswith('<think>\n\n</think>\n\n'):
                        json_str = assistant_content.replace('<think>\n\n</think>\n\n', '', 1)
                    else:
                        json_str = assistant_content
                    
                    output_json = json.loads(json_str)
                    
                    user_profile_lengths.append(len(output_json.get('user_profile', '')))
                    history_summary_lengths.append(len(output_json.get('history_summary', '')))
                    rewritten_query_lengths.append(len(output_json.get('rewritten_query', '')))
                except:
                    continue
            
            report["splits"][split_name] = {
                "total_samples": len(split_data),
                "avg_user_profile_length": sum(user_profile_lengths) / len(user_profile_lengths) if user_profile_lengths else 0,
                "avg_history_summary_length": sum(history_summary_lengths) / len(history_summary_lengths) if history_summary_lengths else 0,
                "avg_rewritten_query_length": sum(rewritten_query_lengths) / len(rewritten_query_lengths) if rewritten_query_lengths else 0
            }
        
        # 保存报告
        report_path = output_path / "stats_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 统计报告已保存: {report_path}")
        print("\n" + "=" * 80)
        print("统计摘要")
        print("=" * 80)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    """主函数"""
    
    converter = SFTDataConverterV2(tenant_id="chengla")
    
    # 转换数据
    samples = converter.convert_excel_to_jsonl(
        excel_path=r"code\data\橙啦-query_RL_训练集.xlsx",
        output_dir="D:\工作文件\RAG开发\Reference-code\DeepRetrieval-main\sft_training_data\data\sft\chengla_v2",
        quality_filter=True
    )
    
    print("\n" + "=" * 80)
    print("✨ SFT数据准备完成！（多任务输出版本）")
    print("=" * 80)
    print("\n📂 输出文件：")
    print("  - data/sft/chengla_v2/train_latest.jsonl")
    print("  - data/sft/chengla_v2/val_latest.jsonl")
    print("  - data/sft/chengla_v2/test_latest.jsonl")
    print("  - data/sft/chengla_v2/stats_report.json")
    print("  - data/sft/chengla_v2/sample_examples.json")
    
    print("\n🎯 训练任务：")
    print("   输入：对话历史")
    print("   输出：JSON格式 {user_profile, history_summary, rewritten_query}")
    
    print("\n🚀 下一步：使用这些数据开始SFT训练")
    print("   训练时需要注意JSON格式输出的解析")


if __name__ == "__main__":
    # 设置随机种子保证可复现
    random.seed(42)
    
    main()

