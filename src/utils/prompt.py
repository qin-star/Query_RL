import os
from pathlib import Path
from typing import Dict, Optional

from jinja2 import DebugUndefined, Template

from src.utils.log import logger
from src.utils.settings import SETTINGS
from src.utils.llm import get_chat_llm

class PromptGenerator:
    """Prompt生成器，用于生成prompt"""

    def __init__(self, prompt_name: str, prompt_dir: str):
        self.prompt_name = prompt_name
        self.prompt_dir = prompt_dir
        self.prompt_path = SETTINGS.PROMPT_SETTINGS.get_prompt_path(prompt_name)
        self.generate_requirement = SETTINGS.PROMPT_SETTINGS.get_generate_requirement(prompt_name)
        self.generate_llm = get_chat_llm(model_name=SETTINGS.PROMPT_SETTINGS.generate_llm)
        self.prompt_version = SETTINGS.PROMPT_SETTINGS.prompt_version
        self.generate_template = """### 角色描述
    你是一名高级评分专家，擅长多模型对比评分。请根据以下要求设计双模型对比评分prompt：
    1. 对比actor_response和reference_response的四个维度：quality_improvement(质量提升), relevance_accuracy(相关性准确性), info_completeness(信息完整性), retrieval_effectiveness(检索有效性)
    2. 输出结构应为JSON格式，包含每个维度的0-1分评分和总体评分，以及详细分析说明

    ### prompt设计原则
    ```
    1、清晰明确：使用分隔符（如 '''、---）区分指令与输入内容，避免歧义
    2、结构化输出：要求模型返回 JSON、XML 或表格等格式，便于程序解析
    3、分步引导复杂任务：将多环节任务拆解为有序步骤，添加"逐步思考"指令
    4、角色扮演（Role-Playing）：赋予模型特定身份（如"资深营养师"），约束回答视角
    ```

    ### 业务需求
    ```
    {{ role }}
    {{ task }}
    {{ input_args }}
    {{ restrictions }}
    {{ output_format }}
    ```
        
    ### 参考例子
    ```
    ### 角色描述
    你是高级评分专家，擅长多模型对比评分

    ### 任务描述
    请对比分析两个模型的回复质量，并给出详细评分

    ### 输入参数
    actor_response: {{ actor_response }}
    reference_response: {{ reference_response }}
    comparison_metrics: {{ comparison_metrics }}

    ### 输出要求
    请输出JSON格式的评分结果，包含以下字段：
    {
        "quality_improvement": 0.85,
        "relevance_accuracy": 0.92,
        "info_completeness": 0.78,
        "retrieval_effectiveness": 0.88,
        "overall_score": 0.86,
        "analysis": "详细分析说明..."
    }
    ```

    """

    def generate_prompt(self, **kwargs):
        """生成prompt"""
        logger.info(f"生成prompt {self.prompt_name} 的参数: {kwargs}")
        generate_content = self.generate_llm.invoke(
            Template(self.generate_template, undefined=DebugUndefined).render(**kwargs)).content
        logger.info(f"生成prompt {self.prompt_name} 的结果: {generate_content}")
        if not os.path.exists(self.prompt_path):
            self.prompt_path = os.path.join(self.prompt_dir, self.prompt_version.replace("-", "/"),
                                            self.prompt_name + ".txt")
            # 备份已有内容
            if os.path.exists(self.prompt_path):
                with open(self.prompt_path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
                if old_content.strip():
                    base_path = self.prompt_path[:-4]
                    counter = 1
                    while os.path.exists(f"{base_path}-backup{counter}.txt"):
                        counter += 1
                    backup_path = f"{base_path}-backup{counter}.txt"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(old_content)
                    logger.info(f"已备份原文件内容到: {backup_path}")
            os.makedirs(os.path.dirname(self.prompt_path), exist_ok=True)
        with open(self.prompt_path, "w", encoding="utf-8") as f:
            f.write(generate_content)

class PromptManager:
    """Prompt管理器，用于统一管理和加载prompt文件"""

    def __init__(self, prompt_dir: str):
        """初始化PromptManager

        Args:
            prompt_dir: prompt文件所在目录的路径
        """
        self.prompt_dir = prompt_dir
        self.prompts: Dict[str, Template] = {}
        self._load_prompts()

    def _load_prompts(self):
        """从prompt目录加载所有prompt文件"""
        # 确保目录存在
        if not os.path.exists(self.prompt_dir):
            raise FileNotFoundError(f"Prompt目录 {self.prompt_dir} 不存在")

        # 如果开启debug模式，则会判断是否生成prompt
        if SETTINGS.PROMPT_SETTINGS.debug:
            for prompt_config in SETTINGS.PROMPT_SETTINGS.prompt_config:
                prompt_name = prompt_config.name
                if prompt_config.llm_generate_enable:
                    try:
                        prompt_generator = PromptGenerator(prompt_name, self.prompt_dir)
                        prompt_generator.generate_prompt(**prompt_config.generate_requirement.model_dump())
                    except Exception as e:
                        logger.error(f"生成prompt {prompt_name} 时出错: {str(e)}")

        # 加载所有.txt文件
        for file_path in Path(self.prompt_dir).glob("**/*.txt"):
            prompt_name = file_path.stem
            # 若开启debug模式，则可以指定prompt读取路径
            if SETTINGS.PROMPT_SETTINGS.debug:
                if SETTINGS.PROMPT_SETTINGS.get_prompt_path(prompt_name):
                    file_path = SETTINGS.PROMPT_SETTINGS.get_prompt_path(prompt_name)
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 创建Template对象
                template = Template(content, undefined=DebugUndefined)
                # 直接使用文件名作为key，简化命名
                self.prompts[prompt_name] = template
            except Exception as e:
                logger.error(f"加载prompt文件 {file_path} 时出错: {str(e)}")

    def get_prompt(self, prompt_name: str) -> Optional[Template]:
        """获取指定名称的prompt模板

        Args:
            prompt_name: prompt的名称（不含.txt扩展名）

        Returns:
            如果找到对应的prompt模板则返回Template对象，否则返回None
        """
        return self.prompts.get(prompt_name)

    def render_prompt(self, prompt_name: str, **kwargs) -> Optional[str]:
        """渲染指定的prompt模板

        Args:
            prompt_name: prompt的名称（不含.txt扩展名）
            **kwargs: 传递给模板的参数

        Returns:
            如果成功渲染则返回渲染后的字符串，否则返回None
        """
        template = self.get_prompt(prompt_name)
        if template:
            try:
                return template.render(**kwargs)
            except Exception as e:
                logger.error(f"渲染prompt {prompt_name} 时出错: {str(e)}")
                return None
        else:
            raise ValueError(f"prompt: {prompt_name} not found")

    def list_prompts(self) -> list[str]:
        """列出所有可用的prompt名称

        Returns:
            prompt名称列表
        """
        return list(self.prompts.keys())

    def reload_prompts(self):
        """重新加载所有prompt文件"""
        self.prompts.clear()
        self._load_prompts()


# 暂时实例化在这里
prompt_dir = SETTINGS.PROMPT_SETTINGS.prompt_base_dir

g_sa_prompt_manager = PromptManager(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + prompt_dir)

if __name__ == '__main__':
    print(g_sa_prompt_manager.render_prompt("ybx-v0.1-reply_generation", tools="test"))
    print("done.")
