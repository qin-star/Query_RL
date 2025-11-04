import os
import typing as t
from pathlib import Path

from dynaconf import Dynaconf
from pydantic import BaseModel, Field

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


class PlatformModelConfig(BaseModel):
    """模型平台配置"""

    PLATFORM_TYPE: str
    """平台类型"""

    MODEL_CONFIG: dict
    """模型配置"""


class ModelSettings(BaseModel):
    """模型配置项"""

    DEFAULT_LLM_MODEL: str = Field(default="qwen3")
    """默认选用的 LLM MODEL名称"""

    DEFAULT_MODEL_CONFIG: PlatformModelConfig = PlatformModelConfig(**{
        "PLATFORM_TYPE": "ChatOpenAI",
        "MODEL_CONFIG": {
            "model": "Qwen/Qwen3-235B-A22B",
            "base_url": "https://api-inference.modelscope.cn/v1/",
            "api_key": "fc8d4030-2ac5-4cd6-9613-9c616b7ca077",
            "temperature": 0.0,
            "extra_body": {"enable_thinking": False}
        }
    })
    """默认选用的 LLM MODEL配置"""

    PLATFORM_MODEL_CONFIG: t.Dict[str, PlatformModelConfig] = {
        "deepSeek-R1": PlatformModelConfig(**{
            "PLATFORM_TYPE": "DeepSeekR1ChatOpenAI",
            "MODEL_CONFIG": {
                "model": "DeepSeek-R1-0528",
                "base_url": "http://10.72.12.30:8078/v1",
                "api_key": "None",
                "temperature": 0.0,
            }
        }),
        "deepseek-chat": PlatformModelConfig(**{
            "PLATFORM_TYPE": "ChatDeepSeek",
            "MODEL_CONFIG": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "api_key": "sk-ecd0fea313964464996b07dfcc5e32e9",
                "temperature": 0.0,
            }
        }),
        "gpt-4o-mini": PlatformModelConfig(**{
            "PLATFORM_TYPE": "ChatOpenAI",
            "MODEL_CONFIG": {
                "model": "gpt-4o-mini",
                "base_url": "https://openkey.cloud/v1",
                "api_key": "sk-jcMjY9Y6ftiktygaDc0cCaD0516b431aAc986e6578EbDbC3",
                "temperature": 0.0,
            }
        }),
        "claude-3-7": PlatformModelConfig(**{
            "PLATFORM_TYPE": "ChatOpenAI",
            "MODEL_CONFIG": {
                "model": "claude-3-7-sonnet-20250219-thinking",
                "base_url": "https://ai-yyds.com/v1",
                "api_key": "sk-kIwioJMmdPNMtUHvC4Db724843C54427B1AbE562859f5cC2",
                "temperature": 0.0,
            }
        }),
        "qwen3": PlatformModelConfig(**{
            "PLATFORM_TYPE": "ChatOpenAI",
            "MODEL_CONFIG": {
                "model": "Qwen/Qwen3-235B-A22B",
                "base_url": "https://api-inference.modelscope.cn/v1/",
                "api_key": "fc8d4030-2ac5-4cd6-9613-9c616b7ca077",
                "temperature": 0.0,
                "extra_body": {"enable_thinking": False}
            }
        }),
        "qwen3_think": PlatformModelConfig(**{
            "PLATFORM_TYPE": "Qwen3ThinkChatOpenAI",
            "MODEL_CONFIG": {
                "model": "Qwen/Qwen3-235B-A22B",
                "base_url": "https://api-inference.modelscope.cn/v1/",
                "api_key": "fc8d4030-2ac5-4cd6-9613-9c616b7ca077",
                "temperature": 0.0,
                "stream": True,
                "extra_body": {"enable_thinking": True}
            }
        }),
    }
    """模型平台配置"""

class GenerateRequirement(BaseModel):
    """生成需求配置"""
    role: str = ""
    task: str = ""
    input_args: str = ""
    restrictions: str = ""
    output_format: str = ""


class PromptConfig(BaseModel):
    """单个prompt的配置"""
    name: str = ""
    prompt_path: str = ""
    llm_generate_enable: bool = False
    generate_requirement: GenerateRequirement = Field(default_factory=GenerateRequirement)

    def is_llm_generate_enabled(self) -> bool:
        """检查是否启用LLM生成"""
        return self.llm_generate_enable


class PromptSettings(BaseModel):
    """Prompt配置项"""
    prompt_base_dir: str = ""
    prompt_version: str = ""
    debug: bool = False
    generate_llm: str = ""
    prompt_config: list[PromptConfig] = Field(default_factory=list)
    prompt_dict: dict[str, PromptConfig] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._build_prompt_dict()

    def _build_prompt_dict(self):
        """构建prompt字典，方便快速查找"""
        self.prompt_dict = {}
        for config in self.prompt_config:
            self.prompt_dict[config.name] = config

    def is_llm_generate_enabled(self, prompt_name: str) -> bool:
        """检查指定prompt是否启用LLM生成

        Args:
            prompt_name: prompt名称

        Returns:
            bool: 是否启用LLM生成
        """
        if prompt_name in self.prompt_dict:
            return self.prompt_dict[prompt_name].is_llm_generate_enabled()
        return False

    def get_prompt_path(self, prompt_name: str) -> str:
        """获取指定prompt的路径

        Args:
            prompt_name: prompt名称

        Returns:
            str: prompt路径
        """
        if prompt_name in self.prompt_dict:
            return self.prompt_dict[prompt_name].prompt_path
        return ""

    def get_generate_requirement(self, prompt_name: str) -> GenerateRequirement:
        """获取指定prompt的生成需求配置

        Args:
            prompt_name: prompt名称

        Returns:
            GenerateRequirement: 生成需求配置
        """
        if prompt_name in self.prompt_dict:
            return self.prompt_dict[prompt_name].generate_requirement
        return GenerateRequirement()

class BasicSettings(BaseModel):
    """
    基本配置信息，其它配置项修改后都需要重启服务才能生效，服务运行期间请勿修改
    """

    MODULE_NAME: str = Field(default="matrix")
    """工程项目代号"""

    LOG_PATH: str = Field(default="logs")
    """日志路径"""

    OUTPUT_PATH: str = Field(default="OUTPUT")
    """日志路径"""

    CHECKPOINT_DB: str = Field(default="plan_and_execute_checkpoints.sqlite")
    """checkpoint db文件"""

    RAG_URL: str = Field(default="http://10.65.171.100:7887/rag/chat")
    """rag url"""

    SANDBOX: dict = Field(default={"host": "127.0.0.1", "port": 5006, "user_id": "plan_and_execute_test"})
    """沙箱配置"""

    LANGFUSE_ENABLE: bool = Field(default=False)
    """langfuse开关"""

    LANGFUSE: dict = Field(default={
        "host": "http://localhost:3000",
        "secret_key": "sk-lf-cdbf9b12-3957-45b1-bf3f-38e5e022bb6d",
        "public_key": "pk-lf-fbcda1a1-9ad2-47de-a0c0-098a8d7aec7f"
    })
    """langfuse配置"""

class Settings(BaseModel):
    BASIC_SETTINGS: BasicSettings = Field(default=BasicSettings())
    MODEL_SETTINGS: ModelSettings = Field(default=ModelSettings())
    PROMPT_SETTINGS: PromptSettings = Field(default=PromptSettings())


def load_config(config_path) -> Settings:
    settings = Dynaconf(
        envvar_prefix="MATRIX",  # 环境变量前缀。设置`MATRIX_FOO='bar'`，使用`settings.FOO`
        settings_files=[os.path.join(config_path, config_file) for config_file in os.listdir(config_path)],
        environments=False,  # 启用多层次日志，支持 dev, pro
        load_dotenv=True,  # 加载 .env
        env_switcher="MATRIX_ENV",  # 用于切换模式的环境变量名称 MATRIX_ENV=production
    )

    return Settings.model_validate(settings.to_dict())



SETTINGS = load_config(CONFIG_PATH)
Path(SETTINGS.BASIC_SETTINGS.OUTPUT_PATH).resolve().mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    import yaml

    print(yaml.dump(SETTINGS.model_dump(), sort_keys=False, allow_unicode=True))