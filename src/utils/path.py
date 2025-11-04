import os
from pathlib import Path
from typing import Dict, Any

from src.utils.log import logger


def find_project_dir_path(abs_file_path: str, dir_to_split="src"):
    path = abs_file_path.split(dir_to_split)[0]
    path = os.path.dirname(path)
    return path


class PathManager:
    """路径管理器，统一管理所有输入输出路径"""
    
    def __init__(self):
        # 获取项目根目录
        self.project_dir = find_project_dir_path(__file__)
        
        # 定义基础目录
        self.data_dir = Path(self.project_dir) / "data"
        self.output_dir = Path(self.project_dir) / "OUTPUT"
        self.config_dir = Path(self.project_dir) / "config"
        self.logs_dir = Path(self.project_dir) / "logs"
        
        # 确保目录存在
        self._ensure_directories()
        
        logger.info(f"项目根目录: {self.project_dir}")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [self.data_dir, self.output_dir, self.config_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    def get_data_file_path(self, filename: str) -> Path:
        """获取数据文件路径"""
        return self.data_dir / filename
    
    def get_output_file_path(self, filename: str) -> Path:
        """获取输出文件路径"""
        return self.output_dir / filename
    
    def get_config_file_path(self, filename: str) -> Path:
        """获取配置文件路径"""
        return self.config_dir / filename
    
    def get_logs_file_path(self, filename: str) -> Path:
        """获取日志文件路径"""
        return self.logs_dir / filename
    
    def list_data_files(self, pattern: str = "*") -> list:
        """列出数据目录中的文件"""
        return list(self.data_dir.glob(pattern))
    
    def list_output_files(self, pattern: str = "*") -> list:
        """列出输出目录中的文件"""
        return list(self.output_dir.glob(pattern))


# 全局路径管理器实例
path_manager = PathManager()


def get_path_config() -> Dict[str, Any]:
    """获取路径配置字典"""
    return {
        "project_dir": str(path_manager.project_dir),
        "data_dir": str(path_manager.data_dir),
        "output_dir": str(path_manager.output_dir),
        "config_dir": str(path_manager.config_dir),
        "logs_dir": str(path_manager.logs_dir),
    }
