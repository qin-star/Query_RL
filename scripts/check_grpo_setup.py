#!/usr/bin/env python3
"""
GRPO训练环境检查脚本
用于验证所有配置和依赖是否正确设置
"""

import os
import sys
import json
import yaml
from pathlib import Path
import subprocess
import importlib

class GRPOSetupChecker:
    """GRPO设置检查器"""
    
    def __init__(self):
        self.base_path = Path("/home/jovyan2/query_rl")
        self.issues = []
        self.warnings = []
        self.success_count = 0
        
    def check_python_version(self):
        """检查Python版本"""
        print("🔍 检查Python版本...")
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 10:
            print(f"✅ Python版本: {sys.version}")
            self.success_count += 1
        else:
            self.issues.append(f"❌ Python版本过低: {sys.version}，需要3.10+")
            
    def check_dependencies(self):
        """检查依赖包"""
        print("\n🔍 检查依赖包...")
        required_packages = [
            "torch", "transformers", "verl", "vllm", 
            "openai", "wandb", "pandas", "numpy", 
            "aiohttp", "pyyaml", "jinja2"
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                print(f"✅ {package}")
                self.success_count += 1
            except ImportError:
                self.issues.append(f"❌ 缺少依赖包: {package}")
                
    def check_model_files(self):
        """检查模型文件"""
        print("\n🔍 检查模型文件...")
        model_path = self.base_path / "model" / "Qwen3-8B"
        
        if model_path.exists():
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            for file in required_files:
                file_path = model_path / file
                if file_path.exists():
                    print(f"✅ {file}")
                    self.success_count += 1
                else:
                    self.warnings.append(f"⚠️  模型文件不存在: {file_path}")
        else:
            self.issues.append(f"❌ 模型目录不存在: {model_path}")
            
    def check_config_files(self):
        """检查配置文件"""
        print("\n🔍 检查配置文件...")
        config_files = [
            "src/config/model_settings.yaml",
            "src/config/basic_settings.yaml",
            "verl_code/config/sales_rag_grpo_dual_model_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.base_path / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    print(f"✅ {config_file}")
                    self.success_count += 1
                except Exception as e:
                    self.issues.append(f"❌ 配置文件格式错误 {config_file}: {e}")
            else:
                self.issues.append(f"❌ 配置文件不存在: {config_file}")
                
    def check_data_files(self):
        """检查数据文件"""
        print("\n🔍 检查数据文件...")
        excel_path = self.base_path / "data" / "sales_rag" / "RL_tranning_data" / "橙啦-query_RL_训练集.xlsx"
        
        if excel_path.exists():
            print(f"✅ 训练数据文件: {excel_path}")
            self.success_count += 1
        else:
            self.issues.append(f"❌ 训练数据文件不存在: {excel_path}")
            
    def check_rag_service(self):
        """检查RAG服务配置"""
        print("\n🔍 检查RAG服务配置...")
        try:
            # 读取basic_settings.yaml获取RAG URL
            basic_settings_path = self.base_path / "src" / "config" / "basic_settings.yaml"
            with open(basic_settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            rag_url = settings.get("BASIC_SETTINGS", {}).get("RAG_URL", "")
            if rag_url:
                print(f"✅ RAG URL配置: {rag_url}")
                self.success_count += 1
                
                # 尝试ping RAG服务
                try:
                    import aiohttp
                    import asyncio
                    
                    async def test_rag_endpoint():
                        async with aiohttp.ClientSession() as session:
                            try:
                                async with session.get(f"{rag_url}/health", timeout=5) as response:
                                    if response.status == 200:
                                        print("✅ RAG服务连接正常")
                                        return True
                            except:
                                pass
                        return False
                    
                    # 运行异步测试
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    rag_available = loop.run_until_complete(test_rag_endpoint())
                    loop.close()
                    
                    if not rag_available:
                        self.warnings.append("⚠️  RAG服务可能不可用，请检查服务状态")
                    
                except ImportError:
                    self.warnings.append("⚠️  无法测试RAG服务连接(aiohttp未安装)")
            else:
                self.issues.append("❌ RAG URL未配置")
                
        except Exception as e:
            self.warnings.append(f"⚠️  无法读取RAG配置: {e}")
            
    def check_directories(self):
        """检查必要目录"""
        print("\n🔍 检查必要目录...")
        directories = [
            "data/sales_rag",
            "outputs/grpo_dual_checkpoints", 
            "cache/scoring",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            if dir_path.exists():
                print(f"✅ {directory}")
                self.success_count += 1
            else:
                print(f"📝 创建目录: {directory}")
                dir_path.mkdir(parents=True, exist_ok=True)
                
    def check_gpu(self):
        """检查GPU"""
        print("\n🔍 检查GPU...")
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ nvidia-smi可用")
                self.success_count += 1
                
                # 解析GPU信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Tesla" in line or "A100" in line or "H100" in line or "RTX" in line:
                        print(f"✅ GPU型号: {line.strip()}")
                        break
            else:
                self.warnings.append("⚠️  nvidia-smi不可用，请检查CUDA安装")
        except FileNotFoundError:
            self.warnings.append("⚠️  nvidia-smi未找到，请检查CUDA安装")
            
    def run_all_checks(self):
        """运行所有检查"""
        print("🚀 开始GRPO训练环境检查...")
        print("=" * 50)
        
        self.check_python_version()
        self.check_dependencies() 
        self.check_model_files()
        self.check_config_files()
        self.check_data_files()
        self.check_rag_service()
        self.check_directories()
        self.check_gpu()
        
        print("\n" + "=" * 50)
        print("📊 检查结果汇总:")
        print(f"✅ 成功项: {self.success_count}")
        print(f"❌ 问题项: {len(self.issues)}")
        print(f"⚠️  警告项: {len(self.warnings)}")
        
        if self.issues:
            print("\n❌ 发现问题:")
            for issue in self.issues:
                print(f"  • {issue}")
                
        if self.warnings:
            print("\n⚠️  警告:")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        print("\n" + "=" * 50)
        
        if self.issues:
            print("❌ 发现未解决的问题，请修复后重新运行检查")
            return False
        else:
            print("✅ 环境检查通过！可以开始训练")
            return True

def main():
    """主函数"""
    checker = GRPOSetupChecker()
    success = checker.run_all_checks()
    
    if success:
        print("\n🎉 下一步操作:")
        print("1. 运行数据预处理:")
        print("   bash scripts/start_grpo_training.sh --help")
        print("2. 启动训练:")
        print("   bash scripts/start_grpo_training.sh single")
        print("   或")
        print("   bash scripts/start_grpo_training.sh ray")
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()