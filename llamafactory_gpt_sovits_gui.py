#!/usr/bin/env python3

import sys
import os
import requests
from datetime import datetime
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, 
    QMessageBox, QProgressBar, QFileDialog, QComboBox, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor

# 添加LlamaFactory的src目录到Python路径
llamafactory_path = r"C:\Git Bash-x64_5.1.5\LlamaFactory\src"
sys.path.append(llamafactory_path)

# 导入LlamaFactory
try:
    from llamafactory.chat import ChatModel
    print("成功导入LlamaFactory!")
except Exception as e:
    print(f"导入LlamaFactory失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class ChatThread(QThread):
    """处理与模型对话的线程"""
    new_text = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, chat_model, messages, query):
        super().__init__()
        self.chat_model = chat_model
        self.messages = messages
        self.query = query
    
    def run(self):
        try:
            print("=== ChatThread开始运行 ===")
            print(f"聊天模型: {self.chat_model}")
            print(f"当前消息列表: {self.messages}")
            print(f"用户查询: {self.query}")
            
            # 添加用户消息
            self.messages.append({"role": "user", "content": self.query})
            print(f"添加用户消息后: {self.messages}")
            
            # 流式获取模型响应
            print("开始流式获取模型响应...")
            response = ""
            try:
                for new_text in self.chat_model.stream_chat(self.messages):
                    print(f"收到新文本: {new_text}")
                    self.new_text.emit(new_text)
                    response += new_text
            except Exception as stream_error:
                print(f"=== 流式获取响应错误: {stream_error} ===")
                # 尝试使用非流式方式获取响应
                try:
                    print("尝试使用非流式方式获取响应...")
                    response = self.chat_model.chat(self.messages)
                    print(f"非流式响应: {response}")
                except Exception as chat_error:
                    print(f"=== 非流式获取响应错误: {chat_error} ===")
                    raise
            
            print(f"完整响应: {response}")
            
            # 检查响应是否为空
            if not response:
                print("响应为空，触发错误信号")
                self.error.emit("模型未返回任何响应")
                return
            
            # 添加助手消息
            self.messages.append({"role": "assistant", "content": response})
            print(f"添加助手消息后: {self.messages}")
            
            self.finished.emit(response)
            print("=== ChatThread运行完成 ===")
        except Exception as e:
            print(f"=== ChatThread错误: {e} ===")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class SpeechThread(QThread):
    """处理语音生成的线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, bool)  # 第二个参数表示是否保存了音频
    error = pyqtSignal(str)
    
    def __init__(self, text, language, gpt_sovits_url, refer_wav_path, prompt_text, prompt_language, save_audio=False, save_path=None):
        super().__init__()
        self.text = text
        self.language = language
        self.gpt_sovits_url = gpt_sovits_url
        self.refer_wav_path = refer_wav_path
        self.prompt_text = prompt_text
        self.prompt_language = prompt_language
        self.save_audio = save_audio
        self.save_path = save_path
    
    def run(self):
        import time
        # 构建请求参数
        data = {
            "text": self.text,
            "text_language": self.language,
            "refer_wav_path": self.refer_wav_path,
            "prompt_text": self.prompt_text,
            "prompt_language": self.prompt_language
        }
        
        print(f"发送语音生成请求，参数: {data}")
        print(f"参考音频路径: {self.refer_wav_path}")
        print(f"参考音频文本: {self.prompt_text}")
        print(f"参考音频语言: {self.prompt_language}")
        print(f"是否保存音频: {self.save_audio}")
        if self.save_audio:
            print(f"保存路径: {self.save_path}")
        
        # 发送请求
        max_retries = 3
        retry_interval = 2
        
        for retry in range(max_retries):
            try:
                print(f"发送语音生成请求 (尝试 {retry+1}/{max_retries})...")
                response = requests.post(self.gpt_sovits_url, json=data, timeout=180, stream=False)
                
                print(f"API响应状态码: {response.status_code}")
                print(f"API响应头: {response.headers}")
                
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    print(f"响应内容类型: {content_type}")
                    
                    if 'audio' in content_type or 'wav' in content_type:
                        # 先将整个响应内容读入内存
                        try:
                            audio_data = response.content
                            total_bytes = len(audio_data)
                            print(f"音频数据大小: {total_bytes} bytes")
                            
                            # 检查音频数据是否为空
                            if total_bytes == 0:
                                print("错误: 音频数据为空")
                                self.error.emit("错误: 音频数据为空")
                                if retry < max_retries - 1:
                                    print(f"重试中... ({retry+1}/{max_retries-1})")
                                    time.sleep(retry_interval)
                                    continue
                                else:
                                    return
                        except Exception as e:
                            print(f"读取响应内容错误: {e}")
                            self.error.emit(f"读取响应内容错误: {str(e)}")
                            if retry < max_retries - 1:
                                print(f"重试中... ({retry+1}/{max_retries-1})")
                                time.sleep(retry_interval)
                                continue
                            else:
                                return
                        
                        # 生成音频文件路径
                        if self.save_audio:
                            # 如果指定了保存路径，使用指定路径
                            if self.save_path:
                                audio_file = self.save_path
                                # 检查路径是否为目录
                                import os
                                if os.path.isdir(audio_file):
                                    # 如果是目录，生成文件名
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    audio_file = os.path.join(audio_file, f"output_{timestamp}.wav")
                                # 确保目录存在
                                directory = os.path.dirname(audio_file)
                                if directory and not os.path.exists(directory):
                                    os.makedirs(directory, exist_ok=True)
                            else:
                                # 否则使用默认路径
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                audio_file = f"output_{timestamp}.wav"
                        else:
                            # 不保存时使用临时文件
                            import tempfile
                            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            audio_file = temp_file.name
                            temp_file.close()
                        
                        # 写入文件
                        try:
                            with open(audio_file, "wb") as f:
                                f.write(audio_data)
                                # 发送进度
                                self.progress.emit(100)
                        except Exception as e:
                            # 如果保存失败，使用临时文件
                            import tempfile
                            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            temp_audio_file = temp_file.name
                            temp_file.close()
                            
                            with open(temp_audio_file, "wb") as f:
                                f.write(audio_data)
                                # 发送进度
                                self.progress.emit(100)
                            
                            audio_file = temp_audio_file
                            self.error.emit(f"保存到指定路径失败，使用临时文件: {str(e)}")
                        
                        print(f"音频文件已生成: {audio_file}")
                        print(f"音频文件大小: {total_bytes} bytes")
                        
                        # 发送完成信号，第二个参数表示是否保存了音频
                        self.finished.emit(audio_file, self.save_audio)
                        return
                    else:
                        try:
                            error_data = response.json()
                            print(f"API错误响应: {error_data}")
                            self.error.emit(str(error_data))
                        except:
                            error_text = response.text[:200]
                            print(f"API错误响应: {error_text}")
                            self.error.emit(f"未知错误，响应内容: {error_text}...")
                        return
                else:
                    try:
                        error_data = response.json()
                        print(f"API错误响应: {error_data}")
                        self.error.emit(str(error_data))
                    except:
                        error_text = response.text[:200]
                        print(f"API错误响应: {error_text}")
                        self.error.emit(f"状态码 {response.status_code}，响应内容: {error_text}...")
                    return
            except requests.exceptions.RequestException as e:
                print(f"网络错误: {e}")
                if "Response ended prematurely" in str(e) or "Connection reset by peer" in str(e):
                    if retry < max_retries - 1:
                        print(f"响应提前结束，重试中... ({retry+1}/{max_retries-1})")
                        time.sleep(retry_interval)
                        continue
                    else:
                        self.error.emit(f"网络错误: {str(e)}")
                else:
                    self.error.emit(f"网络错误: {str(e)}")
                return
            except Exception as e:
                print(f"其他错误: {e}")
                import traceback
                traceback.print_exc()
                self.error.emit(str(e))
                return

class ModelLoadThread(QThread):
    """模型加载线程"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, parent, model_path, adapter_path):
        super().__init__(parent)
        self.parent = parent
        self.model_path = model_path
        self.adapter_path = adapter_path
    
    def run(self):
        try:
            # 构建参数
            model_name_or_path = self.model_path
            adapter_name_or_path = self.adapter_path
            gpt_sovits_url = "http://127.0.0.1:9880"
            
            print(f"=== 开始加载模型 ===")
            print(f"模型路径: {model_name_or_path}")
            print(f"适配器路径: {adapter_name_or_path}")
            
            args = {
                "model_name_or_path": model_name_or_path,
                "use_fast_tokenizer": False,
                "trust_remote_code": True,
                "max_length": self.parent.context_length,
                "max_new_tokens": self.parent.max_new_tokens,
                "top_p": self.parent.top_p,
                "temperature": self.parent.temperature
            }
            
            if adapter_name_or_path:
                args["adapter_name_or_path"] = adapter_name_or_path
                args["finetuning_type"] = "lora"
                print(f"使用LoRA适配器: {adapter_name_or_path}")
            
            print(f"模型参数: {args}")
            
            # 初始化ChatModel
            print("正在初始化ChatModel...")
            self.parent.chat_model = ChatModel(args)
            print("ChatModel初始化成功!")
            
            self.parent.messages = []
            self.parent.gpt_sovits_url = gpt_sovits_url
            
            # 保存当前模型和适配器路径
            self.parent.model_path = model_name_or_path
            self.parent.adapter_path = adapter_name_or_path
            
            print("=== 模型加载完成 ===")
            self.finished.emit()
        except Exception as e:
            print(f"=== 模型加载错误: {e} ===")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class LlamaFactoryGPTSoVITSGUI(QMainWindow):
    """LlamaFactory + GPT-SoVITS 集成系统GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LlamaFactory + GPT-SoVITS 集成系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化变量
        self.chat_model = None
        self.messages = []
        self.gpt_sovits_url = "http://127.0.0.1:9880"
        # 参考音频设置
        self.refer_wav_path = r"D:\音频样本\已处理音频\切分\零修xl\Ref\vocal_读评论.mp3_10.wav_0000000000_0000176640.wav"
        self.prompt_text = "读评论。up的声音是直接真实的声音吗？其实，我原本的声音是这样子的。"
        self.prompt_language = "zh"
        self.output_language = "zh"
        # 音频保存设置
        self.save_audio = False
        self.save_path = ""
        # 模型设置
        self.context_length = 4096  # 默认上下文长度
        self.max_new_tokens = 1024  # 默认最大生成长度
        self.top_p = 0.95  # 默认Top-p采样值
        self.temperature = 0.6  # 默认温度系数
        # GPT-SoVITS设置
        self.api_py_path = r"d:\GPT-SoVITS-v4-20250422-nvidia50\api.py"  # 默认api.py路径
        self.python_exe_path = r"d:\GPT-SoVITS-v4-20250422-nvidia50\runtime\python.exe"  # 默认python.exe路径
        # GPT-SoVITS进程
        self.gpt_sovits_process = None
        
        # 加载默认设置
        self.load_default_settings()
        
        # 添加菜单栏
        self.menu_bar = self.menuBar()
        
        # 对话菜单
        conversation_menu = self.menu_bar.addMenu("对话")
        
        # 清空历史记录动作
        clear_action = QAction("清空历史记录", self)
        clear_action.triggered.connect(self.clear_chat)
        conversation_menu.addAction(clear_action)
        
        # 设置菜单
        settings_menu = self.menu_bar.addMenu("设置")
        
        # 打开设置对话框动作
        open_settings_action = QAction("打开设置", self)
        open_settings_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(open_settings_action)
        
        # 创建主部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建对话显示区域
        self.create_chat_display()
        
        # 创建输入区域
        self.create_input_area()
        
        # 创建状态区域
        self.create_status_area()
        
        # 显示初始消息
        self.append_message("系统", "欢迎使用LlamaFactory + GPT-SoVITS集成系统！")
        self.append_message("系统", "请在设置中配置GPT-SoVITS路径、api.py路径和python.exe路径，然后手动启动GPT-SoVITS服务")
        
        # 显示加载模型消息
        self.append_message("系统", "正在加载模型，请稍候...")
        
        # 在后台线程中加载模型
        self.load_model_in_background()
    
    def create_chat_display(self):
        """创建对话显示区域"""
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 创建文本编辑框
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 10px;
                font-family: Arial;
                font-size: 14px;
            }
        """)
        
        scroll_area.setWidget(self.chat_display)
        self.main_layout.addWidget(scroll_area, 1)
    
    def create_input_area(self):
        """创建输入区域"""
        input_layout = QHBoxLayout()
        
        # 创建输入框
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入消息...")
        self.input_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                flex: 1;
            }
        """)
        self.input_edit.returnPressed.connect(self.send_message)
        
        # 创建发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                margin-left: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        # 默认禁用发送按钮，直到模型加载完成
        self.send_button.setEnabled(False)
        
        # 创建清除按钮
        self.clear_button = QPushButton("清除")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                margin-left: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_button.clicked.connect(self.clear_chat)
        
        input_layout.addWidget(self.input_edit, 1)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.clear_button)
        
        self.main_layout.addLayout(input_layout)
    
    def create_status_area(self):
        """创建状态区域"""
        status_layout = QHBoxLayout()
        
        # 创建状态标签
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
            }
        """)
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                text-align: center;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar, 2)
        
        self.main_layout.addLayout(status_layout)
    
    def load_model_in_background(self):
        """在后台线程中加载模型"""
        self.status_label.setText("状态: 加载模型中...")
        
        # 获取模型和适配器路径
        model_path = getattr(self, 'model_path', r"C:\Git Bash-x64_5.1.5\Qwen3-0.6B\qwen\Qwen3-0___6B")
        adapter_path = getattr(self, 'adapter_path', r"C:\Git Bash-x64_5.1.5\saves\Qwen3-0.6B-Base\lora\train_2026-02-28-19-28-59")
        
        # 创建并启动模型加载线程
        self.model_load_thread = ModelLoadThread(self, model_path, adapter_path)
        self.model_load_thread.finished.connect(self.on_model_load_finished)
        self.model_load_thread.error.connect(self.on_model_load_error)
        self.model_load_thread.start()
    
    def reload_model(self, model_path=None, adapter_path=None):
        """重新加载模型"""
        # 如果没有提供参数，使用当前设置的路径
        if model_path is None:
            model_path = self.model_path
        if adapter_path is None:
            adapter_path = self.adapter_path
        
        if not model_path:
            QMessageBox.warning(self, "警告", "请输入模型路径")
            return
        
        self.status_label.setText("状态: 重新加载模型中...")
        
        # 创建并启动模型加载线程
        self.model_load_thread = ModelLoadThread(self, model_path, adapter_path)
        self.model_load_thread.finished.connect(self.on_model_reload_finished)
        self.model_load_thread.error.connect(self.on_model_load_error)
        self.model_load_thread.start()
    
    def on_model_reload_finished(self):
        """模型重新加载完成回调"""
        self.status_label.setText("状态: 就绪")
        self.append_message("系统", "模型重新加载完成，准备就绪！")
        # 启用发送按钮
        self.send_button.setEnabled(True)
    
    def on_model_load_finished(self):
        """模型加载完成回调"""
        self.status_label.setText("状态: 就绪")
        self.append_message("系统", "模型加载完成，准备就绪！")
        # 启用发送按钮
        self.send_button.setEnabled(True)
    
    def on_model_load_error(self, error):
        """模型加载错误回调"""
        self.status_label.setText("状态: 加载失败")
        self.append_message("系统", f"模型加载失败: {error}")
        QMessageBox.critical(self, "错误", f"模型加载失败: {error}")
    

    
    def load_default_settings(self):
        """加载默认设置"""
        try:
            import json
            import os
            
            if os.path.exists("default_settings.json"):
                with open("default_settings.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 应用默认设置
                if "refer_wav_path" in config:
                    self.refer_wav_path = config["refer_wav_path"]
                if "prompt_text" in config:
                    self.prompt_text = config["prompt_text"]
                if "prompt_language" in config:
                    self.prompt_language = config["prompt_language"]
                if "output_language" in config:
                    self.output_language = config["output_language"]
                if "save_audio" in config:
                    self.save_audio = config["save_audio"]
                if "save_path" in config:
                    self.save_path = config["save_path"]
                if "context_length" in config:
                    self.context_length = config["context_length"]
                if "max_new_tokens" in config:
                    self.max_new_tokens = config["max_new_tokens"]
                if "top_p" in config:
                    self.top_p = config["top_p"]
                if "temperature" in config:
                    self.temperature = config["temperature"]
                if "model_path" in config:
                    self.model_path = config["model_path"]
                if "adapter_path" in config:
                    self.adapter_path = config["adapter_path"]
                if "api_py_path" in config:
                    self.api_py_path = config["api_py_path"]
                if "python_exe_path" in config:
                    self.python_exe_path = config["python_exe_path"]
                
                print("默认设置已加载")
        except Exception as e:
            print(f"加载默认设置失败: {e}")
    
    def start_gpt_sovits(self):
        """启动GPT-SoVITS"""
        import subprocess
        import os
        import time
        import requests
        
        try:
            # 获取用户设置的路径
            api_py_path = self.api_py_path
            python_exe_path = self.python_exe_path
            
            # 检查api.py文件是否存在
            if not os.path.exists(api_py_path):
                self.append_message("系统", f"api.py文件不存在: {api_py_path}")
                return False
            
            # 检查python.exe文件是否存在
            if not os.path.exists(python_exe_path):
                self.append_message("系统", f"python.exe文件不存在: {python_exe_path}")
                return False
            
            # 获取api.py所在的目录作为工作目录
            gpt_sovits_path = os.path.dirname(api_py_path)
            
            print(f"=== 启动GPT-SoVITS ===")
            print(f"工作目录: {gpt_sovits_path}")
            print(f"Python: {python_exe_path}")
            print(f"API文件: {api_py_path}")
            
            # 直接运行Python命令启动API
            self.gpt_sovits_process = subprocess.Popen(
                [python_exe_path, api_py_path],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=gpt_sovits_path
            )
            
            # 等待服务启动
            self.append_message("系统", "正在启动GPT-SoVITS服务...")
            
            # 检查服务是否启动成功
            max_retries = 10  # 减少到10秒，快速完成启动流程
            retry_interval = 1
            service_started = False
            for i in range(max_retries):
                try:
                    response = requests.get("http://127.0.0.1:9880", timeout=1)
                    if response.status_code == 200:
                        self.append_message("系统", "GPT-SoVITS服务启动成功！")
                        service_started = True
                        break
                except:
                    pass
                time.sleep(retry_interval)
            
            # 无论是否检测到服务启动，都认为服务已启动
            # 因为即使显示连接失败，API仍然可以使用
            self.append_message("系统", "GPT-SoVITS服务已启动！")
            return True
            
        except Exception as e:
            print(f"启动GPT-SoVITS失败: {e}")
            import traceback
            traceback.print_exc()
            self.append_message("系统", f"启动GPT-SoVITS失败: {str(e)}")
            return False
    
    def stop_gpt_sovits(self):
        """停止GPT-SoVITS"""
        if self.gpt_sovits_process:
            try:
                # 终止进程
                print("开始终止GPT-SoVITS进程...")
                self.gpt_sovits_process.terminate()
                
                # 等待进程终止
                try:
                    self.gpt_sovits_process.wait(timeout=15)
                    print(f"GPT-SoVITS进程已终止，退出码: {self.gpt_sovits_process.returncode}")
                except subprocess.TimeoutExpired:
                    # 如果超时，强制终止进程
                    print("GPT-SoVITS进程终止超时，强制终止")
                    self.gpt_sovits_process.kill()
                    try:
                        self.gpt_sovits_process.wait(timeout=10)
                        print("GPT-SoVITS进程已强制终止")
                    except subprocess.TimeoutExpired:
                        print("GPT-SoVITS进程强制终止失败")
                
                # 清空进程对象
                self.gpt_sovits_process = None
                
                # 尝试清理端口（使用netstat和taskkill命令）
                print("尝试清理9880端口...")
                try:
                    # 使用netstat查找占用9880端口的进程
                    import subprocess
                    netstat_output = subprocess.check_output(["netstat", "-ano"], shell=True, text=True)
                    for line in netstat_output.split('\n'):
                        if "9880" in line and "LISTENING" in line:
                            # 提取进程ID
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                print(f"发现占用9880端口的进程: PID {pid}")
                                # 尝试终止该进程
                                try:
                                    subprocess.run(["taskkill", "/F", "/PID", pid], shell=True, check=True)
                                    print(f"已终止占用9880端口的进程: PID {pid}")
                                except subprocess.CalledProcessError as e:
                                    print(f"终止进程失败: {e}")
                except Exception as port_error:
                    print(f"清理端口时出错: {port_error}")
                
                self.append_message("系统", "GPT-SoVITS已停止")
                
            except Exception as e:
                print(f"停止GPT-SoVITS失败: {e}")
                import traceback
                traceback.print_exc()
                self.append_message("系统", f"停止GPT-SoVITS失败: {str(e)}")
    
    def append_message(self, sender, message):
        """在聊天显示区域添加消息"""
        if sender == "用户":
            self.chat_display.append(f"<b>用户:</b> {message}")
        elif sender == "助手":
            self.chat_display.append(f"<b>助手:</b> {message}")
        else:
            self.chat_display.append(f"<i>{sender}:</i> {message}")
        
        # 滚动到底部
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
    
    def send_message(self):
        """发送消息"""
        print("=== 开始处理消息 ===")
        query = self.input_edit.text().strip()
        if not query:
            print("消息为空，跳过处理")
            return
        
        # 检查是否有正在运行的对话线程
        if hasattr(self, 'chat_thread') and self.chat_thread.isRunning():
            print("有对话线程正在运行，跳过处理")
            return
        
        # 清空输入框
        self.input_edit.clear()
        print(f"用户输入: {query}")
        
        # 添加用户消息
        self.append_message("用户", query)
        print("用户消息已添加到聊天显示")
        
        # 禁用发送按钮
        self.send_button.setEnabled(False)
        self.status_label.setText("状态: 处理中...")
        print("发送按钮已禁用，状态已更新为处理中")
        
        # 启动对话线程
        print("准备启动ChatThread线程...")
        self.chat_thread = ChatThread(self.chat_model, self.messages, query)
        self.chat_thread.new_text.connect(self.on_new_text)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.error.connect(self.on_chat_error)
        self.chat_thread.start()
        print("ChatThread线程已启动")
    
    def on_new_text(self, text):
        """处理新文本"""
        # 这里可以实现流式显示，暂时简单处理
        pass
    
    def on_chat_finished(self, response):
        """处理对话完成"""
        print(f"=== 对话完成，响应: {response} ===")
        
        # 处理响应，提取answer部分
        processed_response = response
        
        try:
            # 检查响应中是否包含think和answer部分（不区分大小写）
            response_lower = response.lower()
            if "think" in response_lower and "answer" in response_lower:
                # 尝试提取answer部分
                try:
                    # 查找answer部分的开始位置（不区分大小写）
                    answer_start = response_lower.find("answer")
                    if answer_start != -1:
                        # 查找answer部分的内容
                        # 尝试多种格式："answer:", "answer：", "Answer:", "ANSWER:"等
                        # 查找冒号，从answer_start开始
                        colon_index = response.find(":", answer_start)
                        if colon_index != -1:
                            # 提取冒号后的内容
                            processed_response = response[colon_index + 1:].strip()
                            # 移除可能的空格和换行符
                            processed_response = processed_response.replace('\n', ' ').replace('\r', ' ').strip()
                            print(f"提取到answer部分: {processed_response}")
                        else:
                            # 如果没有冒号，尝试其他格式
                            print("未找到冒号，使用原始响应")
                except Exception as e:
                    print(f"处理响应时出错: {e}")
                    # 如果处理出错，使用原始响应
                    pass
            else:
                print("响应中不包含think和answer部分，使用原始响应")
            
            # 确保处理后的响应不为空
            if not processed_response:
                processed_response = response
                print("处理后的响应为空，使用原始响应")
            
            # 添加助手消息到聊天显示
            self.append_message("助手", processed_response)
            print("助手消息已添加到聊天显示")
            
            # 生成语音
            print("准备生成语音...")
            # 检查是否需要保存音频
            save_audio = hasattr(self, 'save_audio_checkbox') and self.save_audio_checkbox.isChecked()
            save_path = self.save_path_edit.text() if save_audio else None
            print(f"保存音频: {save_audio}")
            if save_audio:
                print(f"保存路径: {save_path}")
            self.generate_speech(processed_response, save_audio, save_path)
            print("语音生成已启动")
        except Exception as e:
            print(f"处理对话完成时出错: {e}")
            import traceback
            traceback.print_exc()
            # 即使出错也要显示原始响应
            self.append_message("助手", response)
            self.append_message("系统", f"处理响应时出错: {str(e)}")
        finally:
            # 启用发送按钮
            self.send_button.setEnabled(True)
            self.status_label.setText("状态: 就绪")
            print("发送按钮已启用，状态已更新为就绪")
    
    def on_chat_error(self, error):
        """处理对话错误"""
        print(f"=== 对话错误: {error} ===")
        self.append_message("系统", f"对话错误: {error}")
        self.send_button.setEnabled(True)
        self.status_label.setText("状态: 就绪")
        print("发送按钮已启用，状态已更新为就绪")
    
    def generate_speech(self, text, save_audio=False, save_path=None):
        """生成语音"""
        # 检查是否有正在运行的语音生成线程
        if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
            print("有语音生成线程正在运行，跳过处理")
            return
        
        self.status_label.setText("状态: 生成语音中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 启动语音生成线程
        self.speech_thread = SpeechThread(
            text, 
            self.output_language, 
            self.gpt_sovits_url,
            self.refer_wav_path,
            self.prompt_text,
            self.prompt_language,
            save_audio,
            save_path
        )
        self.speech_thread.progress.connect(self.on_speech_progress)
        self.speech_thread.finished.connect(self.on_speech_finished)
        self.speech_thread.error.connect(self.on_speech_error)
        self.speech_thread.start()
    
    def on_speech_progress(self, progress):
        """处理语音生成进度"""
        self.progress_bar.setValue(progress)
    
    def on_speech_finished(self, audio_file, is_saved):
        """处理语音生成完成"""
        print(f"=== 语音生成完成，文件: {audio_file} ===")
        print(f"是否保存: {is_saved}")
        
        self.status_label.setText("状态: 播放音频中...")
        self.progress_bar.setValue(100)
        
        # 播放音频
        try:
            if os.name == "nt":  # Windows
                import winsound
                winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                print("使用winsound播放音频")
            else:  # Linux/macOS
                # 对于非Windows系统，使用系统命令播放
                os.system(f"play {audio_file}")
                print("使用系统命令播放音频")
        except Exception as e:
            print(f"播放音频时出错: {e}")
            # 如果播放失败，尝试使用系统默认播放器
            if os.name == "nt":
                os.startfile(audio_file)
                print("使用系统默认播放器播放音频")
        
        # 如果不是保存的音频，删除临时文件
        if not is_saved:
            try:
                # 等待音频播放完成后再删除
                import time
                time.sleep(2)  # 等待2秒让音频开始播放
                os.remove(audio_file)
                print(f"临时音频文件已删除: {audio_file}")
            except Exception as e:
                print(f"删除临时文件时出错: {e}")
        
        self.status_label.setText("状态: 就绪")
        self.progress_bar.setVisible(False)
        if is_saved:
            self.append_message("系统", f"语音已生成并保存: {audio_file}")
        else:
            self.append_message("系统", "语音已生成并播放")
    
    def on_speech_error(self, error):
        """处理语音生成错误"""
        self.status_label.setText("状态: 就绪")
        self.progress_bar.setVisible(False)
        self.append_message("系统", f"语音生成错误: {error}")
    
    def clear_chat(self):
        """清除聊天记录"""
        self.chat_display.clear()
        self.messages = []
        self.append_message("系统", "历史记录已清除。")
    
    def open_settings_dialog(self):
        """打开设置对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QGridLayout, QSlider
        from PyQt5.QtCore import Qt
        
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("设置")
        dialog.setGeometry(200, 200, 800, 600)
        
        # 创建主布局
        main_layout = QVBoxLayout(dialog)
        
        # 参考音频设置分组
        audio_group = QGroupBox("参考音频设置")
        audio_layout = QVBoxLayout()
        
        # 参考音频路径
        refer_wav_layout = QHBoxLayout()
        refer_wav_label = QLabel("参考音频路径:")
        refer_wav_edit = QLineEdit(self.refer_wav_path)
        refer_wav_edit.setMinimumWidth(400)
        refer_wav_browse_button = QPushButton("浏览")
        refer_wav_browse_button.clicked.connect(lambda: self.browse_file(refer_wav_edit, "选择参考音频文件", "音频文件 (*.wav *.mp3 *.flac)"))
        refer_wav_layout.addWidget(refer_wav_label)
        refer_wav_layout.addWidget(refer_wav_edit)
        refer_wav_layout.addWidget(refer_wav_browse_button)
        audio_layout.addLayout(refer_wav_layout)
        
        # 参考音频对应文本
        prompt_text_layout = QHBoxLayout()
        prompt_text_label = QLabel("参考音频对应文本:")
        prompt_text_edit = QLineEdit(self.prompt_text)
        prompt_text_edit.setMinimumWidth(400)
        prompt_text_layout.addWidget(prompt_text_label)
        prompt_text_layout.addWidget(prompt_text_edit)
        audio_layout.addLayout(prompt_text_layout)
        
        # 参考音频语言选择
        prompt_language_layout = QHBoxLayout()
        prompt_language_label = QLabel("参考音频语言:")
        prompt_language_combo = QComboBox()
        prompt_language_combo.addItems(["zh", "en", "ja", "ko", "fr", "de", "es", "ru"])
        prompt_language_combo.setCurrentText(self.prompt_language)
        prompt_language_layout.addWidget(prompt_language_label)
        prompt_language_layout.addWidget(prompt_language_combo)
        audio_layout.addLayout(prompt_language_layout)
        
        # 输出语音语言选择
        output_language_layout = QHBoxLayout()
        output_language_label = QLabel("输出语音语言:")
        output_language_combo = QComboBox()
        output_language_combo.addItems(["zh", "en", "ja", "ko", "fr", "de", "es", "ru"])
        output_language_combo.setCurrentText(self.output_language)
        output_language_layout.addWidget(output_language_label)
        output_language_layout.addWidget(output_language_combo)
        audio_layout.addLayout(output_language_layout)
        
        # 音频保存设置
        save_audio_layout = QHBoxLayout()
        save_audio_checkbox = QCheckBox("保存生成的音频")
        save_audio_checkbox.setChecked(self.save_audio)
        save_path_edit = QLineEdit(self.save_path)
        save_path_edit.setMinimumWidth(300)
        save_browse_button = QPushButton("浏览")
        save_browse_button.clicked.connect(lambda: self.browse_save_file(save_path_edit, "保存音频文件", "WAV文件 (*.wav)"))
        save_audio_layout.addWidget(save_audio_checkbox)
        save_audio_layout.addWidget(save_path_edit)
        save_audio_layout.addWidget(save_browse_button)
        audio_layout.addLayout(save_audio_layout)
        
        # 应用按钮
        apply_button = QPushButton("应用")
        apply_button.clicked.connect(lambda: self.apply_dialog_settings(
            refer_wav_edit.text(),
            prompt_text_edit.text(),
            prompt_language_combo.currentText(),
            output_language_combo.currentText(),
            save_audio_checkbox.isChecked(),
            save_path_edit.text()
        ))
        audio_layout.addWidget(apply_button)
        
        # 设为默认按钮
        default_button = QPushButton("设为默认")
        default_button.clicked.connect(lambda: self.set_dialog_default(
            refer_wav_edit.text(),
            prompt_text_edit.text(),
            prompt_language_combo.currentText(),
            output_language_combo.currentText(),
            save_audio_checkbox.isChecked(),
            save_path_edit.text(),
            model_path_edit.text(),
            adapter_path_edit.text(),
            int(context_length_edit.text()) if context_length_edit.text().isdigit() else 4096,
            max_new_tokens_slider.value(),
            round(top_p_slider.value()/100, 2),
            round(temperature_slider.value()/100, 2),
            api_py_path_edit.text(),
            python_exe_path_edit.text()
        ))
        audio_layout.addWidget(default_button)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # 模型设置分组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        # 模型路径
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("模型路径:")
        model_path_edit = QLineEdit(self.model_path)
        model_path_edit.setMinimumWidth(400)
        model_browse_button = QPushButton("浏览")
        model_browse_button.clicked.connect(lambda: self.browse_directory(model_path_edit, "选择模型目录"))
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(model_path_edit)
        model_path_layout.addWidget(model_browse_button)
        model_layout.addLayout(model_path_layout)
        
        # 适配器路径
        adapter_path_layout = QHBoxLayout()
        adapter_path_label = QLabel("适配器路径:")
        adapter_path_edit = QLineEdit(self.adapter_path)
        adapter_path_edit.setMinimumWidth(400)
        adapter_browse_button = QPushButton("浏览")
        adapter_browse_button.clicked.connect(lambda: self.browse_directory(adapter_path_edit, "选择适配器目录"))
        adapter_path_layout.addWidget(adapter_path_label)
        adapter_path_layout.addWidget(adapter_path_edit)
        adapter_path_layout.addWidget(adapter_browse_button)
        model_layout.addLayout(adapter_path_layout)
        
        # 上下文长度设置
        context_length_layout = QHBoxLayout()
        context_length_label = QLabel("Context length:")
        context_length_edit = QLineEdit(str(self.context_length))
        context_length_edit.setMinimumWidth(100)
        context_length_layout.addWidget(context_length_label)
        context_length_layout.addWidget(context_length_edit)
        model_layout.addLayout(context_length_layout)
        
        # 最大生成长度设置
        max_new_tokens_layout = QHBoxLayout()
        max_new_tokens_label = QLabel("最大生成长度:")
        max_new_tokens_slider = QSlider(Qt.Horizontal)
        max_new_tokens_slider.setMinimum(8)
        max_new_tokens_slider.setMaximum(8192)
        max_new_tokens_slider.setValue(self.max_new_tokens)
        max_new_tokens_slider.setTickInterval(256)
        max_new_tokens_slider.setTickPosition(QSlider.TicksBelow)
        max_new_tokens_value = QLabel(str(self.max_new_tokens))
        max_new_tokens_slider.valueChanged.connect(lambda value: max_new_tokens_value.setText(str(value)))
        max_new_tokens_layout.addWidget(max_new_tokens_label)
        max_new_tokens_layout.addWidget(max_new_tokens_slider)
        max_new_tokens_layout.addWidget(max_new_tokens_value)
        model_layout.addLayout(max_new_tokens_layout)
        
        # Top-p采样值设置
        top_p_layout = QHBoxLayout()
        top_p_label = QLabel("Top-p采样值:")
        top_p_slider = QSlider(Qt.Horizontal)
        top_p_slider.setMinimum(1)
        top_p_slider.setMaximum(100)
        top_p_slider.setValue(int(self.top_p * 100))
        top_p_slider.setTickInterval(5)
        top_p_slider.setTickPosition(QSlider.TicksBelow)
        top_p_value = QLabel(str(self.top_p))
        top_p_slider.valueChanged.connect(lambda value: top_p_value.setText(str(round(value/100, 2))))
        top_p_layout.addWidget(top_p_label)
        top_p_layout.addWidget(top_p_slider)
        top_p_layout.addWidget(top_p_value)
        model_layout.addLayout(top_p_layout)
        
        # 温度系数设置
        temperature_layout = QHBoxLayout()
        temperature_label = QLabel("温度系数:")
        temperature_slider = QSlider(Qt.Horizontal)
        temperature_slider.setMinimum(1)
        temperature_slider.setMaximum(150)
        temperature_slider.setValue(int(self.temperature * 100))
        temperature_slider.setTickInterval(10)
        temperature_slider.setTickPosition(QSlider.TicksBelow)
        temperature_value = QLabel(str(self.temperature))
        temperature_slider.valueChanged.connect(lambda value: temperature_value.setText(str(round(value/100, 2))))
        temperature_layout.addWidget(temperature_label)
        temperature_layout.addWidget(temperature_slider)
        temperature_layout.addWidget(temperature_value)
        model_layout.addLayout(temperature_layout)
        
        # 重新加载模型按钮
        reload_model_button = QPushButton("重新加载模型")
        reload_model_button.clicked.connect(lambda: self.reload_model(
            model_path_edit.text(), 
            adapter_path_edit.text(), 
            int(context_length_edit.text()) if context_length_edit.text().isdigit() else 4096,
            max_new_tokens_slider.value(),
            round(top_p_slider.value()/100, 2),
            round(temperature_slider.value()/100, 2)
        ))
        model_layout.addWidget(reload_model_button)
        
        # 设为默认按钮
        model_default_button = QPushButton("设为默认")
        model_default_button.clicked.connect(lambda: self.set_model_default(
            model_path_edit.text(), 
            adapter_path_edit.text(), 
            int(context_length_edit.text()) if context_length_edit.text().isdigit() else 4096,
            max_new_tokens_slider.value(),
            round(top_p_slider.value()/100, 2),
            round(temperature_slider.value()/100, 2)
        ))
        model_layout.addWidget(model_default_button)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # GPT-SoVITS设置分组
        gpt_sovits_group = QGroupBox("GPT-SoVITS设置")
        gpt_sovits_layout = QVBoxLayout()
        
        # api.py路径
        api_py_path_layout = QHBoxLayout()
        api_py_path_label = QLabel("api.py路径:")
        api_py_path_edit = QLineEdit(self.api_py_path)
        api_py_path_edit.setMinimumWidth(400)
        api_py_browse_button = QPushButton("浏览")
        api_py_browse_button.clicked.connect(lambda: self.browse_file(api_py_path_edit, "选择api.py文件", "Python文件 (*.py)"))
        api_py_path_layout.addWidget(api_py_path_label)
        api_py_path_layout.addWidget(api_py_path_edit)
        api_py_path_layout.addWidget(api_py_browse_button)
        gpt_sovits_layout.addLayout(api_py_path_layout)
        
        # python.exe路径
        python_exe_path_layout = QHBoxLayout()
        python_exe_path_label = QLabel("python.exe路径:")
        python_exe_path_edit = QLineEdit(self.python_exe_path)
        python_exe_path_edit.setMinimumWidth(400)
        python_exe_browse_button = QPushButton("浏览")
        python_exe_browse_button.clicked.connect(lambda: self.browse_file(python_exe_path_edit, "选择python.exe文件", "可执行文件 (*.exe)"))
        python_exe_path_layout.addWidget(python_exe_path_label)
        python_exe_path_layout.addWidget(python_exe_path_edit)
        python_exe_path_layout.addWidget(python_exe_browse_button)
        gpt_sovits_layout.addLayout(python_exe_path_layout)
        
        # GPT-SoVITS控制按钮
        gpt_sovits_control_layout = QHBoxLayout()
        start_gpt_sovits_button = QPushButton("启动 GPT-SoVITS")
        start_gpt_sovits_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                margin-right: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        start_gpt_sovits_button.clicked.connect(self.start_gpt_sovits)
        
        stop_gpt_sovits_button = QPushButton("停止 GPT-SoVITS")
        stop_gpt_sovits_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        stop_gpt_sovits_button.clicked.connect(self.stop_gpt_sovits)
        stop_gpt_sovits_button.setEnabled(self.gpt_sovits_process is not None)
        
        gpt_sovits_control_layout.addWidget(start_gpt_sovits_button)
        gpt_sovits_control_layout.addWidget(stop_gpt_sovits_button)
        gpt_sovits_layout.addLayout(gpt_sovits_control_layout)
        
        # 设为默认按钮
        gpt_sovits_default_button = QPushButton("设为默认")
        gpt_sovits_default_button.clicked.connect(lambda: self.set_gpt_sovits_default(api_py_path_edit.text(), python_exe_path_edit.text()))
        gpt_sovits_layout.addWidget(gpt_sovits_default_button)
        
        gpt_sovits_group.setLayout(gpt_sovits_layout)
        main_layout.addWidget(gpt_sovits_group)
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        main_layout.addWidget(close_button)
        
        dialog.exec_()
    
    def browse_file(self, line_edit, title, filter):
        """浏览文件"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", filter)
        if file_path:
            line_edit.setText(file_path)
    
    def browse_save_file(self, line_edit, title, filter):
        """浏览保存文件路径"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(self, title, "", filter)
        if file_path:
            line_edit.setText(file_path)
    
    def browse_directory(self, line_edit, title):
        """浏览目录"""
        from PyQt5.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(self, title)
        if directory:
            line_edit.setText(directory)
    
    def apply_dialog_settings(self, refer_wav_path, prompt_text, prompt_language, output_language, save_audio, save_path):
        """应用对话框设置"""
        self.refer_wav_path = refer_wav_path
        self.prompt_text = prompt_text
        self.prompt_language = prompt_language
        self.output_language = output_language
        self.save_audio = save_audio
        self.save_path = save_path
        self.append_message("系统", "参考音频设置已更新")
    
    def set_dialog_default(self, refer_wav_path, prompt_text, prompt_language, output_language, save_audio, save_path, model_path, adapter_path, context_length, max_new_tokens, top_p, temperature, api_py_path, python_exe_path):
        """设置对话框默认值"""
        try:
            # 保存到配置文件
            import json
            config = {
                "refer_wav_path": refer_wav_path,
                "prompt_text": prompt_text,
                "prompt_language": prompt_language,
                "output_language": output_language,
                "save_audio": save_audio,
                "save_path": save_path,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "context_length": context_length,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "temperature": temperature,
                "api_py_path": api_py_path,
                "python_exe_path": python_exe_path
            }
            
            with open("default_settings.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 更新实例变量
            self.save_audio = save_audio
            self.save_path = save_path
            self.context_length = context_length
            # 限制最大生成长度范围为8~8192
            self.max_new_tokens = max(8, min(8192, max_new_tokens))
            # 限制Top-p采样值范围为0.01~1
            self.top_p = max(0.01, min(1.0, top_p))
            # 限制温度系数范围为0.01~1.5
            self.temperature = max(0.01, min(1.5, temperature))
            
            self.append_message("系统", "设置已设为默认")
            print("默认设置已保存")
        except Exception as e:
            self.append_message("系统", f"保存默认设置失败: {str(e)}")
            print(f"保存默认设置失败: {e}")
    
    def set_model_default(self, model_path, adapter_path, context_length=None, max_new_tokens=None, top_p=None, temperature=None):
        """设置模型默认值"""
        try:
            # 读取现有配置
            import json
            import os
            
            config = {}
            if os.path.exists("default_settings.json"):
                with open("default_settings.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
            
            # 更新模型设置
            config["model_path"] = model_path
            config["adapter_path"] = adapter_path
            if context_length is not None:
                config["context_length"] = context_length
            if max_new_tokens is not None:
                config["max_new_tokens"] = max_new_tokens
            if top_p is not None:
                config["top_p"] = top_p
            if temperature is not None:
                config["temperature"] = temperature
            
            # 保存到配置文件
            with open("default_settings.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 更新实例变量
            self.model_path = model_path
            self.adapter_path = adapter_path
            if context_length is not None:
                self.context_length = context_length
            if max_new_tokens is not None:
                # 限制最大生成长度范围为8~8192
                self.max_new_tokens = max(8, min(8192, max_new_tokens))
            if top_p is not None:
                # 限制Top-p采样值范围为0.01~1
                self.top_p = max(0.01, min(1.0, top_p))
            if temperature is not None:
                # 限制温度系数范围为0.01~1.5
                self.temperature = max(0.01, min(1.5, temperature))
            
            self.append_message("系统", "模型设置已设为默认")
            print("模型默认设置已保存")
        except Exception as e:
            self.append_message("系统", f"保存模型默认设置失败: {str(e)}")
            print(f"保存模型默认设置失败: {e}")
    
    def set_gpt_sovits_default(self, api_py_path, python_exe_path):
        """设置GPT-SoVITS默认值"""
        try:
            # 读取现有配置
            import json
            import os
            
            config = {}
            if os.path.exists("default_settings.json"):
                with open("default_settings.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
            
            # 更新GPT-SoVITS设置
            config["api_py_path"] = api_py_path
            config["python_exe_path"] = python_exe_path
            
            # 保存到配置文件
            with open("default_settings.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 更新实例变量
            self.api_py_path = api_py_path
            self.python_exe_path = python_exe_path
            
            self.append_message("系统", "GPT-SoVITS设置已设为默认")
            print("GPT-SoVITS默认设置已保存")
        except Exception as e:
            self.append_message("系统", f"保存GPT-SoVITS默认设置失败: {str(e)}")
            print(f"保存GPT-SoVITS默认设置失败: {e}")
    
    def reload_model(self, model_path, adapter_path, context_length=None, max_new_tokens=None, top_p=None, temperature=None):
        """重新加载模型"""
        self.model_path = model_path
        self.adapter_path = adapter_path
        if context_length is not None:
            self.context_length = context_length
        if max_new_tokens is not None:
            # 限制最大生成长度范围为8~8192
            self.max_new_tokens = max(8, min(8192, max_new_tokens))
        if top_p is not None:
            # 限制Top-p采样值范围为0.01~1
            self.top_p = max(0.01, min(1.0, top_p))
        if temperature is not None:
            # 限制温度系数范围为0.01~1.5
            self.temperature = max(0.01, min(1.5, temperature))
        self.append_message("系统", "正在重新加载模型，请稍候...")
        self.load_model_in_background()
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 当窗口关闭时，停止GPT-SoVITS服务
        if self.gpt_sovits_process:
            print("窗口关闭，正在停止GPT-SoVITS服务...")
            self.stop_gpt_sovits()
        event.accept()

if __name__ == "__main__":
    print("Starting GUI application...")
    app = QApplication(sys.argv)
    print("QApplication created")
    app.setStyle("Fusion")
    print("Style set to Fusion")
    
    # 设置全局样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ffffff;
        }
    """)
    print("Style sheet applied")
    
    window = LlamaFactoryGPTSoVITSGUI()
    print("Window created")
    window.show()
    print("Window shown")
    sys.exit(app.exec_())
