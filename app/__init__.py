from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def create_app(test_config=None):
    """创建并配置Flask应用"""
    app = Flask(__name__, instance_relative_config=True)

    # 配置应用
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
        DASHSCOPE_API_KEY=os.environ.get('DASHSCOPE_API_KEY'),
    )

    if test_config is None:
        # 非测试模式下加载实例配置
        app.config.from_pyfile('config.py', silent=True)
    else:
        # 测试模式下加载测试配置
        app.config.from_mapping(test_config)

    # 确保实例文件夹存在
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # 配置CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # 注册蓝图
    from app.routes import chat_bp
    app.register_blueprint(chat_bp)

    return app