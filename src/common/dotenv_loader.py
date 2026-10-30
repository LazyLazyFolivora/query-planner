# -*- coding: utf-8 -*-
"""极简 .env 加载器（不依赖 python-dotenv）。

为什么需要它
------------
业务代码只从 os.environ 读配置，但环境变量有两个坏处：
    1. 每个新终端都要重设一遍，切换供应商/模型时很烦；
    2. 容易把 key 敲进命令行，既进历史记录又会被别人看到。
把配置放进项目根目录的 .env 是常规做法，.gitignore 已经忽略了它。

解析规则自己实现，不走 python-dotenv
------------------------------------
"装了用官方库、没装用降级解析"会让同一份 .env 在不同机器上解析出不同结果
（官方库会剥内联注释，空值处理也和降级版不同）。配置相关的环境漂移
比多一个依赖更难排查，所以固定用下面这一套规则。

行为约定
--------
    1. 只在"该变量尚未存在于环境"时写入，真实环境变量优先级更高
       （这样临时 QP_LLM_MODEL=xxx 就能覆盖 .env，方便做消融实验）。
    2. 值写成空白的键视为未配置，不写进环境——"没配 key"和"配了空 key"
       对 is_available() 是同一件事，没必要区分。
    3. 文件不存在时静默返回空 dict：没配 .env 只意味着
       "用系统环境变量或默认值"，不是错误。
"""

from __future__ import annotations

import os
from pathlib import Path


# src/common/dotenv_loader.py -> parents[2] 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """把 .env 里的键值对载入 os.environ，返回本次实际生效的键值。

    path 缺省指向项目根目录 .env。
    override=False 时已有环境变量优先；override=True 则强制用文件里的值。
    """
    env_path = Path(path) if path is not None else ENV_FILE
    if not env_path.is_file():
        return {}

    applied: dict[str, str] = {}
    for key, value in _parse(env_path).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def _parse(env_path: Path) -> dict[str, str]:
    """逐行解析 KEY=VALUE。

    有意收窄语法，不追求兼容所有 shell 用法：
        - 空行与 # 开头的整行注释忽略；export 前缀去掉
        - # 只有在行首或紧跟空白时才算注释，因此值里的普通 # 不会被截断
        - 成对的单/双引号包裹时去掉引号
        - 值为空白的键直接丢弃
    """
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()

        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue

        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        value = value.strip()
        if value:
            values[key] = value
    return values


def _strip_inline_comment(value: str) -> str:
    """去掉 ` 1  # note` 的注释尾巴，但保留不含空白的 #（可能是值的一部分）。"""
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value
