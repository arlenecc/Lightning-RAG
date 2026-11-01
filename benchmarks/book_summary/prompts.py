#!/usr/bin/env python3
"""Shared prompt definitions for the book-chapter summarization benchmark."""

SYSTEM_PROMPT = (
    "你是一名严谨的中文图书编辑，擅长把长篇书稿压缩成高信息密度的摘要。"
    "你只依据给定原文写作，不引入原文之外的知识。"
)

USER_TEMPLATE = """请阅读下面的电子书章节，完成一份供编辑审阅的中文摘要。

【章节标题】{title}

【章节正文】
{body}

【写作要求】
1. 严格依据原文，不得补充原文之外的人物、年代、事件或观点。
2. 按原文脉络组织，不要按小节逐段复述。
3. 保留关键人物、概念、事件、年代、数字与因果关系；专名保持原文写法。
4. 使用 Markdown，依次输出以下四部分：
## 一句话主旨
不超过60字。
## 核心要点
4–6条，每条1–2句。
## 关键词
3–5个，用顿号分隔。
## 本章结论
1–2句；若原文以叙述为主，写“本章以叙述为主”。
5. 总长度400–700字。不要输出“以下是摘要”等无关说明。
"""

GEN_CONFIG = {
    "max_tokens": 1000,
    "temperature": 0,
    "top_p": None,
    "enable_thinking": False,
    "prefill_step_size": 2048,
}
