"""
Enterprise RAG SaaS — LLM 推理服务 (V3.0)
===========================================
提供统一的大模型推理接口，实现多 Tier 路由 + 云端 API 弹性兜底 + SSE 流式生成。

======================================================================
架构设计理念
======================================================================

为什么需要 Tier 路由？
  问题: 不同租户套餐对延迟/质量/成本的要求完全不同
  方案: 按 tier 字段动态选择模型
    free       → Ollama qwen2:7b   (本地CPU, 免费, 首Token ~3s)
    basic      → Ollama qwen2.5:14b (本地CPU, 中等)
    pro        → Ollama qwen2.5:72b (本地CPU, 慢但准)
    enterprise → 同上 + 更大max_tokens
  所有 tier 的 base_url 默认指向 Ollama，实际部署时可在 .env 中覆盖

为什么需要云端兜底 (fallback)？
  问题: Ollama 7B 模型并发能力有限，队列深度 > 20 时响应 > 30s
  方案: 检测队列溢出 → 自动切 DeepSeek API → 冷却期 30s 后切回
  这样: 平时免费，高峰自动花钱保体验

为什么有冷却期 (cooldown)？
  问题: 队列在 19→20→19 间震荡会导致频繁切换（抖动）
  方案: 切到云端后至少保持 30s，切回本地后至少保持 30s
  效果: 将切换频率从 N次/秒 降低到 1次/30s

======================================================================
V1 兼容层
======================================================================
LLMService 在默认模式下直接使用 settings.LLM_PROVIDER/LLM_MODEL/LLM_BASE_URL
这意味着:
  - 不启用 Tier 路由时: 用 .env 中的 LLM_PROVIDER=deepseek 直接推理
  - 启用 Tier 路由时: 按 tier 选择模型 → 默认 Ollama → 兜底 DeepSeek
"""

import time
from typing import Generator, Optional
from openai import OpenAI
from src.config.settings import Settings


class ModelTierRouter:
    """
    多 Tier 模型路由器

    设计决策: 为什么 Tier 配置写死在代码里而不是 .env？
      → 这些是产品策略（套餐定价的一部分），不是运维配置
      → 修改套餐策略应该走代码审查，而非运维随意改环境变量
      → 租户的自定义模型偏好存在 KnowledgeBase.preferred_model 字段
    """

    # Tier → 模型配置映射
    TIER_CONFIG = {
        "free":       {"model": "qwen2:7b",    "base_url": "http://localhost:11434/v1", "max_tokens": 1024},
        "basic":      {"model": "qwen2.5:14b", "base_url": "http://localhost:11434/v1", "max_tokens": 2048},
        "pro":        {"model": "qwen2.5:72b", "base_url": "http://localhost:11434/v1", "max_tokens": 4096},
        "enterprise": {"model": "qwen2.5:72b", "base_url": "http://localhost:11434/v1", "max_tokens": 8192},
    }

    # 云端兜底配置
    FALLBACK_CONFIG = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "cost_yuan_per_1k_tokens": 0.001,  # DeepSeek 极低价格，兜底成本可控
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._clients = {}          # base_url → OpenAI client 缓存
        self._fallback_client = None
        self._request_queue = []    # 请求队列（模拟并发深度）
        self.max_queue_depth = 20   # 队列超过此值触发云端切换

        # 冷却期防抖机制
        self._fallback_since: Optional[float] = None
        self._fallback_cooldown = 30.0  # 云端 → 本地：至少保持云端 30s
        self._local_since: Optional[float] = None
        self._local_cooldown = 30.0     # 本地 → 云端：至少保持本地 30s

    def get_client(self, tier: str = "free", kb_preferred_model: str = None) -> OpenAI:
        """
        获取 OpenAI 兼容客户端
        为什么用 OpenAI 兼容协议？
          → Ollama/vLLM/DeepSeek 都提供 /v1/chat/completions 端点
          → 一套代码，无需 if-else 判断 provider
        """
        config = self.TIER_CONFIG.get(tier, self.TIER_CONFIG["free"]).copy()
        if kb_preferred_model:
            config["model"] = kb_preferred_model  # 租户自定义模型覆盖套餐默认

        base_url = config["base_url"]
        if base_url not in self._clients:
            self._clients[base_url] = OpenAI(api_key="ollama", base_url=base_url, timeout=60.0)
        return self._clients[base_url]

    def get_model_config(self, tier: str) -> dict:
        return self.TIER_CONFIG.get(tier, self.TIER_CONFIG["free"])

    def should_use_fallback(self) -> bool:
        """
        判断是否应使用云端 API 兜底
        逻辑:
          1. 队列深度 >= 20 → 切换到云端（如果不在本地冷却期）
          2. 队列恢复正常 → 在云端保持 30s 后再切回本地
        防抖效果: 将切换频率从高频抖动降低到 ~1次/30s
        """
        now = time.time()
        is_overloaded = len(self._request_queue) >= self.max_queue_depth

        if is_overloaded:
            if self._fallback_since is None:
                if self._local_since and now - self._local_since < self._local_cooldown:
                    return False  # 本地冷却中，暂不切换
                self._fallback_since = now
                self._local_since = None
            return True
        else:
            if self._fallback_since and now - self._fallback_since >= self._fallback_cooldown:
                self._fallback_since = None
                self._local_since = now
                return False
            return self._fallback_since is not None

    def get_fallback_client(self) -> Optional[OpenAI]:
        """获取云兜底客户端（延迟初始化，避免未配置 API Key 时启动报错）"""
        import os
        if self._fallback_client:
            return self._fallback_client
        api_key = os.getenv(self.FALLBACK_CONFIG["api_key_env"])
        if not api_key:
            return None
        self._fallback_client = OpenAI(api_key=api_key, base_url=self.FALLBACK_CONFIG["base_url"], timeout=30.0)
        return self._fallback_client


class LLMService:
    """
    LLM 推理服务 — 统一入口

    两种使用模式:
      Mode A (V1 兼容): 直接用 self.client，按 settings.LLM_PROVIDER 推理
      Mode B (V2 Tier):   按 JWT tier 字段路由 + 队列溢出云端兜底
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.router = ModelTierRouter(settings)

        # V1 兼容: 默认客户端（直接读 .env 配置）
        self.client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
        print(f"[LLMService V2] Tier router ready: {list(self.router.TIER_CONFIG.keys())}")

    def generate(self, prompt: str, tier: str = "free", stream: bool = False,
                 kb_preferred_model: str = None) -> str:
        """
        生成回答（同步模式）

        参数:
          prompt:              完整的 RAG Prompt（含上下文）
          tier:                租户等级 (free/basic/pro/enterprise)
          stream:              是否流式（内部用，对外暴露 generate_stream）
          kb_preferred_model:  知识库级别的自定义模型（覆盖 tier 默认值）

        返回: LLM 生成的完整文本
        """
        config = self.router.get_model_config(tier)
        if kb_preferred_model:
            config = config.copy()
            config["model"] = kb_preferred_model

        model = config["model"]
        client = self.client  # 默认客户端

        # 云端兜底检查
        if self.router.should_use_fallback():
            fallback = self.router.get_fallback_client()
            if fallback:
                model = self.router.FALLBACK_CONFIG["model"]
                client = fallback
                print(f"[LLM] Queue overflow → fallback to {model}")

        # 流式内部实现（收集所有 chunk 后拼接返回）
        if stream:
            chunks = []
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 1024), stream=True,
            )
            for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            return "".join(chunks)

        # 标准同步调用
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 1024),
        )
        return resp.choices[0].message.content

    def generate_stream(self, prompt: str, tier: str = "free") -> Generator[str, None, None]:
        """
        流式生成（用于 SSE 端点）
        返回 Generator，每次 yield 一个 token
        """
        config = self.router.get_model_config(tier)
        client = self.client
        model = config["model"]

        if self.router.should_use_fallback():
            fallback = self.router.get_fallback_client()
            if fallback:
                model = self.router.FALLBACK_CONFIG["model"]
                client = fallback

        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 1024), stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
