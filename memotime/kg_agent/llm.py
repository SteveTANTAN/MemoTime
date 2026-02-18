# =============================
# file: kg_agent/llm.py
# =============================
import os
import time
from typing import List, Dict, Any

# DEFAULT_OPENAI_API_KEY="XXX"
DEFAULT_DASHSCOPE_API_KEY="XXX"
DEFAULT_GOOGLE_API_KEY="XXX"
DEFAULT_DEEPSEEK_API_KEY="XXX"
DEFAULT_LAMBDA_API_KEY="XXX"
DEFAULT_LOCAL_OPENAI_BASE="http://localhost:6666/v1"
DEFAULT_LOCAL_OPENAI_KEY="EMPTY"
Deep_infra_API_KEY= "XXX"
new_llama_api_key = "XXX"

class LLM:
    @staticmethod
    def call(system: str, prompt: str, model: str = None, temperature: float = 0.4) -> str:
        # if model is not specified, read from configuration
        if model is None:
            try:
                from ..config import TPKGConfig
                model = TPKGConfig.DEFAULT_LLM_MODEL
            except:
                # if cannot import configuration, use default value
                model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # lazy import, avoid unnecessary dependencies
        print(f"Model {model} response: ")
        try:
            import google.generativeai as genai
        except Exception:
            genai = None
        try:
            from openai import OpenAI
        except Exception:
            OpenAI = None

        # unified message format (OpenAI compatible)
        def _mk_messages(sys_msg: str, user_msg: str) -> List[Dict[str, Any]]:
            return [
                {"role": "system", "content": sys_msg or "You are an AI assistant that helps people find information."},
                {"role": "user", "content": user_msg},
            ]

        # OpenAI compatible chat call, with retry
        def _chat_with_openai_client(client, model_name: str, messages: List[Dict[str, Any]],
                                     temperature: float = 0.4, max_tokens: int = 2048,
                                     extra_body: Dict[str, Any] = None, retries: int = 3, backoff: float = 2.0) -> str:
            last_err = None
            for i in range(retries):
                try:
                    kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    resp = client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content
                except Exception as e:
                    last_err = e
                    time.sleep(backoff if i < retries - 1 else 0)
            raise last_err

        # routing branch
        result = ""
        sys_msg = system or "You are an AI assistant that helps people find information."
        user_msg = prompt

        # --- Google (Gemini) ---
        if "google" in model:
            if genai is None:
                raise RuntimeError("google.generativeai not installed, cannot call Google model.")
            api_key = os.getenv("GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY)
            if not api_key:
                raise RuntimeError("missing GOOGLE_API_KEY environment variable.")
            genai.configure(api_key=api_key)
            gmodel = "gemini-1.5-flash"
            # reuse your original writing: put system as start history item
            chat = genai.GenerativeModel(gmodel).start_chat(
                history=[{"role": "user", "parts": sys_msg}]
            )
            tries, last_err = 3, None
            for i in range(tries):
                try:
                    resp = chat.send_message(user_msg)
                    return resp.text
                except Exception as e:
                    last_err = e
                    time.sleep(2 if i < tries - 1 else 0)
            raise last_err

        # --- OpenAI series (gpt*) ---
        elif "gpt" in model:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
            api_key = os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
            if not api_key:
                raise RuntimeError("missing OPENAI_API_KEY environment variable.")
            # compatible with your original alias
            if model == "gpt3":
                resolved_model = "gpt-3.5-turbo"
            elif model == "gpt4":
                resolved_model = "gpt-4-turbo"
            else:
                resolved_model = "gpt-4o-mini"
            # resolved_model = "gpt-3.5-turbo" if model == "gpt4" else "gpt-4o-mini"
            client = OpenAI(api_key=api_key)
            print("resolved_model:", resolved_model)
            messages = _mk_messages(sys_msg, user_msg)
            return _chat_with_openai_client(client, resolved_model, messages, temperature)

        # --- Llama 70B (Lambda Labs compatible) ---
        elif "llama70b" in model:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
            api_key =  Deep_infra_API_KEY
            if not api_key:
                raise RuntimeError("missing LAMBDA_API_KEY environment variable.")
            # base_url = "https://api.deepinfra.com/v1/openai"
            
            api_key = new_llama_api_key
            
            base_url = "https://api.groq.com/openai/v1"
            client = OpenAI(api_key=api_key, base_url=base_url)
            resolved_model = "llama-3.3-70b-versatile"
            messages = _mk_messages(sys_msg, user_msg)
            return _chat_with_openai_client(client, resolved_model, messages, temperature)
        # --- Llama 70B (Lambda Labs compatible) ---
        elif "llama8b" in model:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
            api_key =  Deep_infra_API_KEY
            if not api_key:
                raise RuntimeError("missing LAMBDA_API_KEY environment variable.")
            api_key = new_llama_api_key
            
            base_url = "https://api.groq.com/openai/v1"
            client = OpenAI(api_key=api_key, base_url=base_url)
            resolved_model = "llama-3.1-8b-instant"
            messages = _mk_messages(sys_msg, user_msg)
            return _chat_with_openai_client(client, resolved_model, messages, temperature)

        # --- local/self-built Llama compatible (llama*) ---
        elif "llama" in model:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
            base_url = os.getenv("LOCAL_OPENAI_BASE", DEFAULT_LOCAL_OPENAI_BASE)
            api_key = os.getenv("LOCAL_OPENAI_KEY", "EMPTY")
            client = OpenAI(api_key=api_key, base_url=base_url)
            resolved_model = "Meta-Llama-3.1-8B-Instruct"
            messages = _mk_messages(sys_msg, user_msg)
            return _chat_with_openai_client(client, resolved_model, messages, temperature)

        # --- DeepSeek (deep*) compatible ---
        elif "deep" in model:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
            api_key = os.getenv("DEEPSEEK_API_KEY", DEFAULT_DEEPSEEK_API_KEY)
            if not api_key:
                raise RuntimeError("missing DEEPSEEK_API_KEY environment variable.")
            base_url = "https://api.deepseek.com"
            client = OpenAI(api_key=api_key, base_url=base_url)
            resolved_model = "deepseek-chat"
            messages = _mk_messages(sys_msg, user_msg)
            return _chat_with_openai_client(client, resolved_model, messages, temperature)

        # --- Qwen (DashScope compatible OpenAI) ---
        elif "qwen" in model:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
            api_key = os.getenv("DASHSCOPE_API_KEY", DEFAULT_DASHSCOPE_API_KEY) or os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("missing DASHSCOPE_API_KEY or QWEN_API_KEY environment variable.")
            base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
            client = OpenAI(api_key=api_key, base_url=base_url)
            if "qwen3-80b" in model:
                resolved_model = "qwen3-next-80b-a3b-instruct"
            else:
                resolved_model = model
            resolved_model = resolved_model
            messages = _mk_messages(sys_msg, user_msg)
            # compatible with your original extra_body
            return _chat_with_openai_client(
                client, resolved_model, messages,
                temperature=temperature, max_tokens=2048,
                extra_body={"enable_thinking": False}
            )

        # --- fallback: local OpenAI compatible gateway ---
        else:
            if OpenAI is None:
                raise RuntimeError("openai SDK not installed, cannot call OpenAI compatible interface.")
                base_url = os.getenv("LOCAL_OPENAI_BASE", DEFAULT_LOCAL_OPENAI_BASE)
            api_key = os.getenv("LOCAL_OPENAI_KEY", DEFAULT_LOCAL_OPENAI_KEY)
            client = OpenAI(api_key=api_key, base_url=base_url)
            resolved_model = "Meta-Llama-3-8B-Instruct"
            messages = _mk_messages(sys_msg, user_msg)
            return _chat_with_openai_client(client, resolved_model, messages, temperature)
