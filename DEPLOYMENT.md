"""
MLX 모델 API 서버
로컬에서 실행하여 클라우드 앱에 LLM 기능 제공
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from mlx_lm import load, generate
from contextlib import asynccontextmanager
import os

# 전역 변수로 모델 저장
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드"""
    global model, tokenizer
    
    print("🤖 MLX 모델 로딩...")
    model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    model, tokenizer = load(model_name)
    print("✅ 모델 로드 완료")
    
    yield
    
    # 정리 작업
    print("👋 서버 종료")


app = FastAPI(
    title="Dream Bot MLX API",
    description="꿈해몽 상담가 LLM API",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    """생성 요청 모델"""
    prompt: str
    max_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    """생성 응답 모델"""
    text: str
    tokens_generated: int


@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "healthy", "model": "Qwen2.5-7B-Instruct-4bit"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """텍스트 생성 엔드포인트"""
    try:
        # MLX 생성
        response = generate(
            model,
            tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temp=request.temperature,
            top_p=request.top_p,
        )
        
        # 토큰 수 계산
        tokens = tokenizer.encode(response)
        
        return GenerateResponse(
            text=response,
            tokens_generated=len(tokens)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_completion(messages: List[Dict[str, str]], max_tokens: int = 300):
    """ChatGPT 스타일 엔드포인트"""
    try:
        # 메시지를 프롬프트로 변환
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 생성
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.7,
            top_p=0.9,
        )
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 환경 변수로 포트 설정 가능
    port = int(os.getenv("MLX_SERVER_PORT", "8000"))
    
    print(f"🚀 MLX API 서버 시작: http://localhost:{port}")
    print("📝 API 문서: http://localhost:{port}/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 