"""
한밤의 꿈해몽 상담가 - RAG 챗봇
M2 MacBook Air 8GB RAM 최적화 버전
"""

import os
import gc
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
import psutil
import streamlit as st
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate
import mlx.core as mx


class DreamRAGBot:
    """꿈 해석 RAG 챗봇 클래스"""
    
    def __init__(self, index_dir: Path):
        """
        초기화
        
        Args:
            index_dir: 인덱스 파일 디렉토리
        """
        self.index_dir = index_dir
        
        # 컴포넌트 초기화
        self._load_index()
        self._load_embedder()
        self._load_llm()
        
    def _load_index(self):
        """FAISS 인덱스 및 메타데이터 로드"""
        with st.spinner("🔍 검색 인덱스 로딩..."):
            # FAISS 인덱스
            self.index = faiss.read_index(str(self.index_dir / "dream_index.faiss"))
            
            # 청크 데이터
            with open(self.index_dir / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            # 설정 정보
            with open(self.index_dir / "config.json", "r") as f:
                self.config = json.load(f)
            
            st.success(f"✅ {self.config['total_chunks']}개 문서 청크 로드 완료")
    
    def _load_embedder(self):
        """임베딩 모델 로드"""
        with st.spinner("🧠 임베딩 모델 로딩..."):
            self.embedder = SentenceTransformer(self.config['embedding_model'])
            self.embedder.max_seq_length = 512
    
    def _load_llm(self):
        """MLX 기반 Qwen 모델 로드"""
        with st.spinner("🤖 Qwen 모델 로딩 (최초 실행 시 다운로드)..."):
            # Qwen 2.5-7B Int4 모델 (메모리 효율적)
            model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"
            
            try:
                self.model, self.tokenizer = load(model_name)
                st.success("✅ LLM 모델 로드 완료")
            except Exception as e:
                st.error(f"❌ 모델 로드 실패: {e}")
                st.info("모델을 다운로드하려면: `mlx-lm download mlx-community/Qwen2.5-7B-Instruct-4bit`")
                raise
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        유사한 청크 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 청크 수
            
        Returns:
            관련 청크 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # FAISS 검색
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # 결과 추출
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # 유효한 인덱스
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(1 / (1 + dist))  # 거리를 유사도로 변환
                results.append(chunk)
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        LLM을 사용해 응답 생성
        
        Args:
            query: 사용자 질문
            context_chunks: 검색된 컨텍스트
            
        Returns:
            생성된 응답
        """
        # 컨텍스트 구성
        freud_context = []
        who_context = []
        
        for chunk in context_chunks:
            if "freud" in chunk['metadata']['source'].lower():
                freud_context.append(chunk['text'])
            else:
                who_context.append(chunk['text'])
        
        # 프롬프트 구성
        system_prompt = """당신은 꿈 해석 전문가입니다. 
프로이트의 정신분석학적 관점과 WHO의 현대 수면과학을 통합하여 답변합니다.
답변은 반드시 다음 3단계 형식을 따라주세요:

① <고전적 해석>: 프로이트 이론 기반 (약간 엄숙하고 학술적 톤)
② <현대 과학>: WHO 수면 가이드 기반 (간결하고 실용적)  
③ <통합 조언>: 두 관점을 결합한 실천적 조언 (친근하고 격려하는 톤)

각 단계는 2-3문장으로 작성하세요."""

        user_prompt = f"""꿈 내용: {query}

프로이트 자료:
{' '.join(freud_context[:2]) if freud_context else '관련 자료 없음'}

WHO 수면 자료:
{' '.join(who_context[:2]) if who_context else '관련 자료 없음'}

위 자료를 참고하여 3단계 형식으로 답변해주세요."""
        
        # 토큰 제한 (M2 8GB 최적화)
        max_tokens = 300
        
        # 생성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # MLX 생성
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # 창의성과 일관성의 균형
            top_p=0.9,
        )
        
        return response


def init_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_bot' not in st.session_state:
        index_dir = Path("index")
        if index_dir.exists():
            st.session_state.rag_bot = DreamRAGBot(index_dir)
        else:
            st.error("❌ 인덱스를 찾을 수 없습니다. 먼저 `python scripts/build_index.py`를 실행하세요.")
            st.stop()


def main():
    """메인 앱 함수"""
    st.set_page_config(
        page_title="한밤의 꿈해몽 상담가 🌙",
        page_icon="🔮",
        layout="wide"
    )
    
    # 헤더
    st.title("🌙 한밤의 꿈해몽 상담가")
    st.markdown("프로이트의 정신분석과 WHO의 수면과학이 만나는 곳")
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 💡 사용법")
        st.markdown("""
        1. 꿈의 내용을 자세히 입력하세요
        2. 3초 이내에 세 가지 관점의 해석을 받아보세요
        3. 더 나은 수면을 위한 조언도 함께!
        """)
        
        st.markdown("---")
        
        # 메모리 모니터링
        memory = psutil.virtual_memory()
        st.metric("메모리 사용률", f"{memory.percent:.1f}%")
        
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            gc.collect()
            st.rerun()
    
    # 세션 초기화
    init_session_state()
    
    # 대화 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("오늘 밤 꾼 꿈을 들려주세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": "🧑"
        })
        
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # 봇 응답 생성
        with st.chat_message("assistant", avatar="🔮"):
            with st.spinner("꿈을 해석하는 중..."):
                start_time = time.time()
                
                # RAG 검색
                relevant_chunks = st.session_state.rag_bot.search_similar_chunks(prompt, k=6)
                
                # 응답 생성
                response = st.session_state.rag_bot.generate_response(prompt, relevant_chunks)
                
                elapsed_time = time.time() - start_time
                
                # 응답 표시
                st.markdown(response)
                
                # 성능 정보 (축소 가능)
                with st.expander(f"🔍 분석 정보 ({elapsed_time:.1f}초)"):
                    st.markdown("**참고 자료:**")
                    for i, chunk in enumerate(relevant_chunks[:3], 1):
                        source = chunk['metadata']['source']
                        st.markdown(f"{i}. {source} - 유사도: {chunk['score']:.2f}")
                
                # 메모리 정리
                gc.collect()
        
        # 봇 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": "🔮"
        })


if __name__ == "__main__":
    main() 