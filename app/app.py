"""
한밤의 꿈해몽 상담가 - RAG 챗봇
OpenAI API 사용 버전 (Streamlit Cloud 배포용)
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
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

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
            try:
                # CPU 강제 사용, 캐시 디렉토리 지정
                import os
                os.environ['TORCH_HOME'] = './models'
                self.embedder = SentenceTransformer(
                    self.config['embedding_model'], 
                    device="cpu",
                    cache_folder="./models"
                )
                self.embedder.max_seq_length = 512
            except Exception as e:
                st.error(f"임베딩 모델 로드 실패: {e}")
                raise
    
    def _load_llm(self):
        """OpenAI API 클라이언트 설정"""
        with st.spinner("🤖 OpenAI API 클라이언트 설정..."):
            # Streamlit secrets에서 API 키 가져오기
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except Exception:
                st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
                st.info("Streamlit Cloud의 Secrets에 OPENAI_API_KEY를 추가해주세요.")
                raise ValueError("OpenAI API 키가 필요합니다.")
            
            self.client = OpenAI(api_key=api_key)
            st.success("✅ OpenAI API 클라이언트 설정 완료")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        유사한 청크 검색 (품질 개선)
        
        Args:
            query: 검색 쿼리
            k: 반환할 청크 수
            
        Returns:
            관련 청크 리스트 (품질 점수 포함)
        """
        # 검색 쿼리 확장 (프로이트 특화)
        expanded_queries = [
            f"dream interpretation symbol {query}",  # 영어 원문 검색
            f"꿈 해석 상징 의미 {query}",  # 한국어 검색
            f"무의식 욕망 갈등 {query}",  # 정신분석 개념 검색
            query  # 원본 쿼리
        ]
        
        all_results = []
        
        # 다중 쿼리로 검색하여 더 풍부한 결과 확보
        for expanded_query in expanded_queries:
            query_embedding = self.embedder.encode(
                expanded_query,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # FAISS 검색
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k
            )
            
            # 결과 수집
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1:
                    chunk = self.chunks[idx].copy()
                    chunk['score'] = float(1 / (1 + dist))
                    chunk['search_query'] = expanded_query
                    all_results.append(chunk)
        
        # 중복 제거 및 점수 기준 정렬
        unique_results = {}
        for result in all_results:
            chunk_id = result['metadata']['chunk_id']
            if chunk_id not in unique_results or result['score'] > unique_results[chunk_id]['score']:
                unique_results[chunk_id] = result
        
        # 상위 k개 반환 (최소 유사도 임계값 적용)
        final_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
        
        # 품질 필터링: 유사도가 너무 낮은 것 제거
        filtered_results = [r for r in final_results if r['score'] > 0.5]
        
        return filtered_results[:k]
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        OpenAI API를 사용해 응답 생성
        
        Args:
            query: 사용자 질문
            context_chunks: 검색된 컨텍스트
            
        Returns:
            생성된 응답
        """
        # 프로이트 관련 컨텍스트만 구성
        freud_context = []
        
        for chunk in context_chunks:
            # 모든 자료를 프로이트 관점에서 활용
            freud_context.append(chunk['text'])
        
        # 프롬프트 구성 (출처 기반 응답 강조)
        system_prompt = """당신은 지그문트 프로이트 박사입니다. 『꿈의 해석』의 저자이며, 제공된 문헌 자료를 바탕으로 꿈을 해석합니다.

**중요 지침:**
1. 반드시 제공된 문헌 내용을 우선적으로 활용하세요
2. 문헌에 없는 내용은 추측하지 말고, 명확히 구분하여 표시하세요  
3. 출처와 추론을 명확히 분리하여 제시하세요

**답변 형식 (반드시 준수):**

📘 **문헌 기반 분석**
- 제공된 『꿈의 해석』 원문을 직접 인용하며 해석
- 원문에서 언급된 상징과 이론을 적용
- 문헌 속 사례와 연결하여 설명
- 인용할 때는 "원문에 따르면..." 또는 "『꿈의 해석』에서..."로 시작

🎭 **프로이트의 추론적 해석**  
- 문헌 내용을 바탕으로 한 논리적 확장
- 개인적 통찰과 철학적 사색
- "이는 아마도..." 또는 "나의 경험으로는..." 등으로 구분
- 문헌 기반이 아닌 추론임을 명시

**문체 유지:**
- "꿈은 무의식으로 가는 왕도이다"
- "억압된 것은 변장하여 돌아온다"
- 19세기 말 빈 정신분석가의 어조로 대화

**검색 실패 시:** 관련 문헌이 부족하면 "제공된 자료만으로는 불충분하나, 일반적 정신분석 원리로는..." 라고 명시하세요."""

        # 컨텍스트 품질 평가
        context_quality = "충분" if len(freud_context) >= 3 else "부족"
        
        user_prompt = f"""**분석 대상 꿈:** {query}

**『꿈의 해석』 참고 문헌 ({len(freud_context)}개 구절 검색됨):**

{chr(10).join([f"구절 {i+1}: {text}" for i, text in enumerate(freud_context[:8])]) if freud_context else "⚠️ 관련 문헌이 검색되지 않았습니다."}

**지시사항:**
- 위 문헌 구절들을 우선적으로 활용하여 해석하세요
- 문헌 인용과 개인적 추론을 명확히 구분하세요
- 관련 문헌이 {context_quality}하므로 그에 맞게 해석의 범위를 조정하세요
- 반드시 📘 문헌 기반 분석과 🎭 추론적 해석을 분리하여 제시하세요"""
        
        # OpenAI API 호출
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"❌ OpenAI API 호출 실패: {e}")
            return "죄송합니다. 현재 꿈 해석 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."


def init_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_bot' not in st.session_state:
        index_dir = Path("index")
        if index_dir.exists():
            try:
                st.session_state.rag_bot = DreamRAGBot(index_dir)
            except ValueError as e:
                st.error(f"❌ 초기화 실패: {e}")
                st.stop()
        else:
            st.error("❌ 인덱스를 찾을 수 없습니다. 먼저 `python scripts/build_index.py`를 실행하세요.")
            st.stop()


def main():
    """메인 앱 함수"""
    st.set_page_config(
        page_title="프로이트 박사의 꿈해몽 상담소 🧠",
        page_icon="🎭",
        layout="wide"
    )
    
    # 헤더
    st.title("🧠 프로이트 박사의 꿈해몽 상담소")
    st.markdown("*'꿈은 무의식으로 가는 왕도이다'* - 지그문트 프로이트")
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 🎭 프로이트 박사 소개")
        st.markdown("""
        **지그문트 프로이트 (1856-1939)**
        - 정신분석학의 창시자
        - 무의식 이론의 선구자
        - 꿈 해석의 대가
        
        *"꿈은 욕망의 충족이다"*
        """)
        
        st.markdown("---")
        
        st.markdown("### 💭 상담 방법")
        st.markdown("""
        1. 꿈의 내용을 자세히 기술하세요
        2. 프로이트 박사가 상징을 분석합니다
        3. 무의식의 메시지를 발견하세요
        """)
        
        st.markdown("---")
        
        # 모델 정보 표시
        st.info("🧠 프로이트 박사 (OpenAI gpt-4.1 기반)")
        
        st.markdown("---")
        
        # 경고 메시지
        st.warning("""
        ⚠️ **이용 안내**
        
        이 서비스는 **오락 목적**으로만 사용해주세요.
        
        프로이트의 꿈 해석 이론은 20세기 초 이론으로, 현대 과학에서는 검증되지 않은 부분이 많습니다.
        
        실제 심리적 고민이 있으시면 전문 상담사와 상담하시기 바랍니다.
        """)
        
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
    if prompt := st.chat_input("프로이트 박사에게 꿈을 이야기해보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": "🧑"
        })
        
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # 프로이트 박사 응답 생성
        with st.chat_message("assistant", avatar="🎭"):
            with st.spinner("프로이트 박사가 꿈을 분석하는 중..."):
                start_time = time.time()
                
                # RAG 검색 (품질 개선된 다중 쿼리 검색)
                relevant_chunks = st.session_state.rag_bot.search_similar_chunks(prompt, k=15)
                
                # 응답 생성
                response = st.session_state.rag_bot.generate_response(prompt, relevant_chunks)
                
                elapsed_time = time.time() - start_time
                
                # 응답 표시
                st.markdown(response)
                
                # 상세 분석 정보 (출처 투명성 확보)
                with st.expander(f"📚 검색된 『꿈의 해석』 원문 ({len(relevant_chunks)}개 구절, {elapsed_time:.1f}초)"):
                    if relevant_chunks:
                        st.markdown("**실제 활용된 문헌 구절들:**")
                        for i, chunk in enumerate(relevant_chunks[:5], 1):
                            score = chunk['score']
                            text_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                            
                            # 점수에 따른 품질 표시
                            quality = "🟢 높음" if score > 0.7 else "🟡 보통" if score > 0.5 else "🔴 낮음"
                            
                            st.markdown(f"**구절 {i}** (관련도: {score:.3f} - {quality})")
                            st.markdown(f"```{text_preview}```")
                            st.markdown("---")
                    else:
                        st.warning("⚠️ 관련 문헌이 검색되지 않아 일반적 정신분석 원리로 해석했습니다.")
                        
                    # 검색 품질 요약
                    if relevant_chunks:
                        avg_score = sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks)
                        high_quality = sum(1 for chunk in relevant_chunks if chunk['score'] > 0.7)
                        st.info(f"📊 검색 품질 요약: 평균 관련도 {avg_score:.3f}, 고품질 구절 {high_quality}개")
                
                # 메모리 정리
                gc.collect()
        
        # 프로이트 박사 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": "🎭"
        })


if __name__ == "__main__":
    main() 