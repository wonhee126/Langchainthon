"""
Reddit 상담사 챗봇 - RAG 시스템
OpenAI API 사용 버전 (Streamlit Cloud 배포용)
TIFU와 AITA 데이터를 활용한 조언 제공 서비스
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
from sentence_transformers import SentenceTransformer  # 크로스 플랫폼 지원
from openai import OpenAI

class RedditAdviseBot:
    """Reddit 상담사 RAG 챗봇 클래스"""
    
    def __init__(self, index_dir: Path):
        """
        초기화
        
        Args:
            index_dir: 인덱스 파일 디렉토리
        """
        self.index_dir = index_dir
        
        # 컴포넌트 초기화
        self._load_index()
        self._load_embedder()  # 크로스 플랫폼 임베더 로딩
        self._load_llm()
        
    def _load_index(self):
        """FAISS 인덱스 및 메타데이터 로드 (없으면 패스)"""
        index_file = self.index_dir / "reddit_index.faiss"
        chunks_file = self.index_dir / "chunks.pkl"
        config_file = self.index_dir / "config.json"
        
        if not index_file.exists():
            st.info("📁 인덱스 파일이 없습니다. RAG 검색 없이 작동합니다.")
            self.index = None
            self.chunks = []
            self.config = {'total_chunks': 0}
            return
        
        with st.spinner("🔍 검색 인덱스 로딩..."):
            try:
                # FAISS 인덱스
                self.index = faiss.read_index(str(index_file))
                
                # 청크 데이터
                with open(chunks_file, "rb") as f:
                    self.chunks = pickle.load(f)
                
                # 설정 정보
                with open(config_file, "r") as f:
                    self.config = json.load(f)
                
                st.success(f"✅ {self.config['total_chunks']}개 Reddit 포스트 청크 로드 완료")
            except Exception as e:
                st.warning(f"인덱스 로딩 실패: {e}")
                self.index = None
                self.chunks = []
                self.config = {'total_chunks': 0}
    
    def _load_embedder(self):
        """임베딩 모델 로드 (크로스 플랫폼 호환)"""
        if self.index is None:
            st.info("📁 인덱스가 없어서 임베더 로딩을 스킵합니다.")
            self.embedder = None
            return
            
        with st.spinner("🧠 임베딩 모델 로딩..."):
            try:
                import os
                import platform
                
                # 플랫폼별 최적화 (선택적 적용)
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    # Apple Silicon Mac 최적화
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                    os.environ['OMP_NUM_THREADS'] = '1'
                
                # 범용 설정
                os.environ['TORCH_HOME'] = './models'
                
                # CPU 사용으로 크로스 플랫폼 안정성 확보
                self.embedder = SentenceTransformer(
                    self.config['embedding_model'], 
                    device="cpu",  # 모든 플랫폼에서 안정적
                    cache_folder="./models"
                )
                self.embedder.max_seq_length = 512  # 적당한 길이
                st.success("✅ 임베딩 모델 로드 완료")
                
            except Exception as e:
                st.error(f"임베딩 모델 로드 실패: {e}")
                st.warning("🔄 검색 없이 기본 상담 모드로 전환")
                self.embedder = None
    
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
        유사한 경험담/상황 검색 (임시로 비활성화)
        
        Args:
            query: 검색 쿼리 (사용자의 상황/고민)
            k: 반환할 청크 수
            
        Returns:
            관련 경험담 리스트 (유사도 점수 포함)
        """
        # 인덱스나 임베더가 없으면 빈 결과 반환  
        if self.index is None or len(self.chunks) == 0 or not hasattr(self, 'embedder') or self.embedder is None:
            print(f"⚠️ 인덱스 또는 임베더가 없어서 검색 불가 - 쿼리: {query}")
            return []
        
        print(f"🔍 실제 검색 시작 - 쿼리: {query}")
        
        # 실제 검색 로직 활성화!
        expanded_queries = [
            f"비슷한 상황 경험 조언 {query}",  # 한국어 검색
            f"similar situation advice experience {query}",  # 영어 검색
            f"문제 해결 도움 {query}",  # 문제 해결 관련
            query  # 원본 쿼리
        ]
        
        all_results = []
        
        # 다중 쿼리로 검색하여 더 풍부한 결과 확보
        for expanded_query in expanded_queries:
            try:
                query_embedding = self.embedder.encode(
                    expanded_query,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=False  # 안정성을 위해 numpy 사용
                )
            except Exception as e:
                print(f"⚠️ 임베딩 생성 실패 ({expanded_query}): {e}")
                continue
            
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
            chunk_key = f"{result['metadata']['source']}_{result['metadata']['post_id']}_{chunk_id}"
            if chunk_key not in unique_results or result['score'] > unique_results[chunk_key]['score']:
                unique_results[chunk_key] = result
        
        # 상위 k개 반환 (최소 유사도 임계값 적용)
        final_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
        
        # 품질 필터링: 유사도가 너무 낮은 것 제거
        filtered_results = [r for r in final_results if r['score'] > 0.4]
        
        print(f"✅ 검색 완료: {len(filtered_results)}개 관련 경험담 발견")
        return filtered_results[:k]
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        OpenAI API를 사용해 상담 응답 생성
        
        Args:
            query: 사용자 질문/고민
            context_chunks: 검색된 유사 경험담
            
        Returns:
            생성된 상담 응답
        """
        # Reddit 경험담 컨텍스트 구성
        reddit_context = []
        
        for chunk in context_chunks:
            source = chunk['metadata']['source']
            context_info = f"[{source} 경험담] {chunk['text']}"
            reddit_context.append(context_info)
        
        # 프롬프트 구성 (경험 많은 상담사 페르소나)
        system_prompt = """당신은 경험이 풍부한 온라인 상담사입니다. Reddit의 TIFU(Today I F***ed Up) 커뮤니티의 수많은 경험담을 분석하여 조언을 제공합니다.

**역할과 전문성:**
- 다양한 인생 경험과 실수담을 분석한 상담 전문가
- 현실적이고 실용적인 조언 제공
- 공감적이면서도 객관적인 시각 유지
- 비슷한 상황을 겪은 사람들의 경험을 바탕으로 통찰 제공

**상담 스타일:**
- 따뜻하고 이해심 많은 톤
- 판단하지 않고 공감하는 자세
- 구체적이고 실행 가능한 조언
- 비슷한 경험담을 활용한 위로와 격려

**응답 구조 (반드시 준수):**

🤗 **공감과 이해**
- 사용자의 상황에 대한 공감과 이해 표현
- "힘든 상황이셨겠어요", "충분히 이해됩니다" 등의 표현 사용

📖 **비슷한 경험담 분석**
- 제공된 Reddit 경험담들을 바탕으로 유사한 상황 분석
- "비슷한 상황을 겪은 분들의 경험을 보면..." 형태로 시작
- 경험담에서 얻을 수 있는 교훈이나 패턴 설명

💡 **실용적 조언**
- 구체적이고 실행 가능한 단계별 조언
- 상황 개선을 위한 실질적인 방법 제시
- 예상되는 어려움과 대처 방안 포함

🌟 **격려와 희망**
- 상황이 나아질 수 있다는 희망적 메시지
- 사용자의 강점이나 긍정적 측면 강조
- 성장과 학습의 기회로 바라보는 관점 제시

**주의사항:**
- 전문적인 의료/법률 조언은 피하고 일반적인 상담에 집중
- 극단적인 상황에서는 전문가 상담 권유
- 개인의 가치관과 상황을 존중하는 조언
- 과도한 확신보다는 "~해보시는 것이 좋을 것 같아요" 형태의 제안

**금지사항:**
- 부정적이거나 비판적인 표현
- 성급한 결론이나 단정적 판단
- 개인 정보나 민감한 내용 요구
- 불법적이거나 해로운 조언"""

        context_text = "\n\n".join(reddit_context) if reddit_context else "관련 경험담을 찾지 못했지만, 일반적인 조언을 드리겠습니다."
        
        user_prompt = f"""사용자 상황: {query}

관련 Reddit 경험담들:
{context_text}

위의 경험담들을 참고하여 사용자에게 공감적이고 실용적인 조언을 제공해주세요."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")
            return "죄송합니다. 일시적인 오류로 응답을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."

    def chat(self, user_input: str) -> Tuple[str, List[Dict]]:
        """
        사용자 입력에 대한 챗봇 응답 생성
        
        Args:
            user_input: 사용자 입력 (고민/상황)
            
        Returns:
            Tuple[응답 텍스트, 참고한 경험담 목록]
        """
        with st.spinner("🔍 비슷한 경험담을 찾고 있어요..."):
            # 유사한 경험담 검색
            similar_chunks = self.search_similar_chunks(user_input, k=3)
            
        with st.spinner("💭 조언을 준비하고 있어요..."):
            # 응답 생성
            response = self.generate_response(user_input, similar_chunks)
            
        return response, similar_chunks


def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "bot" not in st.session_state:
        # 인덱스 디렉토리 확인 (임시로 관대하게 처리)
        index_dir = Path("index")
        if not index_dir.exists() or not (index_dir / "reddit_index.faiss").exists():
            st.warning("⚠️ 인덱스 파일이 없습니다. RAG 검색 없이 기본 상담 모드로 작동합니다.")
            st.info("완전한 기능을 위해서는 `python scripts/build_index.py`를 실행해주세요.")
            # 임시 더미 인덱스 디렉토리 생성
            index_dir.mkdir(exist_ok=True)
        
        try:
            # 봇 초기화 (인덱스 없어도 작동하도록)
            st.session_state.bot = RedditAdviseBot(index_dir)
        except Exception as e:
            st.error(f"봇 초기화 실패: {e}")
            st.info("인덱스가 없어도 기본 상담은 가능합니다. 계속 진행합니다.")
            st.session_state.bot = None


def main():
    """메인 애플리케이션"""
    st.set_page_config(
        page_title="Reddit 상담사 🤗",
        page_icon="🤗",
        layout="wide"
    )
    
    # 세션 상태 초기화
    init_session_state()
    
    # 헤더
    st.title("🤗 Reddit 상담사")
    st.markdown("""
    안녕하세요! 저는 Reddit 커뮤니티의 수많은 경험담을 학습한 AI 상담사입니다.  
    여러분의 고민이나 상황을 말씀해주시면, 비슷한 경험을 한 분들의 이야기를 바탕으로 조언을 드려요.
    """)
    
    # 사이드바 - 사용법 안내
    with st.sidebar:
        st.header("📖 사용법")
        st.markdown("""
        **어떤 상담을 받을 수 있나요?**
        - 일상생활 문제와 고민
        - 인간관계 갈등
        - 실수나 후회에 대한 조언
        - 도덕적 딜레마 상황
        - 의사결정 도움
        
        **예시 질문:**
        - "친구와 싸웠는데 어떻게 화해할까요?"
        - "실수로 상사에게 실례를 범했어요"
        - "연인과 헤어질지 고민이에요"
        - "가족과의 갈등 때문에 힘들어요"
        """)
        
        st.header("⚠️ 주의사항")
        st.markdown("""
        - 일반적인 조언만 제공합니다
        - 전문적인 의료/법률 상담은 전문가에게
        - 개인정보는 입력하지 마세요
        - 응급상황시 관련 기관에 연락하세요
        """)
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # 참고 경험담 표시 (assistant 메시지에만)
            if message["role"] == "assistant" and "references" in message:
                with st.expander("📚 참고한 경험담들", expanded=False):
                    for i, ref in enumerate(message["references"], 1):
                        source = ref['metadata']['source']
                        title = ref['metadata'].get('title', '제목 없음')
                        score = ref.get('score', 0)
                        
                        st.markdown(f"""
                        **{i}. [{source}] {title}**  
                        유사도: {score:.2f}  
                        {ref['text'][:200]}...
                        """)
    
    # 사용자 입력
    if prompt := st.chat_input("고민이나 상황을 자세히 말씀해주세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # 봇 응답 생성
        with st.chat_message("assistant"):
            try:
                if st.session_state.bot is None:
                    # 봇이 초기화되지 않은 경우 기본 응답
                    response = "죄송합니다. 현재 시스템 초기화에 문제가 있어 제한된 기능만 제공됩니다. 일반적인 상담 조언을 드리겠습니다만, 완전한 기능을 위해서는 관리자에게 문의해주세요."
                    references = []
                else:
                    response, references = st.session_state.bot.chat(prompt)
                
                st.write(response)
                
                # 참고 경험담 표시
                if references:
                    with st.expander("📚 참고한 경험담들", expanded=False):
                        for i, ref in enumerate(references, 1):
                            source = ref['metadata']['source']
                            title = ref['metadata'].get('title', '제목 없음')
                            score = ref.get('score', 0)
                            
                            st.markdown(f"""
                            **{i}. [{source}] {title}**  
                            유사도: {score:.2f}  
                            {ref['text'][:200]}...
                            """)
                
                # 응답을 세션에 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "references": references
                })
                
            except Exception as e:
                error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 기록 지우기"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main() 