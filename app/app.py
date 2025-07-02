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
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                self.embedder = SentenceTransformer(
                    self.config['embedding_model'], 
                    device=device,  # 모든 플랫폼에서 안정적
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
    
    def _translate_to_korean(self, text: str) -> str:
        """영어 텍스트를 한국어로 번역"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a translator that converts English Reddit posts to natural Korean. Keep the original meaning and tone."},
                    {"role": "user", "content": f"Translate this to Korean: {text}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"번역 오류: {e}")
            return text  # 번역 실패시 원문 반환
    
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
        OpenAI API를 사용해 AITA 스타일 판정 및 상담 응답 생성
        
        Args:
            query: 사용자 질문/고민
            context_chunks: 검색된 유사 경험담
            
        Returns:
            생성된 응답
        """
        from collections import Counter
        
        # 판정 집계
        verdicts = []
        verdict_explanations = []
        
        for chunk in context_chunks:
            verdict = chunk['metadata'].get('verdict', 'UNKNOWN')
            if verdict != 'UNKNOWN':
                verdicts.append(verdict)
                # 간단한 설명 추가
                title_ko = self._translate_to_korean(chunk['metadata'].get('title', ''))
                verdict_explanations.append(f"'{title_ko[:50]}...' → {verdict}")
        
        # 판정 결과 계산
        if verdicts:
            verdict_counts = Counter(verdicts)
            final_verdict = verdict_counts.most_common(1)[0][0]
            verdict_summary = ", ".join([f"{v}: {c}표" for v, c in verdict_counts.items()])
        else:
            final_verdict = "INFO"
            verdict_summary = "판정 정보 부족"
        
        # 판정 의미 설명
        verdict_meanings = {
            "YTA": "You're the A-hole (당신이 잘못했어요)",
            "NTA": "Not the A-hole (당신은 잘못하지 않았어요)", 
            "ESH": "Everyone Sucks Here (모두가 잘못했어요)",
            "NAH": "No A-holes Here (아무도 잘못하지 않았어요)",
            "INFO": "Not Enough Info (정보가 부족해요)"
        }
        
        # Reddit 경험담을 문맥으로 구성 (번역 포함)
        reddit_context = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk['metadata']
            source = metadata['source']
            title_ko = self._translate_to_korean(metadata.get('title', '제목 없음'))
            content_ko = self._translate_to_korean(chunk['text'][:300])  # 긴 내용은 앞부분만
            score = metadata.get('score', 0)
            verdict = metadata.get('verdict', 'UNKNOWN')
            
            # 상위 댓글 추가 (판정 근거로 활용)
            comments_text = ""
            comments = metadata.get('comments', [])
            if comments:
                # 점수 기준 상위 2개 댓글만 포함
                top_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)[:2]
                for j, comment in enumerate(top_comments, 1):
                    comment_ko = self._translate_to_korean(comment.get('message', '')[:200])
                    comment_score = comment.get('score', 0)
                    comments_text += f"\n상위댓글{j}: {comment_ko}... (👍 {comment_score})"
            
            context_text = f"""
경험담 {i} [{source}]:
제목: {title_ko}
내용: {content_ko}...
Reddit 점수: {score}
커뮤니티 판정: {verdict}{comments_text}
"""
            reddit_context.append(context_text.strip())

        # 시스템 프롬프트 (AITA 재판관 역할)
        system_prompt = f"""당신은 Reddit AITA(Am I The A-hole) 커뮤니티의 판례를 학습한 **AI 재판관**입니다. 배석 판사 없이 단독으로 사건을 심리합니다.

[재판관 어투 지침]
- 전문 법조인의 공식·엄숙한 말투 사용
- 첫 문단은 반드시 "본 안건은 …" 형태로 사건 요지를 정리
- 판결문 말미에 **주문** 섹션 추가: "주문. 피고인을 YTA로 판결한다."와 같은 형식
- 필요한 경우 "판시사항", "판단 근거" 항목을 포함
- 불필요한 감정 표현, 농담, 캐주얼한 표현 금지

[판결문 권장 구조]
1. 서론: "본 안건은 …" (사건 요지)
2. 판시사항
3. 판단 근거 (관계 법령·판례·댓글 인용)
4. 주문 (최종 판정: YTA/NTA/ESH/NAH/INFO)
5. 필요 시 조언 또는 부대 의견

[판정 기준]
- YTA (You're the A-hole): 사용자의 행동이 부적절하거나 잘못됨
- NTA (Not the A-hole): 사용자는 잘못하지 않음, 상대방이나 상황이 문제
- ESH (Everyone Sucks Here): 모든 당사자가 각각 잘못한 부분이 있음
- NAH (No A-holes Here): 아무도 특별히 잘못하지 않음, 단순한 의견 차이나 불행한 상황
- INFO (Not Enough Info): 판정하기에 정보가 부족함
"""

        context_text = "\n\n".join(reddit_context) if reddit_context else "유사한 판례를 찾지 못했지만, 일반적인 도덕적 기준으로 판단해드리겠습니다."
        
        user_prompt = f"""사용자 상황: {query}

유사한 Reddit AITA 판례들:
{context_text}

판정 집계 결과: {verdict_summary}
주요 판정 경향: {final_verdict} ({verdict_meanings.get(final_verdict, final_verdict)})

위의 판례들과 커뮤니티 의견을 참고하여, 사용자의 상황에 대해 공정한 AITA 판정을 내려주세요."""

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
    st.title("⚖️ AITA 재판부")
    st.markdown("""
    📜 **본 법정은 Reddit AITA 커뮤니티의 방대한 판례를 학습한 AI 재판관입니다.**  
    사건의 사실관계를 진술하시면, 선례와 댓글을 근거로 **YTA/NTA/ESH/NAH** 중 하나의 판결을 선고하겠습니다.
    """)
    
    # AITA 약어 설명
    with st.expander("📖 AITA 판정 기준", expanded=False):
        st.markdown("""
        **YTA** (You're the A-hole) - 당신이 잘못했어요  
        **NTA** (Not the A-hole) - 당신은 잘못하지 않았어요  
        **ESH** (Everyone Sucks Here) - 모두가 잘못했어요  
        **NAH** (No A-holes Here) - 아무도 잘못하지 않았어요  
        **INFO** (Not Enough Info) - 정보가 부족해요  
        """)
    
    # 사이드바 - 사용법 안내
    with st.sidebar:
        st.header("📖 사용법")
        st.markdown("""
        **어떤 판정을 받을 수 있나요?**
        - 인간관계 갈등 상황
        - 도덕적/윤리적 딜레마
        - 일상생활에서의 선택과 행동
        - 가족, 친구, 연인과의 문제
        - 직장이나 학교에서의 갈등
        
        **예시 상황:**
        - "친구 결혼식에 못 간다고 했는데..."
        - "룸메이트가 청소를 안 해서 화냈어요"
        - "부모님이 원하지 않는 선택을 했어요"
        - "남자친구와 돈 문제로 싸웠어요"
        """)
        
        st.header("⚖️ 판정 방식")
        st.markdown("""
        - Reddit AITA 커뮤니티 판례 분석
        - 비슷한 상황의 집단 지성 활용
        - 공정하고 객관적인 도덕적 판단
        - 건설적인 해결책 제시
        """)
        
        st.header("⚠️ 주의사항")
        st.markdown("""
        - 재미와 성찰을 위한 판정입니다
        - 전문적인 상담은 전문가에게
        - 개인정보는 입력하지 마세요
        - 판정에 너무 의존하지 마세요
        """)
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # 참고 경험담 표시 (assistant 메시지에만)
            if message["role"] == "assistant" and "references" in message:
                references = message["references"]
                
                # 판정 집계 표시
                from collections import Counter
                verdicts = [ref['metadata'].get('verdict', 'UNKNOWN') for ref in references if ref['metadata'].get('verdict', 'UNKNOWN') != 'UNKNOWN']
                
                if verdicts:
                    verdict_counts = Counter(verdicts)
                    st.markdown("### ⚖️ AITA 커뮤니티 판정 집계")
                    
                    # 판정 결과를 예쁘게 표시
                    cols = st.columns(len(verdict_counts))
                    for i, (verdict, count) in enumerate(verdict_counts.items()):
                        with cols[i]:
                            st.metric(verdict, f"{count}표")
                
                with st.expander("📚 참고한 경험담들 (한국어 번역)", expanded=False):
                    for i, ref in enumerate(references, 1):
                        source = ref['metadata']['source']
                        title = ref['metadata'].get('title', '제목 없음')
                        score = ref.get('score', 0)
                        verdict = ref['metadata'].get('verdict', 'UNKNOWN')
                        reddit_score = ref['metadata'].get('score', 0)
                        url = ref['metadata'].get('url', '')
                        
                        # 제목과 내용 번역 (봇이 있을 때만)
                        if st.session_state.bot and hasattr(st.session_state.bot, '_translate_to_korean'):
                            title_ko = st.session_state.bot._translate_to_korean(title)
                            content_ko = st.session_state.bot._translate_to_korean(ref['text'][:300])
                        else:
                            title_ko = title
                            content_ko = ref['text'][:300]
                        
                        # 링크 포함 제목
                        if url:
                            title_display = f"[{title_ko}]({url})"
                        else:
                            title_display = title_ko
                        
                        st.markdown(f"""
                        **{i}. [{source}] {title_display}**  
                        유사도: {score:.3f} | Reddit 점수: {reddit_score} | 커뮤니티 판정: **{verdict}**  
                        
                        {content_ko}...
                        """)
                        
                        # 상위 댓글 표시
                        comments = ref['metadata'].get('comments', [])
                        if comments:
                            st.markdown("**💬 주요 댓글들:**")
                            top_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)[:3]
                            for j, comment in enumerate(top_comments, 1):
                                if st.session_state.bot and hasattr(st.session_state.bot, '_translate_to_korean'):
                                    comment_ko = st.session_state.bot._translate_to_korean(comment.get('message', '')[:250])
                                else:
                                    comment_ko = comment.get('message', '')[:250]
                                comment_score = comment.get('score', 0)
                                st.markdown(f"- **댓글{j}:** {comment_ko}... _(👍 {comment_score})_")
                        
                        st.markdown("---")
    
    # 사용자 입력
    if prompt := st.chat_input("판정받고 싶은 상황을 자세히 설명해주세요... (예: 내가 잘못한 걸까요?)"):
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
                    # 판정 집계 표시
                    from collections import Counter
                    verdicts = [ref['metadata'].get('verdict', 'UNKNOWN') for ref in references if ref['metadata'].get('verdict', 'UNKNOWN') != 'UNKNOWN']
                    
                    if verdicts:
                        verdict_counts = Counter(verdicts)
                        st.markdown("### ⚖️ AITA 커뮤니티 판정 집계")
                        
                        # 판정 결과를 예쁘게 표시
                        cols = st.columns(len(verdict_counts))
                        for i, (verdict, count) in enumerate(verdict_counts.items()):
                            with cols[i]:
                                st.metric(verdict, f"{count}표")
                    
                    with st.expander("📚 참고한 경험담들 (한국어 번역)", expanded=False):
                        for i, ref in enumerate(references, 1):
                            source = ref['metadata']['source']
                            title = ref['metadata'].get('title', '제목 없음')
                            score = ref.get('score', 0)
                            verdict = ref['metadata'].get('verdict', 'UNKNOWN')
                            reddit_score = ref['metadata'].get('score', 0)
                            url = ref['metadata'].get('url', '')
                            
                            # 제목과 내용 번역 (실시간)
                            if hasattr(st.session_state.bot, '_translate_to_korean'):
                                title_ko = st.session_state.bot._translate_to_korean(title)
                                content_ko = st.session_state.bot._translate_to_korean(ref['text'][:300])
                            else:
                                title_ko = title
                                content_ko = ref['text'][:300]
                            
                            # 링크 포함 제목
                            if url:
                                title_display = f"[{title_ko}]({url})"
                            else:
                                title_display = title_ko
                            
                            st.markdown(f"""
                            **{i}. [{source}] {title_display}**  
                            유사도: {score:.3f} | Reddit 점수: {reddit_score} | 커뮤니티 판정: **{verdict}**  
                            
                            {content_ko}...
                            """)
                            
                            # 상위 댓글 표시
                            comments = ref['metadata'].get('comments', [])
                            if comments:
                                st.markdown("**💬 주요 댓글들:**")
                                top_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)[:3]
                                for j, comment in enumerate(top_comments, 1):
                                    if hasattr(st.session_state.bot, '_translate_to_korean'):
                                        comment_ko = st.session_state.bot._translate_to_korean(comment.get('message', '')[:250])
                                    else:
                                        comment_ko = comment.get('message', '')[:250]
                                    comment_score = comment.get('score', 0)
                                    st.markdown(f"- **댓글{j}:** {comment_ko}... _(👍 {comment_score})_")
                            
                            st.markdown("---")
                
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