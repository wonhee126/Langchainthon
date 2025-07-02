#!/usr/bin/env python
"""
Reddit 상담사 RAG 인덱스 빌더
AITA JSON 파일을 청크로 나누고 FAISS 벡터 인덱스 생성
"""

import os
import json
import pickle
import gc
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer  # 실제 임베딩 활성화!
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch


class RedditIndexBuilder:
    """Reddit 상담사 인덱스 빌더"""
    
    def __init__(self):
        """초기화"""
        self.embedding_model = "intfloat/multilingual-e5-small"
        self.embedding_dim = 384
        self.index_dir = Path("index")
        self.data_dir = Path("data")
        
        # 텍스트 분할기 설정 (Reddit 포스트에 맞게 조정)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reddit 포스트는 보통 길어서 크기 증가
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 출력 디렉토리 생성
        self.index_dir.mkdir(exist_ok=True)
        
    def infer_verdict(self, comments: List[Dict], top_k: int = 5) -> str:
        """댓글에서 AITA 판정 추출"""
        votes = {"YTA": 0, "NTA": 0, "ESH": 0, "NAH": 0, "INFO": 0}
        
        if not comments:
            return "UNKNOWN"
        
        # 점수 기준 상위 댓글들
        top_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        
        for comment in top_comments:
            message = comment.get('message', '').upper().strip()
            # 각 약어가 댓글 시작 부분에 있는지 확인
            for abbr in votes:
                if message.startswith(abbr):
                    votes[abbr] += 1
                    break
        
        # 가장 많은 표를 받은 판정 반환
        max_count = max(votes.values())
        if max_count > 0:
            return max(votes, key=votes.get)
        return "UNKNOWN"

    def load_json_data(self, filename: str, source_name: str) -> List[Dict]:
        """JSON 파일 로드 및 처리 (AITA 판정 추출 포함)"""
        print(f"📚 {source_name} JSON 파일 로딩 중...")
        
        documents = []
        json_file = self.data_dir / filename
        
        if not json_file.exists():
            print(f"⚠️ {json_file}을 찾을 수 없습니다. 스킵합니다.")
            return documents
        
        # AITA는 JSON 배열 형태, TIFU는 줄 단위 JSON
        if source_name == "AITA":
            # JSON 배열 전체를 한 번에 로드
            with open(json_file, 'r', encoding='utf-8') as f:
                posts = json.load(f)
            
            print(f"📄 {len(posts)}개의 AITA 포스트 로드됨")
            
            for post_num, post in enumerate(tqdm(posts, desc=f"{source_name} 데이터 처리")):
                try:
                    
                    # 데이터 소스별 필드 매핑
                    if source_name == "TIFU":
                        title = post.get('trimmed_title', post.get('title', ''))
                        content = post.get('selftext_without_tldr', post.get('selftext', ''))
                        tldr = post.get('tldr', '')
                        
                        # 전체 텍스트 구성
                        full_text = f"제목: {title}\n\n내용: {content}"
                        if tldr:
                            full_text += f"\n\n요약: {tldr}"
                            
                    elif source_name == "AITA":
                        # 새로운 AITA 데이터 구조: submission 객체 안에 모든 정보
                        if 'submission' in post:
                            submission = post['submission']
                            title = submission.get('title', '')
                            content = submission.get('selftext', '')
                            score = submission.get('score', 0)
                            submission_id = submission.get('submission_id', '')
                            comments = post.get('comments', [])
                            permalink = submission.get('permalink', '')
                        else:
                            # 기존 구조 대응
                            title = post.get('title', '')
                            content = post.get('selftext', '')
                            score = post.get('score', 0)
                            submission_id = post.get('submission_id', post.get('id', ''))
                            comments = post.get('comments', [])
                            permalink = post.get('permalink', '')
                        
                        # 전체 URL 생성
                        if permalink:
                            full_url = f"https://reddit.com{permalink}" if not permalink.startswith("http") else permalink
                        else:
                            full_url = f"https://reddit.com/r/AmItheAsshole/comments/{submission_id}"
                        
                        # AITA 판정 추출
                        verdict = self.infer_verdict(comments)
                        
                        # AITA는 보통 제목에 "AITA for..." 형태
                        full_text = f"제목: {title}\n\n내용: {content}" if title else content
                    
                    else:
                        # 기본 처리 (새로운 데이터 소스 대응)
                        title = post.get('title', '')
                        content = post.get('selftext', post.get('content', post.get('text', '')))
                        full_text = f"제목: {title}\n\n내용: {content}" if title else content
                    
                    # 텍스트가 너무 짧으면 스킵
                    if not content or len(full_text.strip()) < 100:
                        continue
                    
                    # 청크로 분할
                    chunks = self.text_splitter.split_text(full_text)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:  # 너무 짧은 청크 제외
                            documents.append({
                                'text': chunk.strip(),
                                'metadata': {
                                    'source': source_name,
                                    'post_id': submission_id if source_name == "AITA" else post.get('id', post.get('submission_id', f'{source_name.lower()}_{line_num}')),
                                    'title': title,
                                    'chunk_id': i,
                                    'total_chunks': len(chunks),
                                    'score': score if source_name == "AITA" else post.get('score', 0),
                                    'num_comments': post.get('num_comments', 0),
                                    'comments': post.get('comments', []),
                                    'verdict': verdict if source_name == "AITA" else "N/A",
                                    'url': full_url
                                }
                            })
                    
                    # 메모리 관리 (1000개씩 처리 후 정리)
                    if post_num % 1000 == 0 and post_num > 0:
                        gc.collect()
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"포스트 {post_num} 처리 오류: {e}")
                    continue
                    
        else:
            # TIFU는 줄 단위 JSON으로 처리
            with open(json_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"{source_name} 데이터 로딩")):
                    try:
                        post = json.loads(line.strip())
                        
                        # TIFU 데이터 처리
                        title = post.get('trimmed_title', post.get('title', ''))
                        content = post.get('selftext_without_tldr', post.get('selftext', ''))
                        tldr = post.get('tldr', '')
                        
                        # 전체 텍스트 구성
                        full_text = f"제목: {title}\n\n내용: {content}"
                        if tldr:
                            full_text += f"\n\n요약: {tldr}"
                        
                        # 텍스트가 너무 짧으면 스킵
                        if not content or len(full_text.strip()) < 100:
                            continue
                        
                        # 청크로 분할
                        chunks = self.text_splitter.split_text(full_text)
                        
                        for i, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 50:  # 너무 짧은 청크 제외
                                documents.append({
                                    'text': chunk.strip(),
                                    'metadata': {
                                        'source': source_name,
                                        'post_id': post.get('id', post.get('submission_id', f'{source_name.lower()}_{line_num}')),
                                        'title': title,
                                        'chunk_id': i,
                                        'total_chunks': len(chunks),
                                        'score': post.get('score', 0),
                                        'num_comments': post.get('num_comments', 0)
                                    }
                                })
                        
                        # 메모리 관리 (1000개씩 처리 후 정리)
                        if line_num % 1000 == 0 and line_num > 0:
                            gc.collect()
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"라인 {line_num} 처리 오류: {e}")
                        continue
        
        print(f"✅ {source_name}에서 {len(documents)}개 청크 생성")
        return documents
    
    def create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """문서 청크들의 실제 의미론적 임베딩 생성 (크로스 플랫폼)"""
        print("🧠 실제 의미론적 임베딩 생성 중...")
        
        # 플랫폼별 최적화 및 디바이스 설정
        import platform
        
        # MPS (Apple Silicon GPU) 사용 설정
        if torch.backends.mps.is_available() and torch.backends.mps.is_built() and platform.system() == "Darwin":
            device = "mps"
            print("🚀 Apple Silicon MPS 가속 활성화!")
        else:
            device = "cpu"
            print("🖥️ CPU 사용 (MPS 사용 불가)")
            
        print(f"📥 {self.embedding_model} 모델 로딩 (디바이스: {device})...")
        embedder = SentenceTransformer(self.embedding_model, device=device)
        
        # 텍스트 추출
        texts = [doc['text'] for doc in documents]
        print(f"📝 {len(texts)}개 텍스트 임베딩 생성 시작...")
        
        # 배치 처리로 메모리 효율성 확보
        # MPS/GPU 환경에서는 배치 크기를 늘리는 것이 효율적
        batch_size = 128
        print(f"⚡ 배치 크기: {batch_size} (속도 최적화)")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="의미론적 임베딩 생성"):
            batch_texts = texts[i:i+batch_size]
            
            # 임베딩 생성 (정규화 자동 적용)
            batch_embeddings = embedder.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            all_embeddings.append(batch_embeddings)
            
            # 메모리 정리
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        # 모든 배치 결합
        final_embeddings = np.vstack(all_embeddings).astype('float32')
        print(f"✅ 실제 임베딩 생성 완료: {final_embeddings.shape}")
        print("🎯 이제 의미론적 검색이 가능합니다!")
        
        return final_embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """FAISS 인덱스 구축"""
        print("🏗️ FAISS 인덱스 구축 중...")
        
        # 차원 수
        dimension = embeddings.shape[1]
        
        # IndexFlatIP 사용 (코사인 유사도)
        index = faiss.IndexFlatIP(dimension)
        
        # 임베딩 추가
        index.add(embeddings.astype('float32'))
        
        print(f"✅ FAISS 인덱스 구축 완료: {index.ntotal}개 벡터")
        return index
    
    def save_index(self, index, documents: List[Dict], embeddings: np.ndarray):
        """인덱스와 메타데이터 저장"""
        print("💾 인덱스 저장 중...")
        
        # FAISS 인덱스 저장
        faiss.write_index(index, str(self.index_dir / "reddit_index.faiss"))
        
        # 문서 청크 저장
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        # 설정 정보 저장
        config = {
            'embedding_model': self.embedding_model,
            'total_chunks': len(documents),
            'embedding_dimension': embeddings.shape[1],
            'index_type': 'FlatIP',
            'chunk_size': 800,
            'chunk_overlap': 100,
            'data_sources': ['AITA'],
            'data_format': 'JSON',
            'note': 'Using semantic embeddings with multilingual-e5-base model',
            'verdict_enabled': True,
            'verdict_types': ['YTA', 'NTA', 'ESH', 'NAH', 'INFO', 'UNKNOWN']
        }
        
        with open(self.index_dir / "config.json", "w", encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 인덱스 저장 완료: {self.index_dir}")
    
    def build(self):
        """전체 인덱스 빌드 프로세스"""
        print("🤖 Reddit 상담사 RAG 인덱스 빌드 시작 (AITA 전용 버전)")
        print("=" * 60)
        
        try:
            # JSON 파일들 로딩
            documents = []
            
            # TIFU 데이터 로드 제거
            # tifu_docs = self.load_json_data("tifu_all_tokenized_and_filtered.json", "TIFU")
            # documents.extend(tifu_docs)
            
            # AITA 데이터만 로드
            aita_docs = self.load_json_data("aita_data.json", "AITA")
            documents.extend(aita_docs)
            
            if not documents:
                raise ValueError("로드된 문서가 없습니다. 데이터 파일을 확인해주세요.")
            
            print(f"📊 총 {len(documents)}개 문서 청크 로드 완료")
            
            # 임베딩 생성 (임시 랜덤)
            embeddings = self.create_embeddings(documents)
            
            # FAISS 인덱스 구축
            index = self.build_faiss_index(embeddings)
            
            # 저장
            self.save_index(index, documents, embeddings)
            
            print("=" * 60)
            print("🎉 인덱스 빌드 완료!")
            print(f"  - 총 청크 수: {len(documents)}")
            print(f"  - 임베딩 차원: {embeddings.shape[1]}")
            print(f"  - 인덱스 크기: {index.ntotal}")
            print()
            print("🎯 실제 의미론적 임베딩을 사용하여 정확한 검색이 가능합니다!")
            print("   multilingual-e5-base 모델로 한국어-영어 크로스링구얼 검색 지원")
            print()
            print("📁 데이터 파일 정보:")
            print("   - AITA: aita_data.json ✅")
            
        except Exception as e:
            print(f"❌ 빌드 실패: {e}")
            raise


def main():
    """메인 함수"""
    builder = RedditIndexBuilder()
    builder.build()


if __name__ == "__main__":
    main() 