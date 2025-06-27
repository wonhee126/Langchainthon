#!/usr/bin/env python
"""
꿈해몽 RAG 인덱스 빌더
PDF 문서들을 청크로 나누고 FAISS 벡터 인덱스 생성
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
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class DreamIndexBuilder:
    """꿈해몽 문서 인덱스 빌더"""
    
    def __init__(self):
        """초기화"""
        self.embedding_model = "intfloat/multilingual-e5-base"
        self.index_dir = Path("index")
        self.data_dir = Path("data")
        
        # 텍스트 분할기 설정 (더 큰 청크로 설정)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 출력 디렉토리 생성
        self.index_dir.mkdir(exist_ok=True)
        
    def load_pdfs(self) -> List[Dict]:
        """PDF 파일들을 로드하고 청크로 분할"""
        print("📚 PDF 파일 로딩 중...")
        
        documents = []
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"{self.data_dir}에 PDF 파일이 없습니다.")
        
        for pdf_path in tqdm(pdf_files, desc="PDF 로딩"):
            print(f"  - {pdf_path.name} 처리 중...")
            
            # PDF 로더 사용
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # 전체 텍스트 합치기
            full_text = "\n".join([page.page_content for page in pages])
            
            # 청크로 분할
            chunks = self.text_splitter.split_text(full_text)
            
            # 메타데이터와 함께 저장
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 100:  # 너무 짧은 청크 제외
                    documents.append({
                        'text': chunk.strip(),
                        'metadata': {
                            'source': pdf_path.name,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    })
        
        print(f"✅ 총 {len(documents)}개 청크 생성")
        return documents
    
    def create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """문서 청크들의 임베딩 생성"""
        print("🧠 임베딩 모델 로딩...")
        
        # CPU 사용 강제
        embedder = SentenceTransformer(
            self.embedding_model, 
            device="cpu"
        )
        embedder.max_seq_length = 512
        
        print("🔄 임베딩 생성 중...")
        texts = [doc['text'] for doc in documents]
        
        # 배치 처리로 메모리 효율성 증대
        batch_size = 16
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성"):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedder.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # 메모리 정리
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        # 모든 배치 결합
        all_embeddings = np.vstack(embeddings)
        print(f"✅ 임베딩 생성 완료: {all_embeddings.shape}")
        
        return all_embeddings
    
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
        faiss.write_index(index, str(self.index_dir / "dream_index.faiss"))
        
        # 문서 청크 저장
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        # 설정 정보 저장
        config = {
            'embedding_model': self.embedding_model,
            'total_chunks': len(documents),
            'embedding_dimension': embeddings.shape[1],
            'index_type': 'FlatIP',
            'chunk_size': 300,
            'chunk_overlap': 50
        }
        
        with open(self.index_dir / "config.json", "w", encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 인덱스 저장 완료: {self.index_dir}")
    
    def build(self):
        """전체 인덱스 빌드 프로세스"""
        print("🌙 꿈해몽 RAG 인덱스 빌드 시작")
        print("=" * 50)
        
        try:
            # 1. PDF 로딩 및 청크 분할
            documents = self.load_pdfs()
            
            # 2. 임베딩 생성
            embeddings = self.create_embeddings(documents)
            
            # 3. FAISS 인덱스 구축
            index = self.build_faiss_index(embeddings)
            
            # 4. 저장
            self.save_index(index, documents, embeddings)
            
            print("=" * 50)
            print("🎉 인덱스 빌드 완료!")
            print(f"📊 통계:")
            print(f"  - 총 문서 수: {len(set(doc['metadata']['source'] for doc in documents))}")
            print(f"  - 총 청크 수: {len(documents)}")
            print(f"  - 임베딩 차원: {embeddings.shape[1]}")
            print(f"  - 인덱스 크기: {index.ntotal}")
            
        except Exception as e:
            print(f"❌ 빌드 실패: {e}")
            raise


def main():
    """메인 함수"""
    builder = DreamIndexBuilder()
    builder.build()


if __name__ == "__main__":
    main() 