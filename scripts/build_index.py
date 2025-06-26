"""
PDF 텍스트 추출 및 FAISS 인덱스 생성 스크립트
M2 MacBook Air 8GB RAM에 최적화
"""

import os
import gc
import json
import pickle
import psutil
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


class PDFIndexBuilder:
    """PDF 문서를 처리하고 FAISS 인덱스를 생성하는 클래스"""
    
    def __init__(self, embedding_model_name: str = "intfloat/multilingual-e5-base"):
        """
        초기화
        
        Args:
            embedding_model_name: 임베딩 모델 이름 (e5-base는 한국어 지원)
        """
        print(f"🚀 임베딩 모델 로딩: {embedding_model_name}")
        # MPS 환경에서 메모리 부족을 방지하기 위해 CPU 사용
        self.embedder = SentenceTransformer(embedding_model_name, device="cpu")
        
        # M2 Mac 최적화: 배치 크기 축소
        self.embedder.max_seq_length = 512  # 메모리 절약
        self.batch_size = 8  # 8GB RAM에 적합
        
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # 약 200 토큰
            chunk_overlap=50,  # 중복으로 문맥 보존
            separators=["\n\n", "\n", ".", "。", "!", "?", ";", ":", " ", ""],
            length_function=len,
        )
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            (전체 텍스트, 메타데이터)
        """
        print(f"📄 PDF 읽는 중: {pdf_path.name}")
        
        reader = PdfReader(pdf_path)
        text_parts = []
        
        # 메모리 효율적인 페이지별 처리
        for i, page in enumerate(tqdm(reader.pages, desc="페이지 추출")):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
                    
                # 10페이지마다 메모리 정리
                if i % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"⚠️ 페이지 {i+1} 추출 실패: {e}")
                continue
        
        full_text = "\n\n".join(text_parts)
        
        metadata = {
            "source": pdf_path.name,
            "pages": len(reader.pages),
            "characters": len(full_text)
        }
        
        print(f"✅ 추출 완료: {len(full_text):,} 문자")
        
        return full_text, metadata
    
    def create_chunks(self, text: str, source: str) -> List[Dict]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 전체 텍스트
            source: 소스 파일명
            
        Returns:
            청크 리스트 (텍스트 + 메타데이터)
        """
        print(f"✂️ 텍스트 청킹 중...")
        
        chunks = self.text_splitter.split_text(text)
        
        # 각 청크에 메타데이터 추가
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # 빈 청크 제외
                chunk_docs.append({
                    "text": chunk,
                    "metadata": {
                        "source": source,
                        "chunk_id": i,
                        "chunk_size": len(chunk)
                    }
                })
        
        print(f"✅ {len(chunk_docs)}개 청크 생성")
        
        return chunk_docs
    
    def create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        배치 단위로 임베딩 생성 (메모리 효율적)
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            임베딩 벡터 배열
        """
        embeddings = []
        
        # 작은 배치로 나누어 처리
        for i in tqdm(range(0, len(texts), self.batch_size), desc="임베딩 생성"):
            batch = texts[i:i + self.batch_size]
            
            # 임베딩 생성
            batch_embeddings = self.embedder.encode(
                batch,
                normalize_embeddings=True,  # 정규화로 코사인 유사도 계산 최적화
                show_progress_bar=False
            )
            
            embeddings.extend(batch_embeddings)
            
            # 메모리 정리
            if i % (self.batch_size * 10) == 0:
                gc.collect()
                
                # 메모리 사용량 모니터링
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    print(f"⚠️ 메모리 사용률 높음: {memory_percent:.1f}%")
        
        return np.array(embeddings, dtype='float32')
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        FAISS 인덱스 생성
        
        Args:
            embeddings: 임베딩 벡터
            
        Returns:
            FAISS 인덱스
        """
        print(f"🔨 FAISS 인덱스 생성 중...")
        
        dimension = embeddings.shape[1]
        
        # M2 Mac 최적화: 간단한 인덱스 사용
        # IndexFlatL2는 메모리 효율적이고 빠름
        index = faiss.IndexFlatL2(dimension)
        
        # 정규화된 벡터를 사용하므로 IP (내적) 인덱스도 가능
        # index = faiss.IndexFlatIP(dimension)
        
        index.add(embeddings)
        
        print(f"✅ 인덱스 생성 완료: {index.ntotal}개 벡터")
        
        return index
    
    def process_pdfs(self, pdf_dir: Path, output_dir: Path):
        """
        PDF 파일들을 처리하고 인덱스 생성
        
        Args:
            pdf_dir: PDF 파일 디렉토리
            output_dir: 출력 디렉토리
        """
        # 출력 디렉토리 생성
        output_dir.mkdir(exist_ok=True)
        
        # PDF 파일 목록
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"📚 {len(pdf_files)}개 PDF 파일 발견")
        
        all_chunks = []
        
        # 각 PDF 처리
        for pdf_path in pdf_files:
            try:
                # 텍스트 추출
                text, metadata = self.extract_text_from_pdf(pdf_path)
                
                # 청크 생성
                chunks = self.create_chunks(text, pdf_path.name)
                all_chunks.extend(chunks)
                
                # 메모리 정리
                gc.collect()
                
            except Exception as e:
                print(f"❌ {pdf_path.name} 처리 실패: {e}")
                continue
        
        print(f"\n📊 전체 청크 수: {len(all_chunks)}")
        
        # 텍스트만 추출
        texts = [chunk["text"] for chunk in all_chunks]
        
        # 임베딩 생성
        print("\n🧠 임베딩 생성 시작...")
        embeddings = self.create_embeddings_batch(texts)
        
        # FAISS 인덱스 생성
        index = self.build_faiss_index(embeddings)
        
        # 저장
        print("\n💾 인덱스 및 메타데이터 저장 중...")
        
        # FAISS 인덱스 저장
        faiss.write_index(index, str(output_dir / "dream_index.faiss"))
        
        # 청크 정보 저장 (텍스트 + 메타데이터)
        with open(output_dir / "chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        
        # 설정 정보 저장
        config = {
            "embedding_model": "intfloat/multilingual-e5-base",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "total_chunks": len(all_chunks),
            "dimension": embeddings.shape[1],
            "pdf_files": [pdf.name for pdf in pdf_files]
        }
        
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("\n✅ 인덱싱 완료!")
        print(f"📁 출력 위치: {output_dir}")


def main():
    """메인 실행 함수"""
    # 경로 설정
    pdf_dir = Path("data")
    output_dir = Path("index")
    
    # 인덱스 빌더 생성 및 실행
    builder = PDFIndexBuilder()
    builder.process_pdfs(pdf_dir, output_dir)
    
    # 최종 메모리 상태
    memory = psutil.virtual_memory()
    print(f"\n💻 메모리 사용: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")


if __name__ == "__main__":
    main() 