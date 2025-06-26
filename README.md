# 🌙 한밤의 꿈해몽 상담가 - RAG 챗봇

프로이트의 『꿈의 해석』과 WHO 수면 가이드를 기반으로 한 AI 꿈해몽 상담 챗봇입니다.  
M2 MacBook Air 8GB RAM 환경에 최적화되어 있습니다.

## 🎯 주요 기능

- **3단계 꿈 해석**: 고전적 정신분석, 현대 수면과학, 통합 조언
- **RAG 기반**: FAISS 벡터 검색으로 관련 문서 참조
- **빠른 응답**: 3초 이내 응답 생성 (로컬 실행)
- **메모리 효율적**: 8GB RAM에서도 원활히 작동

## 🛠 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| LLM | Qwen 2.5-7B-Instruct-4bit (MLX) |
| 임베딩 | intfloat/multilingual-e5-base |
| 벡터 DB | FAISS (CPU) |
| 프레임워크 | Streamlit + LangChain |
| 런타임 | Python 3.11+ |

## 📦 설치 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
```

### 2. Qwen 모델 다운로드

```bash
# MLX 커뮤니티 모델 다운로드 (약 4.2GB)
mlx-lm download mlx-community/Qwen2.5-7B-Instruct-4bit
```

### 3. PDF 인덱싱

```bash
# PDF 파일을 벡터 인덱스로 변환
python scripts/build_index.py
```

이 과정은 처음 한 번만 실행하면 됩니다. 약 5-10분 소요됩니다.

## 🚀 실행 방법

```bash
# Streamlit 앱 실행
streamlit run app/app.py

# 또는 포트 지정
streamlit run app/app.py --server.port 8080
```

브라우저에서 `http://localhost:8501` 접속

## 📁 프로젝트 구조

```
dream_bot/
├── app/
│   └── app.py              # Streamlit 메인 앱
├── data/
│   ├── freud_dreams.pdf    # 프로이트 『꿈의 해석』
│   └── who_sleep.pdf       # WHO 수면 가이드
├── index/                  # FAISS 인덱스 저장 (자동 생성)
│   ├── dream_index.faiss
│   ├── chunks.pkl
│   └── config.json
├── scripts/
│   └── build_index.py      # PDF 인덱싱 스크립트
└── requirements.txt        # 의존성 목록
```

## 💡 사용법

1. **꿈 입력**: 채팅창에 꿈의 내용을 자세히 설명
2. **해석 확인**: 3가지 관점의 해석을 확인
   - 고전적 해석 (프로이트)
   - 현대 과학 (WHO)
   - 통합 조언
3. **대화 이어가기**: 추가 질문으로 더 깊은 해석 가능

## ⚡ 성능 최적화 팁

### M2 MacBook Air 8GB RAM 최적화

1. **백그라운드 앱 종료**: 실행 전 불필요한 앱 종료
2. **브라우저 탭 최소화**: 메모리 절약
3. **모델 캐싱**: 첫 실행 후 모델이 캐시되어 더 빨라집니다

### 메모리 부족 시

```bash
# 환경 변수로 메모리 제한 설정
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

## 🔧 고급 설정

### PDF 교체

새로운 PDF를 추가하려면:

1. `data/` 폴더에 PDF 복사
2. `python scripts/build_index.py` 재실행
3. 앱 재시작

### 청크 크기 조정

`scripts/build_index.py`에서:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 크기 조정 (기본: 200)
    chunk_overlap=50,    # 중복 조정 (기본: 50)
)
```

### 검색 결과 수 변경

`app/app.py`에서:

```python
relevant_chunks = st.session_state.rag_bot.search_similar_chunks(prompt, k=6)  # k값 조정
```

## 📊 성능 벤치마크

| 지표 | 값 |
|------|-----|
| 인덱싱 시간 | ~5분 (2개 PDF) |
| 모델 로딩 | ~30초 (첫 실행) |
| 응답 생성 | ~2-3초 |
| 메모리 사용 | ~6-7GB |

## 🐛 문제 해결

### "모델을 찾을 수 없음" 오류

```bash
# 모델 재다운로드
mlx-lm download mlx-community/Qwen2.5-7B-Instruct-4bit --force
```

### 메모리 부족 오류

1. 다른 앱 종료
2. 청크 크기 축소
3. 배치 크기 축소

### 인덱스 파일 없음 오류

```bash
# 인덱스 재생성
rm -rf index/
python scripts/build_index.py
```

## 📜 라이선스 및 저작권

- **코드**: MIT License
- **데이터**:
  - Freud's "The Interpretation of Dreams": Public Domain
  - WHO Sleep Guidelines: CC BY-NC-SA 3.0 IGO
- **모델**: Qwen 2.5 - Apache 2.0 License

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

질문이나 제안사항이 있으시면 Issues를 통해 알려주세요!

---

Made with 💜 for better sleep and dream understanding 🌙 