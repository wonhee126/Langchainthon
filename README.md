# 🌙 한밤의 꿈해몽 상담가 - RAG 챗봇

프로이트의 『꿈의 해석』과 WHO 수면 가이드를 기반으로 한 AI 꿈해몽 상담 챗봇입니다.  
OpenAI API를 사용하여 고품질의 꿈 해석을 제공합니다.

## 🎯 주요 기능

- **3단계 꿈 해석**: 고전적 정신분석, 현대 수면과학, 통합 조언
- **RAG 기반**: FAISS 벡터 검색으로 관련 문서 참조
- **빠른 응답**: OpenAI API로 고품질 응답 생성
- **메모리 효율적**: 로컬 LLM 불필요로 메모리 사용량 최소화

## 🛠 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| LLM | OpenAI o3-mini (고정) |
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

### 2. OpenAI API 키 설정

**로컬 실행용:**
```bash
# .streamlit/secrets.toml 파일에 API 키 설정
echo 'OPENAI_API_KEY = "your_openai_api_key_here"' > .streamlit/secrets.toml
```

**Streamlit Cloud 배포용:**
1. GitHub에 코드 푸시
2. [Streamlit Cloud](https://share.streamlit.io)에 로그인
3. 앱 생성 후 Settings → Secrets에서 다음 설정:
   ```
   OPENAI_API_KEY = "your_openai_api_key_here"
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

## 🚀 Streamlit Cloud 배포

### 1. GitHub에 코드 업로드
```bash
git add .
git commit -m "Add dream interpretation app"
git push origin main
```

### 2. Streamlit Cloud 배포
1. [Streamlit Cloud](https://share.streamlit.io)에 GitHub 계정으로 로그인
2. "New app" 클릭
3. GitHub 저장소 선택
4. Main file path: `app/app.py`
5. "Deploy!" 클릭

### 3. API 키 설정
1. 배포된 앱의 "Settings" → "Secrets" 이동
2. 다음 내용 입력:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```
3. "Save" 클릭

이제 전 세계 어디서든 당신의 꿈해몽 앱에 접근할 수 있습니다! 🌍

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

### 배포 및 사용 최적화

1. **Streamlit Cloud 배포**: 무료 호스팅으로 언제든 접근 가능
2. **o3-mini 모델**: 최신 OpenAI 모델로 고품질 응답 보장
3. **토큰 제한**: `max_tokens=600`으로 적절한 응답 길이 유지
4. **캐시 활용**: 임베딩 모델은 로컬 캐시 사용

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
| 임베딩 모델 로딩 | ~10초 (첫 실행) |
| 응답 생성 | ~2-5초 (API 호출) |
| 메모리 사용 | ~1-2GB (로컬 LLM 불필요) |

## 🐛 문제 해결

### "OpenAI API 키 미설정" 오류

**로컬 실행:**
```bash
# secrets.toml 파일 확인
cat .streamlit/secrets.toml

# API 키 설정
echo 'OPENAI_API_KEY = "your_actual_api_key"' > .streamlit/secrets.toml
```

**Streamlit Cloud:**
1. 앱 페이지에서 "Settings" 클릭
2. "Secrets" 탭에서 API 키 확인/수정

### "API 호출 실패" 오류

1. API 키 유효성 확인
2. 계정 잔액 확인
3. 네트워크 연결 확인

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