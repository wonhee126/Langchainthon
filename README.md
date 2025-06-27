# 🎭 프로이트 박사의 꿈해몽 상담소

지그문트 프로이트의 『꿈의 해석』을 기반으로 한 AI 꿈해몽 상담 챗봇입니다.  
RAG(Retrieval-Augmented Generation) 시스템으로 원문을 참조하여 정확한 정신분석학적 해석을 제공합니다.

> *"꿈은 무의식으로 가는 왕도이다"* - 지그문트 프로이트

## ⚠️ 이용 안내

**이 서비스는 오락 목적으로만 사용해주세요.**  
프로이트의 꿈 해석 이론은 20세기 초 이론으로, 현대 과학에서는 검증되지 않은 부분이 많습니다.  
실제 심리적 고민이 있으시면 전문 상담사와 상담하시기 바랍니다.

## 🎯 주요 기능

- **📘 문헌 기반 분석**: 『꿈의 해석』 원문을 직접 인용하며 해석
- **🎭 프로이트의 추론적 해석**: 개인적 통찰과 철학적 사색 제공
- **🔍 고품질 RAG 시스템**: FAISS 벡터 검색으로 관련 원문 자동 검색
- **📊 출처 투명성**: 실제 참고한 문헌 구절을 사용자에게 공개
- **⚡ 빠른 응답**: OpenAI GPT-4o로 고품질 응답 생성

## 🛠 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| LLM | OpenAI GPT-4.1 |
| 임베딩 | intfloat/multilingual-e5-base |
| 벡터 DB | FAISS (CPU) |
| 프레임워크 | Streamlit + LangChain |
| 런타임 | Python 3.11+ |

## 📦 설치 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone <your-repo-url>
cd dream_bot

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는 Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

⚠️ **보안 주의**: API 키는 절대 깃허브에 업로드하지 마세요!

**로컬 실행용:**
```bash
# .streamlit/secrets.toml 파일에 API 키 설정
mkdir -p .streamlit
echo 'OPENAI_API_KEY = "your_openai_api_key_here"' > .streamlit/secrets.toml

# .gitignore에 의해 자동으로 보호됨 (이미 설정됨)
```

**Streamlit Cloud 배포용:**
1. GitHub에 코드 푸시
2. [Streamlit Cloud](https://share.streamlit.io)에 로그인
3. 앱 생성 후 Settings → Secrets에서 다음 설정:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

### 3. 인덱스 확인

프로젝트에는 이미 프로이트의 『꿈의 해석』 인덱스가 포함되어 있습니다.  
만약 인덱스를 재생성하려면:

```bash
# PDF 파일을 벡터 인덱스로 변환
python scripts/build_index.py
```

## 🚀 실행 방법

```bash
# Streamlit 앱 실행
streamlit run app/app.py

# 또는 포트 지정
streamlit run app/app.py --server.port 8080
```

브라우저에서 `http://localhost:8501` 접속

## 🌍 Streamlit Cloud 배포

### 1. GitHub에 코드 업로드
```bash
git add .
git commit -m "Add Freud dream interpretation app"
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

이제 전 세계 어디서든 프로이트 박사의 꿈해몽 상담을 받을 수 있습니다! 🎭

## 📁 프로젝트 구조

```
dream_bot/
├── app/
│   └── app.py              # 🎭 프로이트 박사 메인 앱
├── data/
│   ├── freud_dreams.pdf    # 📚 프로이트 『꿈의 해석』 원문
│   └── who_sleep.pdf       # 📄 WHO 수면 가이드 (참고용)
├── index/                  # 🔍 FAISS 벡터 인덱스 (자동 생성)
│   ├── dream_index.faiss   # 벡터 인덱스
│   ├── chunks.pkl          # 문서 청크 데이터
│   └── config.json         # 인덱스 설정 정보
├── scripts/
│   └── build_index.py      # 🏗️ PDF 인덱싱 스크립트
├── .streamlit/
│   └── secrets.toml        # 🔐 API 키 설정
├── deploy/
│   └── DEPLOYMENT_GUIDE.md # 📖 배포 가이드
└── requirements.txt        # 📦 의존성 목록
```

## 💭 사용법

### 기본 상담 방법
1. **꿈 입력**: 채팅창에 꿈의 내용을 자세히 기술
2. **해석 확인**: 프로이트 박사의 2단계 해석 확인
   - **📘 문헌 기반 분석**: 『꿈의 해석』 원문 인용
   - **🎭 프로이트의 추론적 해석**: 개인적 통찰과 철학적 사색
3. **원문 확인**: "검색된 『꿈의 해석』 원문" 섹션에서 실제 참고 문헌 확인
4. **추가 상담**: 더 자세한 설명이나 다른 해석이 필요하면 대화 계속

### 검색 품질 확인
- **관련도 점수**: 🟢높음(0.7+) / 🟡보통(0.5+) / 🔴낮음(0.5-)
- **검색 품질 요약**: 평균 관련도와 고품질 구절 수 표시
- **원문 미검색 시**: 자동으로 경고 메시지 표시

## ⚡ 성능 및 특징

### RAG 시스템 고도화
- **다중 쿼리 검색**: 영어/한국어/정신분석 용어로 확장 검색
- **품질 필터링**: 유사도 0.5 이하 결과 자동 제거
- **중복 제거**: 동일 청크의 최고 점수만 유지
- **출처 투명성**: 실제 활용된 원문 구절 완전 공개

### 환각 방지 시스템
- **우선순위 강조**: 문헌 내용 우선 활용 지시
- **출처-추론 분리**: 원문 인용과 개인적 해석 명확히 구분
- **검색 실패 대응**: 관련 문헌 부족 시 명시적 경고

## 🔧 고급 설정

### 청크 크기 조정

`scripts/build_index.py`에서:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      # 크기 조정 (기본: 600)
    chunk_overlap=100,   # 중복 조정 (기본: 100)
)
```

### 검색 결과 수 변경

`app/app.py`에서:

```python
relevant_chunks = st.session_state.rag_bot.search_similar_chunks(prompt, k=15)  # k값 조정
```

### 품질 임계값 조정

```python
# 품질 필터링: 유사도가 너무 낮은 것 제거
filtered_results = [r for r in final_results if r['score'] > 0.5]  # 임계값 조정
```

## 📊 성능 벤치마크

| 지표 | 값 |
|------|-----|
| 문서 청크 수 | ~10,500개 (프로이트 원문) |
| 인덱싱 시간 | ~5분 (첫 실행 시) |
| 임베딩 모델 로딩 | ~10초 (첫 실행 시) |
| RAG 검색 | ~1-2초 (다중 쿼리) |
| 응답 생성 | ~3-8초 (GPT-4o API) |
| 메모리 사용 | ~1-2GB (로컬 LLM 불필요) |

## 🐛 문제 해결

### "OpenAI API 키 미설정" 오류

**로컬 실행:**
```bash
# secrets.toml 파일 확인
cat .streamlit/secrets.toml

# API 키 재설정
echo 'OPENAI_API_KEY = "your_actual_api_key"' > .streamlit/secrets.toml
```

**Streamlit Cloud:**
1. 앱 페이지에서 "Settings" 클릭
2. "Secrets" 탭에서 API 키 확인/수정

### "관련 문헌이 검색되지 않았습니다" 경고

1. 꿈 내용을 더 구체적으로 기술
2. 상징적 요소나 감정을 포함하여 재시도
3. 여러 번 시도하여 다른 각도로 검색

### 인덱스 파일 오류

```bash
# 인덱스 재생성
rm -rf index/
python scripts/build_index.py
```

### 세그멘테이션 폴트 (macOS)

sentence-transformers와 PyTorch 호환성 문제:

```bash
# Python 3.11 환경 사용
conda create -n dream_bot python=3.11
conda activate dream_bot
pip install -r requirements.txt
```

## 🔒 보안 가이드

### API 키 보호
- **✅ 안전**: `.streamlit/secrets.toml`에 저장 (`.gitignore`로 보호됨)
- **❌ 위험**: 코드 내에 하드코딩, `.env` 파일을 깃허브에 업로드
- **확인 방법**: `git status`에서 `secrets.toml`이 나타나지 않아야 함

### 깃허브 업로드 전 체크리스트
```bash
# 1. API 키가 보호되는지 확인
git status | grep secrets.toml  # 아무것도 출력되지 않아야 함

# 2. 민감한 파일들이 제외되었는지 확인
cat .gitignore | grep -E "(secrets|\.env|\.key)"

# 3. 깨끗한 상태로 커밋
git add .
git commit -m "Add Freud dream interpretation app"
git push origin main
```

## 📚 프로이트 이론 배경

이 앱에서 사용하는 주요 정신분석 개념들:

- **무의식**: 의식 밖에 있는 정신 영역
- **꿈의 응축**: 여러 생각이 하나의 상징으로 압축
- **꿈의 전치**: 중요한 내용이 사소한 것으로 변장
- **상징화**: 무의식적 욕망의 우회적 표현
- **억압**: 받아들이기 어려운 충동의 무의식 속 저장

## 📜 라이선스 및 저작권

- **코드**: MIT License
- **데이터**:
  - Freud's "The Interpretation of Dreams": Public Domain
  - WHO Sleep Guidelines: CC BY-NC-SA 3.0 IGO
- **AI 모델**: 
  - OpenAI GPT-4o: OpenAI API 이용약관
  - intfloat/multilingual-e5-base: MIT License

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/DreamFeature`)
3. Commit your changes (`git commit -m 'Add dream interpretation feature'`)
4. Push to the branch (`git push origin feature/DreamFeature`)
5. Open a Pull Request

## 📞 문의

질문이나 제안사항이 있으시면 Issues를 통해 알려주세요!

---

Made with 🎭 for exploring the unconscious mind through dreams 