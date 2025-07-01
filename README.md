# 🤗 Reddit 상담사 챗봇

Reddit 커뮤니티의 TIFU(Today I F***ed Up) 데이터를 활용한 AI 상담사 챗봇입니다.

## 📖 프로젝트 소개

이 프로젝트는 Reddit의 실제 경험담을 학습하여, 사용자의 고민이나 문제 상황에 대해 공감적이고 실용적인 조언을 제공하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.

### 주요 기능
- 🔍 **유사 경험담 검색**: 사용자 상황과 비슷한 Reddit 포스트 검색
- 💬 **공감적 상담**: 따뜻하고 이해심 많은 조언 제공
- 📚 **경험 기반 조언**: 실제 사람들의 경험담을 바탕으로 한 현실적 조언
- 🌟 **희망과 격려**: 문제 해결과 성장을 위한 긍정적 메시지

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Vector DB**: FAISS
- **Embedding**: multilingual-e5-base
- **Data Processing**: JSON
- **Language**: Python 3.11+

## 📂 프로젝트 구조

```
LangchainThon/
├── app/
│   └── app.py              # 메인 RAG 챗봇 앱
├── data/
│   └── tifu_all_tokenized_and_filtered.json  # TIFU 데이터
├── index/
│   ├── reddit_index.faiss  # FAISS 벡터 인덱스
│   ├── chunks.pkl          # 문서 청크 데이터
│   └── config.json         # 인덱스 설정
├── scripts/
│   └── build_index.py      # 인덱스 빌드 스크립트
├── requirements.txt        # 의존성 파일
└── README.md
```

## 🚀 설치 및 실행 (Windows & macOS 공통)

이 프로젝트는 `conda`와 `environment.yml` 파일을 사용하여 개발 환경을 통일합니다. 아래 명령어를 실행하면 필요한 모든 라이브러리가 설치된 `reddit-rag` 가상환경이 자동으로 생성됩니다.

### 1. 저장소 클론

```bash
# 저장소 클론
git clone <repository-url>
cd LangchainThon
```

### 2. Conda 환경 생성 및 의존성 설치

터미널에서 아래 명령어 단 하나만 실행하세요.

```bash
# environment.yml 파일을 사용하여 'reddit-rag' Conda 환경 생성
conda env create -f environment.yml
```

성공적으로 환경이 생성되었다면, 아래 명령어로 활성화합니다.

```bash
# 'reddit-rag' 환경 활성화
conda activate reddit-rag
```

> **💡 `environment.yml`이란?**
> 프로젝트에 필요한 Python 버전, Conda 패키지 (`faiss-cpu` 등), Pip 패키지를 모두 정의해놓은 명세서입니다. 이 파일 덕분에 팀원 모두가 OS에 상관없이 동일한 개발 환경을 명령어 하나로 구축할 수 있습니다.

### 3. 데이터 준비

`data` 폴더에 `tifu_all_tokenized_and_filtered.json` 파일이 있는지 확인해주세요.

### 3. 인덱스 빌드

```bash
# 벡터 인덱스 생성 (최초 1회만 실행)
python scripts/build_index.py
```

### 4. OpenAI API 키 설정

Streamlit secrets 파일 생성:
```bash
mkdir .streamlit
echo 'OPENAI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
```

### 5. 앱 실행

```bash
# Reddit 상담사 챗봇 실행
streamlit run app/app.py
```

## 👥 팀 개발 가이드

### 역할 분담
- **데이터 처리**: JSON 데이터 전처리 및 정제
- **벡터 검색**: FAISS 인덱스 최적화 및 검색 품질 개선
- **프롬프트 엔지니어링**: 상담사 페르소나 및 응답 품질 개선
- **UI/UX**: Streamlit 인터페이스 디자인 및 사용성 개선

### 개발 워크플로우

1. **브랜치 생성**: `feature/기능명` 형태로 브랜치 생성
2. **개발**: 각자 담당 기능 개발
3. **테스트**: 로컬에서 메인 앱 테스트
4. **통합**: 기능 통합 및 테스트
5. **배포**: Streamlit Cloud 배포

### 개발 팁

#### 빠른 테스트
```bash
# 로컬에서 앱 테스트
streamlit run app/app.py
```

#### 인덱스 재빌드
```bash
# 데이터 변경 시 인덱스 재생성
python scripts/build_index.py
```

#### 메모리 최적화
- 배치 처리로 메모리 사용량 제어
- 주기적인 `gc.collect()` 호출
- 큰 파일은 스트리밍 방식으로 처리

## 📊 데이터 형식

### TIFU JSON 형식
```json
{
  "title": "게시물 제목",
  "selftext_without_tldr": "본문 내용",
  "tldr": "요약",
  "score": 점수,
  "num_comments": 댓글수,
  "id": "포스트ID"
}
```

## 🔧 커스터마이징

### 새로운 데이터 소스 추가

1. `scripts/build_index.py`에 로더 함수 추가
2. 데이터 형식에 맞는 전처리 로직 구현
3. 메타데이터 스키마 정의

### 상담사 페르소나 변경

`app/app.py`의 `system_prompt` 수정:
- 상담 스타일 조정
- 응답 구조 변경
- 주의사항 추가

### 검색 품질 개선

- `search_similar_chunks()` 함수의 쿼리 확장 로직 수정
- 유사도 임계값 조정
- 검색 결과 후처리 로직 개선

## 📈 성능 최적화

### 인덱스 최적화
- 청크 크기 조정 (현재: 800자)
- 오버랩 비율 조정 (현재: 100자)
- FAISS 인덱스 타입 변경 고려

### 응답 속도 개선
- 배치 임베딩 처리
- 캐싱 전략 적용
- 비동기 처리 도입

## 🚨 주의사항

- **API 사용량**: OpenAI API 사용량 모니터링 필요
- **메모리 관리**: 대용량 데이터 처리 시 메모리 부족 주의
- **데이터 보안**: 개인정보가 포함된 데이터 처리 시 주의
- **상담 범위**: 전문적인 의료/법률 조언은 제공하지 않음

## 🐛 문제 해결

### 자주 발생하는 문제

1. **ImportError: sentence-transformers**
   - 현재 PyTorch 의존성 문제로 임시 비활성화
   - CPU 환경에서는 설치 후 주석 해제

2. **FAISS 인덱스 로드 실패**
   - `python scripts/build_index.py` 재실행
   - 데이터 파일 경로 확인

3. **OpenAI API 오류**
   - API 키 확인
   - 사용량 한도 확인

## 📝 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여하기

1. 이슈 생성 또는 기능 제안
2. 포크 후 개발
3. 풀 리퀘스트 생성
4. 코드 리뷰 및 병합

---

**팀 프로젝트 진행 상황**
- [ ] 데이터 전처리 완료
- [ ] 벡터 인덱스 최적화
- [ ] 프롬프트 엔지니어링
- [ ] UI/UX 개선
- [ ] 성능 테스트
- [ ] 배포 준비 