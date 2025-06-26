# 🚀 한밤의 꿈해몽 상담가 - 배포 가이드

팀원들과 공유하기 위한 3가지 배포 방법을 소개합니다.

## 🎯 배포 옵션 요약

| 방법 | 난이도 | 비용 | 안정성 | 추천 상황 |
|------|--------|------|--------|----------|
| 1. ngrok (로컬) | ⭐ | 무료 | 낮음 | 임시 데모, 테스트 |
| 2. Streamlit Cloud + OpenAI | ⭐⭐ | $10-20/월 | 높음 | **프로덕션 추천** |
| 3. 하이브리드 (MLX API + Cloud) | ⭐⭐⭐ | 무료* | 중간 | 비용 절감 필요 시 |

## 📱 Option 1: ngrok으로 즉시 공유 (5분)

### 장점
- ✅ MLX 모델 그대로 사용
- ✅ 추가 비용 없음
- ✅ 5분 안에 공유 가능

### 단점
- ❌ 맥북을 켜놓아야 함
- ❌ 동시 사용자 제한
- ❌ URL이 매번 바뀜

### 실행 방법

```bash
# 1. ngrok 설치
brew install ngrok

# 2. 공유 스크립트 실행
chmod +x deploy/ngrok_share.sh
./deploy/ngrok_share.sh

# 3. 생성된 URL을 팀원들에게 공유
# 예: https://abc123.ngrok.io
```

## ☁️ Option 2: Streamlit Community Cloud (권장)

### 장점
- ✅ 무료 호스팅
- ✅ 안정적인 URL
- ✅ 자동 업데이트

### 단점
- ❌ OpenAI API 키 필요 ($10-20/월)
- ❌ MLX 사용 불가

### 배포 단계

#### 1️⃣ GitHub 저장소 준비

```bash
# Git 초기화
git init
git add .
git commit -m "Initial commit"

# GitHub에 푸시
git remote add origin https://github.com/YOUR_USERNAME/dream-bot.git
git push -u origin main
```

#### 2️⃣ Streamlit Cloud 설정

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. GitHub 계정 연결
3. "New app" 클릭
4. 설정:
   - Repository: `YOUR_USERNAME/dream-bot`
   - Branch: `main`
   - Main file path: `app/app_cloud.py`

#### 3️⃣ Secrets 설정

Streamlit Cloud 앱 설정에서:

```toml
# .streamlit/secrets.toml 형식
OPENAI_API_KEY = "sk-..."
```

#### 4️⃣ requirements 파일 지정

```bash
# Streamlit Cloud는 requirements_cloud.txt 사용
mv requirements_cloud.txt requirements.txt
```

#### 5️⃣ 배포

"Deploy" 버튼 클릭 → 3-5분 후 앱 실행

공유 URL: `https://YOUR_APP_NAME.streamlit.app`

## 🔧 Option 3: 하이브리드 배포 (고급)

MLX를 로컬 API로, 프론트엔드는 클라우드에 배포

### 아키텍처
```
[Streamlit Cloud] ←→ [ngrok] ←→ [로컬 MLX API]
```

### 설정 방법

#### 1️⃣ MLX API 서버 실행

```bash
# FastAPI 설치
pip install fastapi uvicorn

# API 서버 실행
python app/mlx_api_server.py
```

#### 2️⃣ ngrok으로 API 노출

```bash
ngrok http 8000
# https://xyz789.ngrok.io 생성됨
```

#### 3️⃣ 클라우드 앱 수정

`app/app_hybrid.py` 생성:

```python
# MLX API URL을 환경변수로 설정
MLX_API_URL = st.secrets.get("MLX_API_URL")

# OpenAI 대신 MLX API 호출
response = requests.post(
    f"{MLX_API_URL}/chat",
    json={"messages": messages}
)
```

#### 4️⃣ Streamlit Cloud 배포

Secrets에 추가:
```toml
MLX_API_URL = "https://xyz789.ngrok.io"
```

## 📊 비용 비교

| 항목 | ngrok | OpenAI API | 하이브리드 |
|------|-------|------------|------------|
| 호스팅 | $0 | $0 (Streamlit) | $0 |
| LLM | $0 (로컬) | ~$10-20/월 | $0 (로컬) |
| 전기료 | ~$5/월 | $0 | ~$5/월 |
| **총 비용** | **~$5/월** | **~$10-20/월** | **~$5/월** |

## 🎯 추천 시나리오

### 1. 단기 데모/테스트
→ **ngrok 사용**
```bash
./deploy/ngrok_share.sh
```

### 2. 팀 내부 장기 사용
→ **Streamlit Cloud + OpenAI**
- 안정적이고 관리 쉬움
- 비용 대비 가치 높음

### 3. 비용 최적화 필요
→ **하이브리드 방식**
- 설정 복잡하지만 비용 절감
- 맥북 상시 가동 필요

## 🚨 주의사항

1. **보안**
   - API 키는 절대 코드에 하드코딩 금지
   - Secrets 또는 환경변수 사용

2. **성능**
   - 동시 사용자 5명 이상 시 OpenAI API 권장
   - MLX는 단일 사용자에 최적화

3. **가용성**
   - ngrok 무료 버전은 8시간마다 URL 변경
   - 프로덕션은 클라우드 배포 필수

## 📞 문제 해결

### ngrok 연결 안 됨
```bash
# 방화벽 확인
sudo pfctl -d  # macOS 방화벽 임시 해제
```

### Streamlit Cloud 빌드 실패
```bash
# Python 버전 명시
echo "python-3.11" > runtime.txt
```

### OpenAI API 한도 초과
- 사용량 모니터링: https://platform.openai.com/usage
- Rate limit 설정 추가

## 🎉 다음 단계

배포 완료 후:
1. 사용자 피드백 수집
2. 응답 속도 모니터링
3. 비용 추적
4. 기능 개선

---

궁금한 점이 있으시면 언제든 문의해주세요! 🌙 