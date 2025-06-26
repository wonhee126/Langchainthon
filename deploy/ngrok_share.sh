#!/bin/bash

# ngrok을 사용한 로컬 Streamlit 앱 공유 스크립트

echo "🌐 팀원들과 앱을 공유합니다..."

# ngrok 설치 확인
if ! command -v ngrok &> /dev/null; then
    echo "📦 ngrok을 설치합니다..."
    brew install ngrok
fi

# 가상환경 활성화
source venv/bin/activate

# 환경 변수 설정
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export MLX_MEMORY_LIMIT_GB=6

# Streamlit 실행 (백그라운드)
echo "🚀 Streamlit 앱 시작..."
streamlit run app/app.py --server.headless true --server.port 8501 &
STREAMLIT_PID=$!

# 잠시 대기
sleep 5

# ngrok 터널 생성
echo "🔗 공유 링크 생성 중..."
ngrok http 8501

# 종료 시 Streamlit도 종료
trap "kill $STREAMLIT_PID" EXIT 