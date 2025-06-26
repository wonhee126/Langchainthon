#!/bin/bash

# 한밤의 꿈해몽 상담가 - 실행 스크립트 (OpenAI API 버전)

echo "🌙 한밤의 꿈해몽 상담가를 시작합니다..."

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ 가상환경을 찾을 수 없습니다. 먼저 ./setup.sh를 실행하세요."
    exit 1
fi

# .streamlit/secrets.toml 파일 확인
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "❌ .streamlit/secrets.toml 파일을 찾을 수 없습니다."
    echo "다음 내용으로 .streamlit/secrets.toml 파일을 생성하세요:"
    echo 'OPENAI_API_KEY = "your_openai_api_key_here"'
    exit 1
fi

# OpenAI API 키 확인
if ! grep -q "OPENAI_API_KEY" .streamlit/secrets.toml; then
    echo "❌ secrets.toml 파일에 OPENAI_API_KEY가 설정되지 않았습니다."
    exit 1
fi

echo "✅ OpenAI API 키 설정 확인됨"

# 메모리 상태 확인
echo "💻 현재 메모리 상태:"
python -c "import psutil; m = psutil.virtual_memory(); print(f'  사용 중: {m.percent:.1f}% ({m.used/1024**3:.1f}GB / {m.total/1024**3:.1f}GB)')"

# Streamlit 실행
echo ""
echo "🚀 앱을 시작합니다..."
echo "브라우저에서 http://localhost:8501 로 접속하세요."
echo ""
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

streamlit run app/app.py 