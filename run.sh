#!/bin/bash

# 한밤의 꿈해몽 상담가 - 실행 스크립트

echo "🌙 한밤의 꿈해몽 상담가를 시작합니다..."

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ 가상환경을 찾을 수 없습니다. 먼저 ./setup.sh를 실행하세요."
    exit 1
fi

# 환경 변수 설정 (M2 Mac 최적화)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export MLX_MEMORY_LIMIT_GB=6

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