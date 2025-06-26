#!/bin/bash

# 한밤의 꿈해몽 상담가 - 설치 스크립트
# M2 MacBook Air 최적화

echo "🌙 한밤의 꿈해몽 상담가 설치를 시작합니다..."

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python 버전 확인
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo -e "${RED}❌ Python 3.11 이상이 필요합니다. 현재 버전: $python_version${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 버전 확인: $python_version${NC}"

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 가상환경 생성 중...${NC}"
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
echo -e "${YELLOW}📚 패키지 설치 중...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# MLX 모델 다운로드 확인
echo -e "${YELLOW}🤖 Qwen 모델 확인 중...${NC}"
if ! mlx-lm list | grep -q "Qwen2.5-7B-Instruct-4bit"; then
    echo -e "${YELLOW}📥 Qwen 모델 다운로드 (약 4.2GB)...${NC}"
    mlx-lm download mlx-community/Qwen2.5-7B-Instruct-4bit
else
    echo -e "${GREEN}✅ Qwen 모델이 이미 설치되어 있습니다.${NC}"
fi

# 인덱스 생성
if [ ! -d "index" ] || [ ! -f "index/dream_index.faiss" ]; then
    echo -e "${YELLOW}🔨 PDF 인덱싱 중...${NC}"
    python scripts/build_index.py
else
    echo -e "${GREEN}✅ 인덱스가 이미 존재합니다.${NC}"
fi

# 환경 변수 설정
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export MLX_MEMORY_LIMIT_GB=6

echo -e "${GREEN}✨ 설치가 완료되었습니다!${NC}"
echo ""
echo "실행 방법:"
echo "  $ source venv/bin/activate"
echo "  $ streamlit run app/app.py"
echo ""
echo "또는 run.sh 스크립트를 사용하세요:"
echo "  $ ./run.sh" 