#!/bin/bash

# 한밤의 꿈해몽 상담가 - 설치 스크립트
# OpenAI API 버전

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

# .streamlit/secrets.toml 파일 생성 확인
if [ ! -d ".streamlit" ]; then
    mkdir .streamlit
fi

if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "${YELLOW}📝 secrets.toml 파일 생성...${NC}"
    cat > .streamlit/secrets.toml << EOF
# Streamlit Cloud Secrets 설정
# 이 파일은 로컬 테스트용입니다.
# 실제 배포 시에는 Streamlit Cloud 웹사이트에서 설정해야 합니다.

OPENAI_API_KEY = "your_openai_api_key_here"
EOF
    echo -e "${YELLOW}⚠️  .streamlit/secrets.toml 파일에 실제 OpenAI API 키를 설정해주세요!${NC}"
else
    echo -e "${GREEN}✅ secrets.toml 파일이 이미 존재합니다.${NC}"
fi

# 인덱스 생성
if [ ! -d "index" ] || [ ! -f "index/dream_index.faiss" ]; then
    echo -e "${YELLOW}🔨 PDF 인덱싱 중...${NC}"
    python scripts/build_index.py
else
    echo -e "${GREEN}✅ 인덱스가 이미 존재합니다.${NC}"
fi

echo -e "${GREEN}✨ 설치가 완료되었습니다!${NC}"
echo ""
echo -e "${YELLOW}🔑 중요: OpenAI API 키를 설정하세요:${NC}"
echo "  1. https://platform.openai.com/api-keys 에서 API 키 생성"
echo "  2. .streamlit/secrets.toml 파일에서 'your_openai_api_key_here'를 실제 키로 교체"
echo ""
echo -e "${YELLOW}🚀 Streamlit Cloud 배포:${NC}"
echo "  1. GitHub에 코드 푸시"
echo "  2. https://share.streamlit.io 에서 앱 생성"
echo "  3. Settings → Secrets에서 OPENAI_API_KEY 설정"
echo ""
echo "실행 방법:"
echo "  $ source venv/bin/activate"
echo "  $ streamlit run app/app.py"
echo ""
echo "또는 run.sh 스크립트를 사용하세요:"
echo "  $ ./run.sh" 