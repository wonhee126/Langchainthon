#!/usr/bin/env python3
"""
네트워크 연결 테스트 스크립트
"""

import socket
import requests
import subprocess
import time

def get_local_ip():
    """로컬 IP 주소 확인"""
    try:
        # 외부 서버에 연결해서 로컬 IP 확인
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def test_streamlit():
    """Streamlit 연결 테스트"""
    print("🔍 Streamlit 연결 테스트 중...")
    
    # 로컬 테스트
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        print(f"✅ 로컬 연결 성공: http://localhost:8501")
    except:
        print("❌ 로컬 연결 실패")
        return False
    
    # 외부 IP 테스트
    local_ip = get_local_ip()
    try:
        response = requests.get(f"http://{local_ip}:8501", timeout=5)
        print(f"✅ 외부 연결 성공: http://{local_ip}:8501")
        return True
    except:
        print(f"❌ 외부 연결 실패: http://{local_ip}:8501")
        return False

def check_ngrok():
    """ngrok 상태 확인"""
    print("\n🔍 ngrok 상태 확인...")
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=3)
        tunnels = response.json()
        if tunnels.get('tunnels'):
            for tunnel in tunnels['tunnels']:
                if tunnel['proto'] == 'https':
                    print(f"✅ ngrok 터널: {tunnel['public_url']}")
                    return tunnel['public_url']
        else:
            print("❌ ngrok 터널 없음")
    except:
        print("❌ ngrok API 접속 실패")
    
    return None

if __name__ == "__main__":
    print("🌐 네트워크 연결 테스트")
    print("=" * 40)
    
    # Streamlit 테스트
    streamlit_ok = test_streamlit()
    
    # ngrok 테스트
    ngrok_url = check_ngrok()
    
    print("\n📋 결과 요약:")
    print("-" * 20)
    
    if streamlit_ok:
        local_ip = get_local_ip()
        print(f"🔗 공유 가능 URL: http://{local_ip}:8501")
    
    if ngrok_url:
        print(f"🔗 ngrok URL: {ngrok_url}")
    
    if not streamlit_ok and not ngrok_url:
        print("❌ 모든 연결 실패")
        print("\n💡 해결 방법:")
        print("1. 방화벽 설정 확인")
        print("2. ngrok 계정 생성 후 인증")
        print("3. 다른 포트 사용") 