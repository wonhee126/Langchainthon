import json

# JSONL 형식(줄마다 하나의 JSON 오브젝트)일 경우
with open('/Users/woongs/Documents/Langchainthon/data/tifu_all_tokenized_and_filtered.json', 'r') as f:
    data = [json.loads(line) for line in f]

# score 기준으로 내림차순 정렬
sorted_data = sorted(data, key=lambda x: x.get('score', 0), reverse=True)

# 상위 1300개만 추출
top_1300 = sorted_data[:1300]

# 원하는 파일로 저장 (선택사항)
with open('data/tifu_top_1300.json', 'w') as f: # 파일 경로 수정            
    json.dump(top_1300, f, indent=2)