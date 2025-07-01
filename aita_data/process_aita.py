import sqlite3
import json
from collections import defaultdict

# 데이터베이스 연결
conn = sqlite3.connect('aita_data/AmItheAsshole.sqlite')
# 결과를 딕셔너리 형태로 받기 위해 row_factory 설정
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

chunk_size = 1000
file_counter = 1
offset = 0

while True:
    print(f"Fetching submissions from offset {offset}...")
    # 1. submission을 100개씩 나눠서 불러옵니다.
    cursor.execute(f"""
        SELECT * FROM submission 
        ORDER BY score DESC 
        LIMIT {chunk_size} OFFSET {offset};
    """)
    submissions_chunk = cursor.fetchall()

    if not submissions_chunk:
        print("No more submissions to process.")
        break

    # 2. 한 번의 쿼리로 chunk의 모든 submission에 대한 상위 10개 댓글 가져오기
    submission_ids_in_chunk = [row['submission_id'] for row in submissions_chunk]
    placeholders = ','.join('?' for _ in submission_ids_in_chunk)

    comment_query = f"""
    SELECT * FROM (
        SELECT 
            *,
            SUBSTR(parent_id, INSTR(parent_id, '_') + 1) as parent_submission_id,
            ROW_NUMBER() OVER(PARTITION BY SUBSTR(parent_id, INSTR(parent_id, '_') + 1) ORDER BY score DESC) as rn
        FROM 
            comment
        WHERE 
            SUBSTR(parent_id, INSTR(parent_id, '_') + 1) IN ({placeholders})
    )
    WHERE rn <= 10;
    """
    cursor.execute(comment_query, submission_ids_in_chunk)
    all_comments_for_chunk = cursor.fetchall()

    # 3. 댓글들을 submission ID별로 그룹화합니다.
    comments_by_submission_id = defaultdict(list)
    for comment_row in all_comments_for_chunk:
        comments_by_submission_id[comment_row['parent_submission_id']].append(comment_row)

    # 4. submission과 그룹화된 댓글을 조합하여 최종 결과 생성
    results_chunk = []
    print(f"--> Processing {len(submissions_chunk)} submissions for chunk {file_counter}...")
    for i, sub_row in enumerate(submissions_chunk):
        print(f"  ({offset + i + 1}) Combining submission ID: {sub_row['submission_id']}")
        submission_id_str = sub_row['submission_id']
        comments_for_this_sub = comments_by_submission_id.get(submission_id_str, [])
        
        submission_data = {
            'submission': dict(sub_row),
            'comments': [dict(comment) for comment in comments_for_this_sub]
        }
        results_chunk.append(submission_data)

    # 5. 처리된 chunk를 파일로 저장합니다.
    file_name = f'iffelton/result_{file_counter}.json'
    print(f"--> Saving chunk {file_counter} to {file_name}")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results_chunk, f, ensure_ascii=False, indent=4)
    
    offset += chunk_size
    file_counter += 1

print("\n모든 작업이 완료되었습니다.")

conn.close() 