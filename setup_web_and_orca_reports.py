import os
import shutil

orca_dir = '/home/youngchan/lerobot/orca_reports'
web_dir = '/home/youngchan/lerobot/web_reports'

os.makedirs(orca_dir, exist_ok=True)
os.makedirs(web_dir, exist_ok=True)

# 1. Copy existing local orca html files to orca_reports/
files_to_copy = [
    'orca_terminal_1_teleop_softgripper.html',
    'orca_terminal_2_qwen_3d_pick_place.html',
    'orca_terminal_3_gemini_dualcam_pick_place.html',
    'orca_terminal_4_gemini_dualcam_korean_5step.html',
    'orca_terminals_index.html',
    'orca_master_report.html'
]

for f in files_to_copy:
    src = os.path.join('/home/youngchan/lerobot', f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(orca_dir, f))

print("Orca local reports saved in:", orca_dir)

# 2. Create web-compatible HTML files in web_reports/ with relative HTTP links

# Web Index HTML
web_index = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>orca Claude Code 4 Terminal Reports (Web HTTP Server)</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-orange: #fb923c;
            --accent-purple: #c084fc;
            --accent-teal: #2dd4bf;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 3rem; border-bottom: 2px solid var(--card-border); padding-bottom: 2rem; }
        h1 { font-size: 2.5rem; color: #fff; margin-bottom: 0.5rem; }
        p.subtitle { color: var(--text-muted); font-size: 1.1rem; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            transition: transform 0.2s, border-color 0.2s;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .card:hover { transform: translateY(-4px); border-color: var(--accent-blue); }
        .card-header { margin-bottom: 1rem; }
        .card-tag {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            display: inline-block;
            margin-bottom: 0.75rem;
        }
        .tag-t1 { background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
        .tag-t2 { background: rgba(251, 146, 60, 0.2); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
        .tag-t3 { background: rgba(192, 132, 252, 0.2); color: var(--accent-purple); border: 1px solid var(--accent-purple); }
        .tag-t4 { background: rgba(45, 212, 191, 0.2); color: var(--accent-teal); border: 1px solid var(--accent-teal); }
        
        .card-title { font-size: 1.25rem; color: #fff; margin-bottom: 0.5rem; }
        .card-desc { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem; line-height: 1.5; }
        .btn {
            display: inline-block;
            width: 100%;
            text-align: center;
            background: #2563eb;
            color: #fff;
            padding: 0.75rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: background 0.2s;
        }
        .btn:hover { background: #1d4ed8; }
        .master-banner {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            border: 1px solid #6366f1;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .master-banner h3 { color: #fff; margin-bottom: 0.25rem; font-size: 1.2rem; }
        .master-banner p { color: #c7d2fe; font-size: 0.95rem; }
        .btn-master {
            background: #4f46e5;
            color: #fff;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 700;
        }
        .btn-master:hover { background: #4338ca; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌐 인터넷/웹 서버 표준 보고서 포털</h1>
            <p class="subtitle">표준 웹 브라우저(HTTP)에서 동작하는 4개 터미널 세션 개별 및 통합 보고서</p>
        </header>

        <div class="master-banner">
            <div>
                <h3>💡 단일 화면 탭 전환 마스터 페이지</h3>
                <p>페이지 이동 없이 탭 클릭만으로 모든 보고서와 실행 코드를 전환할 수 있습니다.</p>
            </div>
            <a href="./master.html" class="btn-master">마스터 탭 보고서 열기 &rarr;</a>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-tag tag-t1">Terminal 1</span>
                    <h2 class="card-title">SO-101 Teleop & Soft Gripper</h2>
                    <p class="card-desc">손목 90도 각도 보정, 소프트 그리퍼 파지 수동 보정 및 Joint Lag 충돌 안전 복귀 구현</p>
                </div>
                <a href="./terminal_1.html" class="btn">1번 보고서 보기 &rarr;</a>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-tag tag-t2">Terminal 2</span>
                    <h2 class="card-title">Qwen2-VL 3D Perception</h2>
                    <p class="card-desc">Qwen2-VL 2D 감지 → Astra S Depth 3D 좌표 변환 연동 및 5단계 자율 Pick-and-Place 구현</p>
                </div>
                <a href="./terminal_2.html" class="btn">2번 보고서 보기 &rarr;</a>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-tag tag-t3">Terminal 3</span>
                    <h2 class="card-title">Gemini VLM & Dual-Cam Baseline</h2>
                    <p class="card-desc">Gemini VLM + YOLO 듀얼 카메라 실시간 스트리밍, 대화형 클릭 UI, Z축 (+5cm) 오프셋 미세조정</p>
                </div>
                <a href="./terminal_3.html" class="btn">3번 보고서 보기 &rarr;</a>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-tag tag-t4">Terminal 4</span>
                    <h2 class="card-title">Gemini Panel Guard & 5-Step Korean</h2>
                    <p class="card-desc">우측 패널(뎁스/손목) 클릭 오작동 차단 가드 구현, 5단계 정밀 제어 루프 및 한국어 가이드 전환</p>
                </div>
                <a href="./terminal_4.html" class="btn">4번 보고서 보기 &rarr;</a>
            </div>
        </div>
    </div>
</body>
</html>"""

with open(os.path.join(web_dir, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(web_index)

# Helper function to modify hrefs to relative HTTP links for web server
def make_web_relative(src_name, target_name):
    src_path = os.path.join(orca_dir, src_name)
    if os.path.exists(src_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Add home button back to web index
        home_nav = '<div style="margin-bottom: 1rem;"><a href="./index.html" style="color:#38bdf8; text-decoration:none; font-weight:bold;">&larr; 웹 포털 메인으로 돌아가기</a></div>'
        content = content.replace('<body>', '<body>\n    <div class="container">' + home_nav + '</div>')
        with open(os.path.join(web_dir, target_name), 'w', encoding='utf-8') as f:
            f.write(content)

make_web_relative('orca_terminal_1_teleop_softgripper.html', 'terminal_1.html')
make_web_relative('orca_terminal_2_qwen_3d_pick_place.html', 'terminal_2.html')
make_web_relative('orca_terminal_3_gemini_dualcam_pick_place.html', 'terminal_3.html')
make_web_relative('orca_terminal_4_gemini_dualcam_korean_5step.html', 'terminal_4.html')
make_web_relative('orca_master_report.html', 'master.html')

print("Web HTTP reports successfully generated in:", web_dir)
