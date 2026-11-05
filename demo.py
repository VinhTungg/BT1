# -*- coding: utf-8 -*-
"""Bản Demo Phân Loại Bình Luận Toxic / Non-toxic — Đề tài BTL 1c (PTIT)

Hỗ trợ 3 chế độ sử dụng:
1. Chế độ Tương tác Terminal:
       python demo.py
2. Chế độ Suy luận nhanh từ dòng lệnh:
       python demo.py "Thằng ngu này cút đi"
3. Chế độ Giao diện Web trực quan:
       python demo.py --web
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text, preprocess_for_phobert, segment_vietnamese

# Cấu hình mặc định
DEFAULT_PORT = 7860
PHOBERT_DIR = ROOT / "models" / "phobert_best"
SVM_PATH = ROOT / "models" / "resplit_v2" / "improvements" / "word_char_svm.joblib"
PHOBERT_THRESHOLD = 0.52

SAMPLE_COMMENTS = [
    ("Tích cực / Khen ngợi", "Chúc các bạn sinh viên PTIT thi tốt và đạt kết quả cao trong kỳ này nhé!"),
    ("Bình thường / Hỏi đáp", "Cho mình hỏi thủ tục xin giấy xác nhận sinh viên tại phòng một cửa với ạ."),
    ("Than phiền (Non-toxic)", "Học phí kỳ này lại tăng rồi, chán chả buồn nói nữa."),
    ("Xúc phạm nặng (Toxic)", "Thằng ngu học này cút mẹ mày đi, sủa ít thôi."),
    ("Lừa đảo / Chửi bới (Toxic)", "Con chó này làm ăn như cc, lừa đảo tiền sinh viên à?"),
    ("Tiêu cực dịch vụ (Non-toxic)", "Cổng đăng ký tín chỉ lại sập, làm ăn chán đời thực sự.")
]


class ModelManager:
    """Quản lý việc nạp và suy luận của cả PhoBERT và SVM."""

    def __init__(self):
        self.phobert_model = None
        self.phobert_tokenizer = None
        self.svm_model = None
        self.is_loaded = False

    def load_models(self):
        if self.is_loaded:
            return

        print("🔄 Đang khởi tạo mô hình...")

        # 1. Nạp SVM
        if SVM_PATH.exists():
            t0 = time.time()
            self.svm_model = joblib.load(SVM_PATH)
            print(f"  ✅ Đã nạp Linear SVM (Word+Char) [{(time.time()-t0)*1000:.1f}ms]")
        else:
            print(f"  ⚠️ Không tìm thấy file SVM tại {SVM_PATH}")

        # 2. Nạp PhoBERT
        if PHOBERT_DIR.exists():
            t0 = time.time()
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(str(PHOBERT_DIR), use_fast=False)
            self.phobert_model = AutoModelForSequenceClassification.from_pretrained(str(PHOBERT_DIR))
            self.phobert_model.eval()
            print(f"  ✅ Đã nạp PhoBERT v2 Checkpoint [{(time.time()-t0)*1000:.1f}ms]")
        else:
            print(f"  ⚠️ Không tìm thấy PhoBERT tại {PHOBERT_DIR}")

        self.is_loaded = True
        print("🚀 Hệ thống đã sẵn sàng phục vụ suy luận!\n")

    def predict(self, raw_text: str) -> dict:
        """Thực hiện suy luận trên cả 2 mô hình và trả về kết quả chi tiết."""
        if not self.is_loaded:
            self.load_models()

        raw_text = raw_text.strip()
        if not raw_text:
            return {
                "text": "",
                "error": "Vui lòng nhập văn bản cần kiểm tra."
            }

        cleaned = clean_text(raw_text)
        tokenized = segment_vietnamese(cleaned)
        phobert_input = preprocess_for_phobert(raw_text)

        result = {
            "text": raw_text,
            "text_clean": cleaned,
            "text_tokenized": tokenized,
            "models": {}
        }

        # 1. Dự đoán với PhoBERT v2
        if self.phobert_model is not None:
            t0 = time.time()
            inputs = self.phobert_tokenizer(
                phobert_input, return_tensors="pt", max_length=128, padding=True, truncation=True
            )
            with torch.no_grad():
                out = self.phobert_model(**inputs)
                probs = torch.softmax(out.logits, dim=1)[0].numpy()
                prob_toxic = float(probs[1])
                pred = 1 if prob_toxic >= PHOBERT_THRESHOLD else 0
            t_ms = (time.time() - t0) * 1000

            result["models"]["phobert"] = {
                "name": "PhoBERT v2 (Deep Learning - Người 3)",
                "label": "TOXIC" if pred == 1 else "NON-TOXIC",
                "is_toxic": bool(pred == 1),
                "probability": prob_toxic,
                "confidence_pct": round(prob_toxic * 100 if pred == 1 else (1 - prob_toxic) * 100, 1),
                "threshold": PHOBERT_THRESHOLD,
                "latency_ms": round(t_ms, 2)
            }

        # 2. Dự đoán với Linear SVM
        if self.svm_model is not None:
            t0 = time.time()
            df_feat = pd.DataFrame([{"text_clean": cleaned, "text_tokenized": tokenized}])
            pred = int(self.svm_model.predict(df_feat)[0])
            margin = float(self.svm_model.decision_function(df_feat)[0])
            prob_toxic = float(1.0 / (1.0 + np.exp(-margin)))
            t_ms = (time.time() - t0) * 1000

            result["models"]["svm"] = {
                "name": "Linear SVM Word+Char (Machine Learning - Người 2)",
                "label": "TOXIC" if pred == 1 else "NON-TOXIC",
                "is_toxic": bool(pred == 1),
                "probability": prob_toxic,
                "confidence_pct": round(prob_toxic * 100 if pred == 1 else (1 - prob_toxic) * 100, 1),
                "latency_ms": round(t_ms, 2)
            }

        return result


manager = ModelManager()


# ---------------------------------------------------------------------------
# Chế độ 1: Interactive Terminal CLI
# ---------------------------------------------------------------------------
def print_cli_banner():
    print("=" * 78)
    print("       🎓 HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG (PTIT)")
    print("   🔴 BẢN DEMO NHẬN DIỆN BÌNH LUẬN TOXIC / NON-TOXIC TRÊN FACEBOOK")
    print("       Nhóm sinh viên PTIT — Đề tài BTL 1c")
    print("=" * 78)


def render_cli_result(res: dict):
    if "error" in res:
        print(f"❌ {res['error']}")
        return

    print("\n" + "─" * 78)
    print(f"💬 BÌNH LUẬN: \"{res['text']}\"")
    print("─" * 78)

    for key in ("phobert", "svm"):
        if key not in res["models"]:
            continue
        m = res["models"][key]
        tag = "🔴 TOXIC (ĐỘC HẠI)" if m["is_toxic"] else "🟢 NON-TOXIC (AN TOÀN)"
        bar_len = 24
        filled = int(m["probability"] * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"📌 [{m['name']}]")
        print(f"   Kết luận:       {tag}")
        print(f"   Xác suất Toxic: [{bar}] {m['probability']*100:5.1f}% (Độ tin cậy: {m['confidence_pct']}%)")
        print(f"   Thời gian xử lý: {m['latency_ms']} ms")
        print()


def run_cli_interactive():
    print_cli_banner()
    manager.load_models()

    print("💡 Hướng dẫn sử dụng:")
    print("  - Gõ một bình luận tiếng Việt bất kỳ và nhấn Enter để kiểm tra.")
    print("  - Nhập số 1 đến 6 để dùng thử nhanh các câu mẫu kiểm thử.")
    print("  - Gõ 'q' hoặc 'exit' để thoát chương trình.\n")

    print("📋 Danh sách câu mẫu nhanh:")
    for idx, (cat, s) in enumerate(SAMPLE_COMMENTS, 1):
        print(f"   [{idx}] ({cat}): \"{s}\"")
    print()

    while True:
        try:
            user_input = input("👉 Nhập câu cần kiểm tra: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "exit", "quit"):
            print("👋 Đã thoát chương trình demo.")
            break

        # Nếu người dùng nhập số 1-6
        if user_input.isdigit() and 1 <= int(user_input) <= len(SAMPLE_COMMENTS):
            user_input = SAMPLE_COMMENTS[int(user_input) - 1][1]
            print(f"   (Đã chọn mẫu: \"{user_input}\")")

        res = manager.predict(user_input)
        render_cli_result(res)


# ---------------------------------------------------------------------------
# Chế độ 2: Web Server GUI (HTML/CSS hiện đại, không cần cài đặt thêm)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PTIT Toxic Comment Classifier — Demo</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #c0262b;
            --primary-hover: #9e1c20;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --toxic: #ef4444;
            --toxic-bg: rgba(239, 68, 68, 0.15);
            --nontoxic: #10b981;
            --nontoxic-bg: rgba(16, 185, 129, 0.15);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body {
            background: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 16px;
        }
        .container {
            width: 100%;
            max-width: 900px;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        .badge-ptit {
            display: inline-block;
            background: rgba(192, 38, 43, 0.2);
            color: #ff6b6b;
            border: 1px solid rgba(192, 38, 43, 0.4);
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
        }
        h1 {
            font-size: 28px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 8px;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 15px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        textarea {
            width: 100%;
            height: 100px;
            background: #0f172a;
            border: 1.5px solid var(--card-border);
            border-radius: 12px;
            padding: 14px 16px;
            color: #fff;
            font-size: 15px;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus {
            border-color: var(--primary);
        }
        .btn-predict {
            background: linear-gradient(135deg, #c0262b, #e11d48);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 14px 24px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-predict:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }
        .btn-predict:active {
            transform: translateY(0);
        }
        .samples-section {
            margin-top: 14px;
        }
        .samples-title {
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 500;
            margin-bottom: 8px;
        }
        .chips-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .chip {
            background: #0f172a;
            border: 1px solid var(--card-border);
            color: #cbd5e1;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .chip:hover {
            border-color: #64748b;
            color: #fff;
            background: #1e293b;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            margin-top: 10px;
        }
        .model-card {
            background: #0f172a;
            border: 1.5px solid var(--card-border);
            border-radius: 14px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #1e293b;
        }
        .model-title {
            font-size: 15px;
            font-weight: 600;
            color: #e2e8f0;
        }
        .model-tag {
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 6px;
            background: #334155;
            color: #94a3b8;
        }
        .outcome-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 18px;
            margin-bottom: 16px;
        }
        .outcome-toxic {
            background: var(--toxic-bg);
            color: var(--toxic);
            border: 1.5px solid rgba(239, 68, 68, 0.4);
        }
        .outcome-nontoxic {
            background: var(--nontoxic-bg);
            color: var(--nontoxic);
            border: 1.5px solid rgba(16, 185, 129, 0.4);
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .stat-val {
            color: #fff;
            font-weight: 600;
        }
        .prob-bar-container {
            width: 100%;
            height: 10px;
            background: #1e293b;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .prob-bar {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }
        .prob-bar-toxic { background: linear-gradient(90deg, #f97316, #ef4444); }
        .prob-bar-nontoxic { background: linear-gradient(90deg, #10b981, #059669); }
        .empty-placeholder {
            text-align: center;
            color: var(--text-muted);
            padding: 30px 10px;
            font-size: 15px;
        }
        footer {
            margin-top: auto;
            text-align: center;
            color: #64748b;
            font-size: 13px;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge-ptit">BÀI TẬP LỚN 1C — NHÓM SINH VIÊN PTIT</div>
            <h1>Nhận Diện Bình Luận Toxic / Non-toxic</h1>
            <p class="subtitle">Đối đầu trực tiếp: Deep Learning (PhoBERT v2) vs Machine Learning (Linear SVM)</p>
        </header>

        <div class="card">
            <div class="input-group">
                <textarea id="commentInput" placeholder="Nhập bình luận tiếng Việt cần phân loại (ví dụ: than phiền học phí, hỏi han, chửi bới, xúc phạm...)..."></textarea>
                <button class="btn-predict" onclick="doPredict()">
                    <span>🔍</span> Phân Loại Ngay
                </button>
            </div>

            <div class="samples-section">
                <div class="samples-title">Hoặc chọn nhanh câu mẫu để thử:</div>
                <div class="chips-container" id="chipsContainer"></div>
            </div>
        </div>

        <div class="card" id="resultCard" style="display: none;">
            <h2 style="font-size: 18px; margin-bottom: 16px; color: #fff;">📊 Kết Quả Đánh Giá Đối Đầu</h2>
            <div class="results-grid" id="resultsGrid"></div>
        </div>

        <footer>
            Học viện Công nghệ Bưu chính Viễn thông (PTIT) • Học máy & Xử lý ngôn ngữ tự nhiên
        </footer>
    </div>

    <script>
        const samples = [
            "Chúc các bạn sinh viên PTIT thi tốt và đạt điểm cao nhé!",
            "Cho mình hỏi thủ tục xin giấy xác nhận sinh viên với ạ.",
            "Học phí kỳ này lại tăng rồi, chán chả buồn nói nữa.",
            "Thằng ngu học này cút mẹ mày đi, sủa ít thôi.",
            "Con chó này làm ăn như cc, lừa đảo sinh viên à?",
            "Cổng đăng ký tín chỉ lại sập, làm ăn chán đời thực sự."
        ];

        const container = document.getElementById('chipsContainer');
        samples.forEach(s => {
            const chip = document.createElement('div');
            chip.className = 'chip';
            chip.innerText = s.length > 40 ? s.substring(0, 40) + '...' : s;
            chip.title = s;
            chip.onclick = () => {
                document.getElementById('commentInput').value = s;
                doPredict();
            };
            container.appendChild(chip);
        });

        async function doPredict() {
            const text = document.getElementById('commentInput').value.trim();
            if (!text) {
                alert('Vui lòng nhập nội dung bình luận!');
                return;
            }

            const btn = document.querySelector('.btn-predict');
            btn.innerHTML = '<span>⏳</span> Đang phân loại...';
            btn.disabled = true;

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                alert('Lỗi kết nối máy chủ: ' + err);
            } finally {
                btn.innerHTML = '<span>🔍</span> Phân Loại Ngay';
                btn.disabled = false;
            }
        }

        function renderResults(data) {
            const card = document.getElementById('resultCard');
            const grid = document.getElementById('resultsGrid');
            grid.innerHTML = '';

            const models = [
                { key: 'phobert', badge: 'Mô hình Người 3' },
                { key: 'svm', badge: 'Mô hình Người 2' }
            ];

            models.forEach(m => {
                if (!data.models || !data.models[m.key]) return;
                const info = data.models[m.key];
                const isToxic = info.is_toxic;
                const probPct = (info.probability * 100).toFixed(1);

                const div = document.createElement('div');
                div.className = 'model-card';
                div.innerHTML = `
                    <div>
                        <div class="model-header">
                            <span class="model-title">${info.name.split('(')[0]}</span>
                            <span class="model-tag">${m.badge}</span>
                        </div>
                        <div class="outcome-badge ${isToxic ? 'outcome-toxic' : 'outcome-nontoxic'}">
                            <span>${isToxic ? '🔴' : '🟢'}</span>
                            <span>${isToxic ? 'TOXIC (Độc hại)' : 'NON-TOXIC (An toàn)'}</span>
                        </div>
                        <div class="stat-row">
                            <span>Xác suất Toxic:</span>
                            <span class="stat-val">${probPct}%</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar ${isToxic ? 'prob-bar-toxic' : 'prob-bar-nontoxic'}" style="width: ${probPct}%;"></div>
                        </div>
                    </div>
                    <div>
                        <div class="stat-row" style="margin-bottom: 4px;">
                            <span>Độ tin cậy dự đoán:</span>
                            <span class="stat-val" style="color: ${isToxic ? '#ef4444' : '#10b981'};">${info.confidence_pct}%</span>
                        </div>
                        <div class="stat-row" style="margin-bottom: 0;">
                            <span>Thời gian suy luận:</span>
                            <span class="stat-val">${info.latency_ms} ms</span>
                        </div>
                    </div>
                `;
                grid.appendChild(div);
            });

            card.style.display = 'block';
            card.scrollIntoView({ behavior: 'smooth' });
        }

        document.getElementById('commentInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                doPredict();
            }
        });
    </script>
</body>
</html>
"""


class DemoRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                text = data.get("text", "")
                result = manager.predict(text)
                res_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Giảm log ồn ào ở console
        pass


def run_web_demo(port: int = DEFAULT_PORT):
    manager.load_models()
    server_address = ("", port)
    
    # Cho phép tái sử dụng địa chỉ nếu port vừa giải phóng
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, DemoRequestHandler) as httpd:
        url = f"http://localhost:{port}"
        print("=" * 78)
        print(f"🌐 ĐÃ KHỞI CHẠY GIAO DIỆN WEB DEMO TẠI: {url}")
        print("💡 Hãy mở trình duyệt và truy cập địa chỉ trên để thử nghiệm.")
        print("   Nhấn Ctrl + C để dừng máy chủ Web.")
        print("=" * 78)

        # Tự động mở trình duyệt sau 0.8s
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Đã dừng máy chủ Web Demo.")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bản Demo Phân loại Toxic Comment (PhoBERT & SVM).")
    parser.add_argument("text", nargs="?", default=None, help="Câu cần kiểm tra trực tiếp (nếu có).")
    parser.add_argument("--web", action="store_true", help="Khởi chạy giao diện Web trực quan trong trình duyệt.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Cổng chạy Web Demo (mặc định 7860).")
    args = parser.parse_args()

    if args.web:
        run_web_demo(port=args.port)
    elif args.text:
        manager.load_models()
        res = manager.predict(args.text)
        render_cli_result(res)
    else:
        run_cli_interactive()


if __name__ == "__main__":
    main()
