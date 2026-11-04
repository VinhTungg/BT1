# -*- coding: utf-8 -*-
"""Tạo và đồng bộ file Jupyter Notebook huấn luyện PhoBERT cho Colab và thư mục cục bộ.

Chạy: python scripts/generate_notebook.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    source_nb = REPO_ROOT / "notebooks" / "train_phobert.ipynb"
    if not source_nb.exists():
        raise FileNotFoundError(f"Không tìm thấy notebook nguồn: {source_nb}")

    with open(source_nb, "r", encoding="utf-8") as f:
        notebook_content = json.load(f)

    targets = [
        REPO_ROOT / "colab_phobert" / "train_phobert.ipynb",
        REPO_ROOT / "notebooks" / "train_phobert.ipynb",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(notebook_content, f, indent=1, ensure_ascii=False)
        print(f"Wrote synchronized notebook to {target}")


if __name__ == "__main__":
    main()
