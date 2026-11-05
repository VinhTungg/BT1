import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== TẠO BỘ DỮ LIỆU 5-FOLD GROUPKFOLD (V3) ===")
    
    train_path = "E:/BT1/data/clean/train.csv"
    val_path = "E:/BT1/data/clean/val.csv"
    test_path = "E:/BT1/data/clean/test.csv"
    
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_val = pd.read_csv(val_path, encoding="utf-8")
    df_test = pd.read_csv(test_path, encoding="utf-8")
    
    # Gộp train + val thành pool 7.669 câu (95 bài)
    df_pool = pd.concat([df_train, df_val], ignore_index=True)
    
    print(f"Tổng số mẫu pool (Train + Val): {len(df_pool)}")
    print(f"Số bài viết unique (post_id): {df_pool['post_id'].nunique()}")
    print(f"Số mẫu Toxic: {(df_pool['label'] == 1).sum()} ({(df_pool['label'] == 1).mean()*100:.2f}%)")
    
    # Chia 5-Fold StratifiedGroupKFold theo post_id và label
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df_pool["fold"] = -1
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_pool, df_pool["label"], df_pool["post_id"])):
        df_pool.loc[val_idx, "fold"] = fold
        
    # Thống kê từng fold
    fold_stats = {}
    for f in range(5):
        sub = df_pool[df_pool["fold"] == f]
        n_samples = len(sub)
        n_posts = sub["post_id"].nunique()
        n_toxic = int((sub["label"] == 1).sum())
        n_nontoxic = int((sub["label"] == 0).sum())
        toxic_pct = round(n_toxic / n_samples * 100, 2)
        
        fold_stats[f"fold_{f}"] = {
            "samples": n_samples,
            "posts": n_posts,
            "non_toxic": n_nontoxic,
            "toxic": n_toxic,
            "toxic_pct": toxic_pct
        }
        print(f" - Fold {f}: {n_samples} câu | {n_posts} bài | {n_toxic} toxic ({toxic_pct}%)")
        
    # Kiểm tra chống leakage
    post_sets = [set(df_pool[df_pool["fold"] == f]["post_id"]) for f in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            overlap = post_sets[i].intersection(post_sets[j])
            assert len(overlap) == 0, f"Rò rỉ post_id giữa Fold {i} và Fold {j}: {overlap}"
            
    test_posts = set(df_test["post_id"])
    pool_posts = set(df_pool["post_id"])
    assert len(test_posts.intersection(pool_posts)) == 0, "Rò rỉ post_id với tập Test!"
    print("✅ Kiểm tra rò rỉ (Data Leakage): 100% PASS (Không trùng post_id giữa các Fold và với Test)!")
    
    # Lưu ra các thư mục
    out_dirs = [
        "E:/BT1/data/kfold_v3",
        "E:/BT1/colab_phobert/data",
        "E:/BT1/colab_phobert_kfold/data"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        out_csv = os.path.join(d, "train_pool_5folds.csv")
        df_pool.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Đã lưu: {out_csv}")
        
    manifest = {
        "dataset_version": "kfold_v3",
        "strategy": "5-Fold StratifiedGroupKFold theo post_id",
        "random_seed": 42,
        "total_pool_samples": len(df_pool),
        "total_posts": df_pool["post_id"].nunique(),
        "total_toxic": int((df_pool["label"] == 1).sum()),
        "test_samples_reserved": len(df_test),
        "test_posts_reserved": df_test["post_id"].nunique(),
        "fold_statistics": fold_stats,
        "pool_csv_sha256": compute_sha256("E:/BT1/data/kfold_v3/train_pool_5folds.csv")
    }
    
    manifest_path = "E:/BT1/data/kfold_v3/manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Đã tạo manifest: {manifest_path}")

if __name__ == "__main__":
    main()

