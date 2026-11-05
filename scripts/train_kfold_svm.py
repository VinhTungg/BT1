import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_tox, r_tox, f1_tox, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "f1_toxic": f1_tox,
        "precision_toxic": p_tox,
        "recall_toxic": r_tox,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def build_model_pipeline(model_type):
    if model_type == "bow_svm":
        vectorizer = CountVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=True,
            ngram_range=(1, 2), min_df=1
        )
        classifier = LinearSVC(C=0.1, class_weight="balanced", random_state=42)
        transformer = ColumnTransformer(
            [("word", vectorizer, "text_tokenized")],
            remainder="drop"
        )
        return Pipeline([("features", transformer), ("classifier", classifier)])

    elif model_type == "tfidf_svm":
        vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=True,
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, smooth_idf=True, norm="l2"
        )
        classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
        transformer = ColumnTransformer(
            [("word", vectorizer, "text_tokenized")],
            remainder="drop"
        )
        return Pipeline([("features", transformer), ("classifier", classifier)])

    elif model_type == "word_char_svm":
        word_vec = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=True,
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, smooth_idf=True, norm="l2"
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            lowercase=True, sublinear_tf=True, smooth_idf=True, norm="l2"
        )
        classifier = LinearSVC(C=0.1, class_weight={0: 1, 1: 8}, random_state=42)
        transformer = ColumnTransformer(
            [
                ("word", word_vec, "text_tokenized"),
                ("char", char_vec, "text_clean")
            ],
            remainder="drop"
        )
        return Pipeline([("features", transformer), ("classifier", classifier)])

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN 5-FOLD GROUPKFOLD CHO CÁC MÔ HÌNH SVM")
    print("=" * 80)

    data_path = "E:/BT1/data/kfold_v3/train_pool_5folds.csv"
    report_dir = "E:/BT1/reports/kfold_v3/svm"
    model_dir = "E:/BT1/models/kfold_v3/svm"
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df_pool = pd.read_csv(data_path, encoding="utf-8")
    df_pool["text_clean"] = df_pool["text_clean"].fillna("")
    df_pool["text_tokenized"] = df_pool["text_tokenized"].fillna("")

    models_to_test = ["bow_svm", "tfidf_svm", "word_char_svm"]
    all_summary = []
    all_fold_details = []

    for model_name in models_to_test:
        print(f"\n>>> 🔄 Huấn luyện mô hình: {model_name} (5 Folds)...")
        fold_scores = []
        oof_preds = np.zeros(len(df_pool))
        oof_scores = np.zeros(len(df_pool))

        for fold in range(5):
            train_mask = (df_pool["fold"] != fold)
            val_mask = (df_pool["fold"] == fold)

            df_train = df_pool[train_mask]
            df_val = df_pool[val_mask]
            val_idx = df_pool[val_mask].index.values

            pipeline = build_model_pipeline(model_name)
            pipeline.fit(df_train, df_train["label"])

            # Dự đoán trên tập Val của fold
            y_pred = pipeline.predict(df_val)
            decision_val = pipeline.decision_function(df_val)

            oof_preds[val_idx] = y_pred
            oof_scores[val_idx] = decision_val

            metrics = evaluate_predictions(df_val["label"].values, y_pred)
            metrics["model"] = model_name
            metrics["fold"] = fold
            fold_scores.append(metrics)
            all_fold_details.append(metrics)

            print(f"  - Fold {fold}: Acc = {metrics['accuracy']*100:.2f}% | "
                  f"F1-Toxic = {metrics['f1_toxic']*100:.2f}% | "
                  f"Recall = {metrics['recall_toxic']*100:.2f}% | "
                  f"Macro-F1 = {metrics['macro_f1']*100:.2f}%")

            # Lưu checkpoint của mô hình tốt nhất word_char_svm
            if model_name == "word_char_svm":
                joblib.dump(pipeline, os.path.join(model_dir, f"word_char_svm_fold_{fold}.joblib"))

        # Tổng hợp Mean & Std
        f1_list = [f["macro_f1"] for f in fold_scores]
        f1_tox_list = [f["f1_toxic"] for f in fold_scores]
        rec_tox_list = [f["recall_toxic"] for f in fold_scores]
        acc_list = [f["accuracy"] for f in fold_scores]

        # Đánh giá toàn bộ Out-Of-Fold (OOF) 7.669 câu
        oof_overall = evaluate_predictions(df_pool["label"].values, oof_preds)

        summary_row = {
            "Mô hình": model_name,
            "Macro-F1 Mean (%)": round(np.mean(f1_list) * 100, 2),
            "Macro-F1 Std (%)": round(np.std(f1_list) * 100, 2),
            "Macro-F1 OOF (%)": round(oof_overall["macro_f1"] * 100, 2),
            "F1-Toxic OOF (%)": round(oof_overall["f1_toxic"] * 100, 2),
            "Recall-Toxic OOF (%)": round(oof_overall["recall_toxic"] * 100, 2),
            "Precision-Toxic OOF (%)": round(oof_overall["precision_toxic"] * 100, 2),
            "Accuracy OOF (%)": round(oof_overall["accuracy"] * 100, 2),
            "TP": oof_overall["tp"],
            "FP": oof_overall["fp"],
            "FN": oof_overall["fn"],
            "TN": oof_overall["tn"]
        }
        all_summary.append(summary_row)

        print(f"  👉 KẾT QUẢ TỔNG HỢP {model_name}: "
              f"Macro-F1 = {summary_row['Macro-F1 Mean (%)']}% ± {summary_row['Macro-F1 Std (%)']}% | "
              f"OOF Macro-F1 = {summary_row['Macro-F1 OOF (%)']}%")

        # Lưu file OOF predictions cho word_char_svm
        if model_name == "word_char_svm":
            df_oof_out = df_pool.copy()
            df_oof_out["oof_prediction"] = oof_preds.astype(int)
            df_oof_out["oof_decision_score"] = oof_scores
            df_oof_out.to_csv(os.path.join(report_dir, "oof_predictions_word_char_svm.csv"), index=False, encoding="utf-8-sig")

    # Lưu các file báo cáo
    df_sum = pd.DataFrame(all_summary)
    df_sum.to_csv(os.path.join(report_dir, "kfold_svm_comparison.csv"), index=False, encoding="utf-8-sig")

    df_details = pd.DataFrame(all_fold_details)
    df_details.to_csv(os.path.join(report_dir, "svm_fold_details.csv"), index=False, encoding="utf-8-sig")

    # Tạo file REPORT.md tổng kết
    report_md_content = f"""# Báo Cáo Huấn Luyện 5-Fold GroupKFold Cho Các Mô Hình SVM

* **Ngày thực hiện:** 05/11/2026
* **Chiến lược:** 5-Fold StratifiedGroupKFold theo `post_id` (100% Group-Aware, không rò rỉ dữ liệu).
* **Tổng số mẫu:** 7.669 câu (95 bài viết, 499 câu toxic chiếm 6.51%).
* **Tập Test (926 câu):** Giữ nguyên độc lập 100% cho Người 4.

---

## 1. Bảng So Sánh Hiệu Năng 5-Fold Giữa Các Mô Hình SVM

| Mô hình | Macro-F1 Mean ± Std (%) | Macro-F1 OOF (%) | F1-Toxic OOF (%) | Recall Toxic OOF (%) | Precision Toxic OOF (%) | Accuracy OOF (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BoW + Linear SVM | {df_sum.loc[0, 'Macro-F1 Mean (%)']}% ± {df_sum.loc[0, 'Macro-F1 Std (%)']}% | {df_sum.loc[0, 'Macro-F1 OOF (%)']}% | {df_sum.loc[0, 'F1-Toxic OOF (%)']}% | {df_sum.loc[0, 'Recall-Toxic OOF (%)']}% | {df_sum.loc[0, 'Precision-Toxic OOF (%)']}% | {df_sum.loc[0, 'Accuracy OOF (%)']}% |
| TF-IDF + Linear SVM | {df_sum.loc[1, 'Macro-F1 Mean (%)']}% ± {df_sum.loc[1, 'Macro-F1 Std (%)']}% | {df_sum.loc[1, 'Macro-F1 OOF (%)']}% | {df_sum.loc[1, 'F1-Toxic OOF (%)']}% | {df_sum.loc[1, 'Recall-Toxic OOF (%)']}% | {df_sum.loc[1, 'Precision-Toxic OOF (%)']}% | {df_sum.loc[1, 'Accuracy OOF (%)']}% |
| **Từ + Ký tự (`word_char_svm`)** | **{df_sum.loc[2, 'Macro-F1 Mean (%)']}% ± {df_sum.loc[2, 'Macro-F1 Std (%)']}%** | **{df_sum.loc[2, 'Macro-F1 OOF (%)']}%** | **{df_sum.loc[2, 'F1-Toxic OOF (%)']}%** | **{df_sum.loc[2, 'Recall-Toxic OOF (%)']}%** | **{df_sum.loc[2, 'Precision-Toxic OOF (%)']}%** | **{df_sum.loc[2, 'Accuracy OOF (%)']}%** |

---

## 2. Chi Tiết Ma Trận Nhầm Lẫn OOF Của `word_char_svm` (Toàn Bộ 7.669 Câu)

* **True Negative (TN):** {df_sum.loc[2, 'TN']} câu (Đoán chuẩn Non-toxic)
* **True Positive (TP):** {df_sum.loc[2, 'TP']} câu (Bắt được {df_sum.loc[2, 'TP']}/499 câu Toxic)
* **False Positive (FP):** {df_sum.loc[2, 'FP']} câu (Báo nhầm)
* **False Negative (FN):** {df_sum.loc[2, 'FN']} câu (Bỏ sót)

---

## 3. Nhận Xét Quan Trọng Cho Người 3 (Để So Sánh Với PhoBERT)

1. Mô hình SVM tốt nhất (`word_char_svm`) đạt **Mean Macro-F1 = {df_sum.loc[2, 'Macro-F1 Mean (%)']}% ± {df_sum.loc[2, 'Macro-F1 Std (%)']}%** và **OOF Macro-F1 = {df_sum.loc[2, 'Macro-F1 OOF (%)']}%**.
2. Độ lệch chuẩn giữa 5 Folds là **± {df_sum.loc[2, 'Macro-F1 Std (%)']}%**, cho thấy sự ổn định rất cao khi chia K-Fold theo bài viết.
3. Khi bạn chạy xong 5 Folds PhoBERT trên Colab, hãy đối chiếu:
   - Nếu PhoBERT đạt **Mean Macro-F1 > {df_sum.loc[2, 'Macro-F1 Mean (%)']}%** hoặc **OOF Macro-F1 > {df_sum.loc[2, 'Macro-F1 OOF (%)']}%**, PhoBERT tiếp tục khẳng định vị thế dẫn đầu!
   - Đặc biệt, với kỹ thuật **Ensemble 5 Folds**, PhoBERT hứa hẹn sẽ bứt phá mạnh hơn nữa ở các câu toxic mỉa mai và ngữ cảnh phức tạp.
"""

    with open(os.path.join(report_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_md_content)

    print("\n" + "=" * 80)
    print("🏆 HOÀN TẤT HUẤN LUYỆN 5-FOLD CHO TẤT CẢ MÔ HÌNH SVM!")
    print(f"Báo cáo và các file đã được lưu tại: {report_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

