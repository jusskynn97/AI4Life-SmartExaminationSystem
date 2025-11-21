# run_real_test.py
import torch
import pandas as pd
import json
import os
from train_pipeline import (
    NeuralCDM, QMatrixBuilder, DiagnosticReporter, CATEngine
)

# ====================== CẤU HÌNH ======================
DATA_DIR = 'data/a0910'
MODEL_PATH = 'best_model.pth'
Q_MATRIX_REAL_PATH = 'q_matrix_real.csv'   # file được sinh ra khi train
# Nếu q_matrix_real.csv chưa có thì dùng q_matrix.csv cũ (nhưng ưu tiên q_matrix_real.csv)

# ====================== LOAD DỮ LIỆU ======================
print("Đang tải danh sách câu hỏi và học sinh...")

# Load train/valid để lấy full list student & question (để map đúng index)
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
valid_df = pd.read_csv(f'{DATA_DIR}/valid.csv')
test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')

full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
all_students_raw = sorted(full_df['user_id'].unique())
all_questions_raw = sorted(full_df['item_id'].unique())

# Tạo mapping giống hệt lúc train
student_map = {sid: idx for idx, sid in enumerate(all_students_raw)}
question_map = {qid: idx for idx, qid in enumerate(all_questions_raw)}

n_students = len(all_students_raw)
n_questions = len(all_questions_raw)

print(f"Số học sinh: {n_students}")
print(f"Số câu hỏi: {n_questions}")

# ====================== LOAD Q-MATRIX ======================
if os.path.exists(Q_MATRIX_REAL_PATH):
    print(f"Đang load Q-matrix từ {Q_MATRIX_REAL_PATH}")
    q_df = pd.read_csv(Q_MATRIX_REAL_PATH, index_col=0)
    question_ids_from_q = q_df.index.tolist()
    kc_names = q_df.columns.tolist()
    q_matrix = q_df.values.astype(int)
else:
    print("Không tìm thấy q_matrix_real.csv → dùng file q_matrix.csv cũ")
    q_df = pd.read_csv('q_matrix.csv', index_col=0)
    kc_names = q_df.columns.tolist()
    q_matrix = q_df.values.astype(int)
    question_ids_from_q = q_df.index.tolist()

n_kcs = len(kc_names)

# Kiểm tra tính nhất quán
assert len(question_ids_from_q) == n_questions, "Q-matrix và số câu hỏi không khớp!"

# ====================== LOAD MODEL ======================
print("Đang load mô hình đã train...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralCDM(
    n_students=n_students,
    n_questions=n_questions,
    n_kcs=n_kcs,
    hidden_dim=128
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Load model thành công!")

# ====================== TẠO CAT ENGINE & REPORTER ======================
cat_engine = CATEngine(model, q_matrix, question_pool=all_questions_raw)
reporter = DiagnosticReporter()

# ====================== CHỌN HỌC SINH ======================
print("\n" + "="*60)
print("DANH SÁCH MỘT SỐ HỌC SINH (gõ đúng user_id gốc):")
sample_students = all_students_raw[:10].tolist()
print("Ví dụ:", sample_students)

student_id_raw = input("\nNhập user_id của học sinh muốn kiểm tra (ví dụ: 15607): ").strip()

if student_id_raw == '':
    student_id_raw = '15607'  # mặc định một em có nhiều dữ liệu
    print(f"→ Dùng mặc định: {student_id_raw}")

if student_id_raw not in student_map:
    print("ID không tồn tại! Dùng một ID mẫu...")
    student_id_raw = all_students_raw[100]
    
student_id_idx = student_map[student_id_raw]
print(f"Đang xử lý học sinh: {student_id_raw} (index: {student_id_idx})")

# ====================== TẠO BÀI KIỂM TRA THÍCH NGHI ======================
print("\nĐang tạo bài kiểm tra thích nghi 10 câu...")
adaptive_test = cat_engine.generate_adaptive_test(student_id_idx, n_questions=10)

print("Câu hỏi được chọn theo thứ tự (từ dễ → khó hoặc ngược lại theo năng lực):")
for i, qid in enumerate(adaptive_test):
    related_kcs = [kc_names[j] for j, val in enumerate(q_matrix[question_map[qid]]) if val == 1]
    print(f"  {i+1:2}. {qid} → KC: {', '.join(related_kcs)}")

# ====================== BÁO CÁO CHẨN ĐOÁN ======================
print("\nĐang tạo báo cáo chẩn đoán kiến thức...")
report = reporter.generate_report(
    model=model,
    student_id=student_id_idx,
    q_matrix=q_matrix,
    kc_names=kc_names,
    threshold_weak=0.55,
    threshold_strong=0.80
)

# In báo cáo đẹp
reporter.print_report(report)

# Lưu báo cáo
report_filename = f"report_student_{student_id_raw}.json"
reporter.save_report(report, report_filename)
print(f"Đã lưu báo cáo chi tiết → {report_filename}")

# ====================== HOÀN TẤT ======================
print("\nHOÀN THÀNH! Bạn đã chạy thành công NeuralCDM trên dữ liệu thực")
print("Các file quan trọng:")
print("  • best_model.pth          ← mô hình")
print("  • q_matrix_real.csv       ← ma trận Q")
print(f"  • {report_filename}   ← báo cáo học sinh {student_id_raw}")