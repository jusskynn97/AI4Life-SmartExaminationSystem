# cat_exam_real.py
import streamlit as st
import torch
import pandas as pd
import os
from train_pipeline import NeuralCDM, CATEngine, DiagnosticReporter

st.set_page_config(page_title="Thi Thích Nghi Toán 10 - Có Đề Thật", layout="centered")
st.title("TOÁN 10 - KIỂM TRA THÍCH NGHI THÔNG MINH")
st.markdown("**Dùng AI NeuralCDM - Ra đề theo đúng năng lực của bạn**")

# ====================== LOAD TẤT CẢ ======================
@st.cache_resource
def load_all():
    DATA_DIR = 'data/a0910'

    # --- 1. Load dữ liệu train/valid/test ---
    train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
    valid_df = pd.read_csv(f'{DATA_DIR}/valid.csv')
    test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')

    for df in [train_df, valid_df, test_df]:
        df.rename(columns={'user_id': 'student_id', 'item_id': 'question_id'}, inplace=True)

    full = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    all_students_raw = sorted(full['student_id'].unique())
    all_questions_raw = sorted(full['question_id'].unique())

    student_map = {sid: idx for idx, sid in enumerate(all_students_raw)}
    question_map = {qid: idx for idx, qid in enumerate(all_questions_raw)}

    # --- 2. Load Q-MATRIX TỪ FILE q_matrix.csv CỦA BẠN (có tên KC tiếng Việt đẹp) ---
    q_matrix_path = 'q_matrix.csv'  # ← DÙNG FILE CỦA BẠN
    qdf = pd.read_csv(q_matrix_path, index_col=0)  # index là Q0000, Q0001,...
    kc_names = qdf.columns.tolist()               # ← Tên KC tiếng Việt
    q_matrix = qdf.values.astype(int)

    # Kiểm tra tính nhất quán
    matrix_questions = qdf.index.tolist()
    missing = [q for q in all_questions_raw if q not in matrix_questions]
    if missing:
        st.warning(f"Thiếu {len(missing)} câu trong q_matrix.csv. Sẽ bỏ qua chúng.")
        all_questions_raw = [q for q in all_questions_raw if q in matrix_questions]

    # --- 3. Load model đã train ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralCDM(
        n_students=len(all_students_raw),
        n_questions=len(all_questions_raw),
        n_kcs=len(kc_names),
        hidden_dim=128
    ).to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- 4. Load nội dung câu hỏi (bắt buộc có) ---
    content_path = f'{DATA_DIR}/questions_with_content.csv'
    if not os.path.exists(content_path):
        st.error("Không tìm thấy questions_with_content.csv!")
        st.stop()

    qreal = pd.read_csv(content_path)

    # ÉP ĐỊNH DẠNG question_id → Q0000, Q0001,... (dù file có số 0,1,2 hay chữ Q)
    def format_qid(x):
        if pd.isna(x):
            return x
        return f"Q{str(int(float(x))).zfill(4)}"
    
    qreal['question_id'] = qreal['question_id'].apply(format_qid)
    qreal = qreal.set_index('question_id')

    # Chỉ giữ lại những câu có trong q_matrix
    qreal = qreal[qreal.index.isin(all_questions_raw)]

    st.success(f"Đã tải {len(qreal)} câu hỏi có nội dung + Q-matrix đẹp!")

    # --- 5. Tạo CAT Engine ---
    cat = CATEngine(model, q_matrix, question_pool=all_questions_raw)

    return {
        'model': model,
        'cat': cat,
        'q_matrix': q_matrix,
        'kc_names': kc_names,
        'student_map': student_map,
        'qreal': qreal,
        'device': device,
        'all_questions_raw': all_questions_raw
    }

data = load_all()

# ====================== SESSION STATE ======================
if 'started' not in st.session_state:
    st.session_state.started = False
    st.session_state.student_idx = None
    st.session_state.answered = []

# ====================== GIAO DIỆN ======================
if not st.session_state.started:
    st.markdown("### Nhập mã học sinh để bắt đầu")
    sid = st.text_input("Mã học sinh (ví dụ: 15607)", "15607")
    if st.button("Bắt Đầu Thi Thật", type="primary"):
        try:
            sid_int = int(sid)
            if sid_int in data['student_map']:
                st.session_state.student_idx = data['student_map'][sid_int]
                st.session_state.started = True
                st.session_state.answered = []
                st.success(f"Đã tìm thấy học sinh {sid_int}!")
                st.rerun()
            else:
                st.error(f"Không tìm thấy học sinh {sid_int}")
        except ValueError:
            st.error("Vui lòng nhập số!")

else:
    raw_id = next(k for k, v in data['student_map'].items() if v == st.session_state.student_idx)
    st.markdown(f"**Học sinh:** {raw_id} | Đã làm: {len(st.session_state.answered)}/20 câu")

    if len(st.session_state.answered) >= 20:
        st.balloons()
        st.success("HOÀN THÀNH BÀI THI!")

        with st.spinner("Đang chẩn đoán kiến thức..."):
            report = DiagnosticReporter.generate_report(
                data['model'], st.session_state.student_idx,
                data['q_matrix'], data['kc_names'],
                threshold_weak=0.6, threshold_strong=0.82
            )
        DiagnosticReporter.print_report(report)

        if st.button("Thi lại học sinh khác"):
            st.session_state.clear()
            st.rerun()

    else:
        answered_qids = [x[0] for x in st.session_state.answered]

        with st.spinner("AI đang chọn câu hỏi phù hợp nhất với bạn..."):
            next_qid = data['cat'].select_next_question(
                st.session_state.student_idx,
                answered=answered_qids
            )

            # Nếu câu được chọn không có nội dung → fallback chọn trong số có nội dung
            if next_qid not in data['qreal'].index:
                available = [q for q in data['all_questions_raw'] if q not in answered_qids and q in data['qreal'].index]
                if not available:
                    st.error("Hết câu hỏi có nội dung!")
                    st.stop()
                # Chọn câu có Fisher Information cao nhất trong số có nội dung
                best_q, best_info = None, -1
                s = torch.tensor([st.session_state.student_idx], device=data['device'])
                with torch.no_grad():
                    for qid in available:
                        qi = data['cat'].qid_to_idx[qid]
                        qq = torch.tensor([qi], device=data['device'])
                        qm = torch.tensor(data['q_matrix'][qi], dtype=torch.float, device=data['device']).unsqueeze(0)
                        p = data['model'](s, qq, qm).item()
                        info = p * (1 - p)
                        if info > best_info:
                            best_info = info
                            best_q = qid
                next_qid = best_q

        q = data['qreal'].loc[next_qid]
        st.markdown(f"### Câu {len(st.session_state.answered)+1}")
        st.markdown(f"**Mã:** `{next_qid}` | **Kiến thức:** {', '.join([kc for kc, val in zip(data['kc_names'], data['q_matrix'][data['cat'].qid_to_idx[next_qid]]) if val==1])}")
        st.latex(q['content'])

        cols = st.columns(4)
        options = ['A', 'B', 'C', 'D']
        for i, col in enumerate(cols):
            btn = col.button(
                f"{options[i]}. {q[f'option_{options[i].lower()}']}",
                use_container_width=True,
                type="primary" if options[i] == q['correct_answer'] else "secondary"
            )
            if btn:
                correct = 1 if options[i] == q['correct_answer'] else 0
                st.session_state.answered.append((next_qid, correct))
                if correct:
                    st.success("Rất tốt! Đúng rồi!")
                else:
                    st.error(f"Sai rồi! Đáp án đúng là: **{q['correct_answer']}**")
                st.rerun()