"""
NeuralCDM Training Pipeline - Complete Implementation
Tích hợp dataset thực từ CSV + Model training + CAT Engine
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# ==================== COLLATE FN (BẮT BUỘC) ====================
def cdm_collate(batch):
    student   = torch.LongTensor([b['student_id'] for b in batch])
    question  = torch.LongTensor([b['question_id'] for b in batch])
    qrow      = torch.stack([b['q_matrix_row'] for b in batch])  # (B, n_kcs)
    label     = torch.FloatTensor([b['correct'] for b in batch])
    return {
        'student_id': student,
        'question_id': question,
        'q_matrix_row': qrow,
        'correct': label
    }

# ==================== BƯỚC 1: TẢI DATASET THỰC ====================
class DatasetLoader:
    """Tải dữ liệu từ file CSV"""
    
    @staticmethod
    def load_csv(filepath, rename_cols=True):
        df = pd.read_csv(filepath)
        if rename_cols:
            df = df.rename(columns={
                'user_id': 'student_id',
                'item_id': 'question_id',
                'score': 'correct'
            })
        if 'correct' in df.columns:
            df['correct'] = df['correct'].astype(int)
        return df

    @staticmethod
    def load_train(data_dir='data/a0910'):
        return DatasetLoader.load_csv(f'{data_dir}/train.csv')

    @staticmethod
    def load_valid(data_dir='data/a0910'):
        return DatasetLoader.load_csv(f'{data_dir}/valid.csv')

    @staticmethod
    def load_test(data_dir='data/a0910'):
        return DatasetLoader.load_csv(f'{data_dir}/test.csv')

# ==================== BƯỚC 2: XÂY DỰNG Q-MATRIX TỪ item.csv ====================
class QMatrixBuilder:
    
    @staticmethod
    def build_from_item_csv(item_csv_path='data/a0910/item.csv', question_ids=None):
        df = pd.read_csv(item_csv_path)
        
        # Lấy tất cả KC duy nhất
        all_kcs = set()
        for kcs_str in df['knowledge_code']:
            kcs_str = str(kcs_str).strip()
            kcs_str = kcs_str.replace('"', '').replace("'", "")
            if kcs_str.startswith('[') and kcs_str.endswith(']'):
                kcs_str = kcs_str[1:-1]
            kcs = [int(x.strip()) for x in kcs_str.split(',') if x.strip().isdigit()]
            all_kcs.update(kcs)
        
        kc_list = sorted(all_kcs)
        n_kcs = len(kc_list)
        kc_to_idx = {kc: idx for idx, kc in enumerate(kc_list)}
        
        # Khởi tạo Q-matrix
        if question_ids is None:
            question_ids = df['item_id'].unique()
        n_questions = len(question_ids)
        q_matrix = np.zeros((n_questions, n_kcs), dtype=int)
        
        qid_to_row = {qid: idx for idx, qid in enumerate(question_ids)}
        
        for _, row in df.iterrows():
            qid = row['item_id']
            if qid not in qid_to_row:
                continue
            row_idx = qid_to_row[qid]
            kcs_str = str(row['knowledge_code']).strip()
            kcs_str = kcs_str.replace('"', '').replace("'", "")
            if kcs_str.startswith('[') and kcs_str.endswith(']'):
                kcs_str = kcs_str[1:-1]
            kcs = [int(x.strip()) for x in kcs_str.split(',') if x.strip().isdigit()]
            for kc in kcs:
                if kc in kc_to_idx:
                    q_matrix[row_idx, kc_to_idx[kc]] = 1
        
        return q_matrix, kc_list, kc_to_idx, qid_to_row

    @staticmethod
    def save_to_file(q_matrix, question_ids, kc_names, filepath='q_matrix.csv'):
        df = pd.DataFrame(q_matrix, index=question_ids, columns=kc_names)
        df.to_csv(filepath)
        print(f"Lưu Q-matrix: {filepath}")

# ==================== BƯỚC 3: MÔ HÌNH NEURALCDM ====================
class NeuralCDM(nn.Module):
    def __init__(self, n_students, n_questions, n_kcs, hidden_dim=128, dropout=0.2):
        super(NeuralCDM, self).__init__()
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_kcs = n_kcs
        
        self.student_emb = nn.Embedding(n_students, hidden_dim)
        self.question_emb = nn.Embedding(n_questions, hidden_dim)
        self.kc_emb = nn.Embedding(n_kcs, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, student_ids, question_ids, q_matrix_rows):
        student_emb = self.student_emb(student_ids)
        question_emb = self.question_emb(question_ids)
        
        kc_weights = q_matrix_rows.unsqueeze(-1)
        kc_emb_all = self.kc_emb.weight.unsqueeze(0).expand(student_ids.size(0), -1, -1)
        kc_emb = (kc_emb_all * kc_weights).sum(dim=1)
        
        kc_count = q_matrix_rows.sum(dim=1, keepdim=True).clamp(min=1.0)
        kc_emb = kc_emb / kc_count
        
        x = torch.cat([student_emb, question_emb, kc_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x).squeeze(-1)
    
    def get_student_knowledge(self, student_id, q_matrix):
      self.eval()
      device = next(self.parameters()).device  # ← LẤY DEVICE CỦA MODEL
      with torch.no_grad():
          knowledge = torch.zeros(self.n_kcs, device=device)
          counts = torch.zeros(self.n_kcs, device=device)
          
          student_tensor = torch.LongTensor([student_id]).to(device)  # ← .to(device)

          for q in range(self.n_questions):
              question_tensor = torch.LongTensor([q]).to(device)  # ← .to(device)
              q_matrix_row = torch.FloatTensor(q_matrix[q]).unsqueeze(0).to(device)  # ← .to(device)
              
              prob = self.forward(student_tensor, question_tensor, q_matrix_row)
              
              for kc_idx in range(self.n_kcs):
                  if q_matrix[q, kc_idx] == 1:
                      knowledge[kc_idx] += prob.item()
                      counts[kc_idx] += 1
          
          # Chuyển về CPU để trả về numpy
          knowledge = knowledge.cpu() / (counts.cpu() + 1e-8)
      return knowledge.numpy()

# ==================== BƯỚC 4: DATASET ====================
class CDMDataset(Dataset):
    def __init__(self, data, q_matrix, student_map, question_map):
        self.data = data.reset_index(drop=True)
        self.q_matrix = torch.FloatTensor(q_matrix)
        self.student_map = student_map
        self.question_map = question_map
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'student_id': int(row['student_id']),
            'question_id': int(row['question_id']),
            'q_matrix_row': self.q_matrix[int(row['question_id'])],
            'correct': int(row['correct'])
        }

# ==================== BƯỚC 5: TRAINER ====================
class NeuralCDMTrainer:
    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            probs = self.model(
                batch['student_id'].to(self.device),
                batch['question_id'].to(self.device),
                batch['q_matrix_row'].to(self.device)
            )
            loss = self.criterion(probs, batch['correct'].to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = correct = total = 0
        with torch.no_grad():
            for batch in loader:
                probs = self.model(
                    batch['student_id'].to(self.device),
                    batch['question_id'].to(self.device),
                    batch['q_matrix_row'].to(self.device)
                )
                loss = self.criterion(probs, batch['correct'].to(self.device))
                total_loss += loss.item()
                preds = (probs > 0.5).long()
                correct += (preds == batch['correct'].to(self.device).long()).sum().item()
                total += len(probs)
        return total_loss / len(loader), correct / total
    
    def train(self, train_loader, val_loader, n_epochs=30, early_stopping=5):
        best_val_loss = float('inf')
        patience = 0
        print(f"\nBắt đầu training trên {self.device}")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12}")
        print("-" * 50)
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            print(f"{epoch+1:<6} {train_loss:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience += 1
                if patience >= early_stopping:
                    print(f"Early stopping tại epoch {epoch+1}")
                    break
        print(f"\nTraining hoàn thành! Best Val Loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, filepath='best_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }, filepath)

# ==================== BƯỚC 6: CAT ENGINE ====================
class CATEngine:
    def __init__(self, model, q_matrix, question_pool):
        self.model = model
        self.q_matrix = q_matrix
        self.question_pool = question_pool
        self.qid_to_idx = {qid: idx for idx, qid in enumerate(question_pool)}
        self.device = next(model.parameters()).device

    def select_next_question(self, student_id, answered):
      unanswered = [q for q in self.question_pool if q not in answered]
      if not unanswered: return None
      best_q = None
      best_info = -1
      s = torch.tensor([student_id], dtype=torch.long, device=self.device)  # ← DÙNG self.device
      with torch.no_grad():
          for qid in unanswered:
              qi = self.qid_to_idx[qid]
              qq = torch.tensor([qi], dtype=torch.long, device=self.device)
              qm = torch.tensor(self.q_matrix[qi], dtype=torch.float, device=self.device).unsqueeze(0)
              p = self.model(s, qq, qm).item()
              info = p * (1 - p)
              if info > best_info:
                  best_info = info
                  best_q = qid
      return best_q

    def generate_adaptive_test(self, student_id, n_questions=10):
        answered = []
        test = []
        for _ in range(n_questions):
            q = self.select_next_question(student_id, answered)
            if q is None: break
            test.append(q)
            answered.append(q)
        return test

# ==================== BƯỚC 7: DIAGNOSTIC REPORT ====================
class DiagnosticReporter:
    @staticmethod
    def generate_report(model, student_id, q_matrix, kc_names, threshold_weak=0.6, threshold_strong=0.8):
        knowledge = model.get_student_knowledge(student_id, q_matrix)
        report = {
            'student_id': student_id,
            'knowledge_levels': {},
            'weak_kcs': [],
            'strong_kcs': [],
            'recommendations': []
        }
        for i, kc_name in enumerate(kc_names):
            mastery = knowledge[i]
            report['knowledge_levels'][kc_name] = float(mastery)
            if mastery < threshold_weak:
                report['weak_kcs'].append({
                    'kc': kc_name,
                    'mastery': float(mastery),
                    'recommendation': f"Cần học lại {kc_name}"
                })
            elif mastery >= threshold_strong:
                report['strong_kcs'].append({
                    'kc': kc_name,
                    'mastery': float(mastery)
                })
        report['weak_kcs'].sort(key=lambda x: x['mastery'])
        report['strong_kcs'].sort(key=lambda x: x['mastery'], reverse=True)
        return report
    
    @staticmethod
    def save_report(report, filepath='report.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Lưu báo cáo: {filepath}")
    
    @staticmethod
    def print_report(report):
        print("\n" + "="*60)
        print(f"BÁO CÁO CHẨN ĐOÁN - Học sinh: {report['student_id']}")
        print("="*60)
        print("\nNĂNG LỰC TỔNG QUAN:")
        for kc, mastery in report['knowledge_levels'].items():
            bar = "█" * int(mastery * 30) + "░" * (30 - int(mastery * 30))
            print(f"  {kc:<20} {bar} {mastery*100:>5.1f}%")
        if report['weak_kcs']:
            print("\nCẦN CẢI THIỆN:")
            for item in report['weak_kcs']:
                print(f"  • {item['kc']}: {item['mastery']*100:.1f}% → {item['recommendation']}")
        if report['strong_kcs']:
            print("\nĐÃ THÀNH THẠO:")
            for item in report['strong_kcs']:
                print(f"  • {item['kc']}: {item['mastery']*100:.1f}%")
        print("\n" + "="*60 + "\n")

# ==================== MAIN ====================
def main():
    print("="*60)
    print("NEURALCDM - DỮ LIỆU THỰC TỪ CSV")
    print("="*60)
    
    DATA_DIR = 'data/a0910'
    BATCH_SIZE = 64
    N_EPOCHS = 30

    # 1. TẢI DỮ LIỆU
    print("\nBƯỚC 1: TẢI DỮ LIỆU")
    train_df = DatasetLoader.load_train(DATA_DIR)
    valid_df = DatasetLoader.load_valid(DATA_DIR)
    test_df  = DatasetLoader.load_test(DATA_DIR)
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    # Gộp để lấy danh sách đầy đủ
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    all_students_raw = sorted(full_df['student_id'].unique())
    all_questions_raw = sorted(full_df['question_id'].unique())

    # Tạo map
    student_map = {sid: idx for idx, sid in enumerate(all_students_raw)}
    question_map = {qid: idx for idx, qid in enumerate(all_questions_raw)}

    # ÁNH XẠ TRÊN TẤT CẢ DATAFRAME
    for df in [train_df, valid_df, test_df]:
        df['student_id'] = df['student_id'].map(student_map)
        df['question_id'] = df['question_id'].map(question_map)

    # ĐẶT TÊN RÕ RÀNG
    all_students = all_students_raw
    all_questions = all_questions_raw  # ĐÃ SỬA: ĐỊNH NGHĨA all_questions

    # 2. XÂY DỰNG Q-MATRIX
    print("\nBƯỚC 2: XÂY DỰNG Q-MATRIX")
    q_matrix, kc_list, _, _ = QMatrixBuilder.build_from_item_csv(
        item_csv_path=f'{DATA_DIR}/item.csv',
        question_ids=all_questions
    )
    KC_NAMES = [f'KC{kc}' for kc in kc_list]
    QMatrixBuilder.save_to_file(q_matrix, all_questions, KC_NAMES, 'q_matrix_real.csv')

    # 3. DATASET & DATALOADER
    print("\nBƯỚC 3: CHUẨN BỊ DỮ LIỆU")
    train_dataset = CDMDataset(train_df, q_matrix, student_map, question_map)
    valid_dataset = CDMDataset(valid_df, q_matrix, student_map, question_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cdm_collate)
    val_loader   = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=cdm_collate)

    # 4. TẠO MODEL
    print("\nBƯỚC 4: TẠO MÔ HÌNH")
    model = NeuralCDM(
        n_students=len(all_students),
        n_questions=len(all_questions),
        n_kcs=len(KC_NAMES),
        hidden_dim=128
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. HUẤN LUYỆN
    print("\nBƯỚC 5: HUẤN LUYỆN")
    trainer = NeuralCDMTrainer(model)
    trainer.train(train_loader, val_loader, n_epochs=N_EPOCHS, early_stopping=5)

    # 6. CAT ENGINE
    print("\nBƯỚC 6: CAT ENGINE")
    cat_engine = CATEngine(model, q_matrix, question_pool=all_questions)
    test_student_idx = valid_df.iloc[0]['student_id']
    adaptive_test = cat_engine.generate_adaptive_test(test_student_idx, n_questions=10)
    print(f"Adaptive test (item_id): {adaptive_test}")

    # 7. BÁO CÁO
    print("\nBƯỚC 7: BÁO CÁO CHẨN ĐOÁN")
    reporter = DiagnosticReporter()
    report = reporter.generate_report(model, test_student_idx, q_matrix, KC_NAMES)
    reporter.print_report(report)
    reporter.save_report(report, 'student_report_real.json')

    print("\nHOÀN THÀNH!")
    print("Files:")
    print("  • q_matrix_real.csv")
    print("  • best_model.pth")
    print("  • student_report_real.json")

if __name__ == "__main__":
    main()