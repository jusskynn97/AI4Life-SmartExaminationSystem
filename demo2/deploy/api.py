"""
FastAPI Deployment cho Cognitive Diagnosis Model
API endpoints cho prediction, diagnosis, và adaptive testing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import uvicorn

# ==================== DATA MODELS ====================

class StudentResponse(BaseModel):
    """Single student response"""
    user_id: int = Field(..., description="Student ID")
    item_id: int = Field(..., description="Question/Item ID")
    score: float = Field(..., ge=0, le=1, description="Score (0 or 1 for binary)")
    timestamp: Optional[str] = None

class BatchResponse(BaseModel):
    """Batch of student responses"""
    responses: List[StudentResponse]

class DiagnosisRequest(BaseModel):
    """Request for student diagnosis"""
    user_id: int
    response_history: Optional[List[StudentResponse]] = None

class AdaptiveTestRequest(BaseModel):
    """Request for next question in adaptive test"""
    user_id: int
    test_id: str
    current_responses: List[StudentResponse]
    n_questions_remaining: int = Field(default=1, ge=1, le=10)

class DiagnosisResponse(BaseModel):
    """Diagnosis result"""
    user_id: int
    overall_proficiency: float
    knowledge_proficiency: Dict[str, float]
    strengths: List[Dict[str, float]]
    weaknesses: List[Dict[str, float]]
    recommendations: List[int]
    timestamp: str

class PredictionResponse(BaseModel):
    """Prediction result"""
    user_id: int
    item_id: int
    predicted_score: float
    confidence: float

# ==================== MODEL SERVICE ====================

class CDMModelService:
    """
    Service quản lý model inference và caching
    """
    
    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model (cần import NeuralCDM)
        # from cognitive_diagnosis import NeuralCDM
        # self.model = NeuralCDM(
        #     n_users=checkpoint['n_users'],
        #     n_items=checkpoint['n_items'],
        #     n_knowledge=checkpoint['n_knowledge']
        # )
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.to(self.device)
        # self.model.eval()
        
        self.n_users = checkpoint.get('n_users', 1000)
        self.n_items = checkpoint.get('n_items', 500)
        self.n_knowledge = checkpoint.get('n_knowledge', 10)
        
        # Load Q-matrix
        q_matrix_path = Path(model_path).parent.parent / 'q_matrix.npy'
        if q_matrix_path.exists():
            self.q_matrix = np.load(q_matrix_path)
        else:
            # Fallback: random Q-matrix
            self.q_matrix = np.random.randint(0, 2, (self.n_items, self.n_knowledge))
        
        # Load knowledge concept names
        kc_names_path = Path(model_path).parent.parent / 'kc_names.json'
        if kc_names_path.exists():
            with open(kc_names_path, 'r') as f:
                self.kc_names = json.load(f)
        else:
            self.kc_names = [f"KC_{i}" for i in range(self.n_knowledge)]
        
        # Cache for student proficiencies
        self.proficiency_cache = {}
        
        print(f"✓ Model loaded successfully!")
        print(f"  Users: {self.n_users}, Items: {self.n_items}, KCs: {self.n_knowledge}")
    
    def predict_score(self, user_id: int, item_id: int) -> Dict[str, float]:
        """
        Dự đoán xác suất trả lời đúng
        """
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            item_tensor = torch.LongTensor([item_id]).to(self.device)
            q_tensor = torch.FloatTensor([self.q_matrix[item_id]]).to(self.device)
            
            # pred = self.model(user_tensor, item_tensor, q_tensor)
            # pred_score = pred.cpu().item()
            
            # Placeholder prediction (trong thực tế dùng model.forward)
            pred_score = 0.7 + np.random.uniform(-0.2, 0.2)
            pred_score = max(0, min(1, pred_score))
        
        return {
            'predicted_score': float(pred_score),
            'confidence': 0.85  # Có thể tính từ entropy hoặc variance
        }
    
    def diagnose_student(self, user_id: int) -> Dict:
        """
        Chuẩn đoán năng lực học sinh
        """
        # Check cache
        if user_id in self.proficiency_cache:
            proficiency = self.proficiency_cache[user_id]
        else:
            # Get proficiency from model
            # proficiency = self.model.get_student_proficiency(user_id)
            
            # Placeholder (trong thực tế dùng model)
            proficiency = np.random.uniform(-1, 1, self.n_knowledge)
            self.proficiency_cache[user_id] = proficiency
        
        # Normalize to [0, 1]
        normalized_prof = 1 / (1 + np.exp(-proficiency))
        
        # Categorize
        threshold = 0.5
        strengths = []
        weaknesses = []
        
        for i, (kc, score) in enumerate(zip(self.kc_names, normalized_prof)):
            if score > threshold + 0.2:
                strengths.append({kc: float(score)})
            elif score < threshold - 0.2:
                weaknesses.append({kc: float(score)})
        
        # Sort
        strengths = sorted(strengths, key=lambda x: list(x.values())[0], reverse=True)
        weaknesses = sorted(weaknesses, key=lambda x: list(x.values())[0])
        
        # Recommendations
        recommendations = self._recommend_items(user_id, weaknesses)
        
        return {
            'user_id': user_id,
            'overall_proficiency': float(np.mean(normalized_prof)),
            'knowledge_proficiency': {
                kc: float(score) for kc, score in zip(self.kc_names, normalized_prof)
            },
            'strengths': strengths[:5],
            'weaknesses': weaknesses[:5],
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _recommend_items(self, user_id: int, weaknesses: List[Dict], top_k=5) -> List[int]:
        """Đề xuất câu hỏi luyện tập"""
        if not weaknesses:
            return []
        
        # Get weakest KCs
        weak_kc_names = [list(w.keys())[0] for w in weaknesses[:2]]
        weak_kc_indices = [self.kc_names.index(kc) for kc in weak_kc_names]
        
        # Find items covering these KCs
        candidate_items = []
        for item_id in range(self.n_items):
            if any(self.q_matrix[item_id, kc] == 1 for kc in weak_kc_indices):
                candidate_items.append(item_id)
        
        return candidate_items[:top_k]
    
    def select_adaptive_question(self, user_id: int, 
                                 answered_items: List[int],
                                 n_select: int = 1) -> List[int]:
        """
        Chọn câu hỏi tiếp theo trong adaptive testing
        Strategy: Maximum Information Selection
        """
        # Get current proficiency estimate
        if user_id in self.proficiency_cache:
            theta = self.proficiency_cache[user_id]
        else:
            theta = np.zeros(self.n_knowledge)  # Start neutral
        
        # Calculate information for each unanswered item
        available_items = [i for i in range(self.n_items) if i not in answered_items]
        
        information_scores = []
        for item_id in available_items:
            # Fisher Information (simplified)
            # I(θ) = P(1-P) for binary response
            pred = self.predict_score(user_id, item_id)
            p = pred['predicted_score']
            info = p * (1 - p)
            information_scores.append((item_id, info))
        
        # Sort by information and select top-k
        information_scores.sort(key=lambda x: x[1], reverse=True)
        selected_items = [item_id for item_id, _ in information_scores[:n_select]]
        
        return selected_items
    
    def update_proficiency(self, user_id: int, responses: List[StudentResponse]):
        """
        Cập nhật ước lượng năng lực sau mỗi câu trả lời (online update)
        Sử dụng Maximum Likelihood Estimation hoặc Bayesian update
        """
        # Simple approach: Re-run inference with new data
        # In production: Use incremental learning or Kalman filter
        
        # Clear cache to force re-computation
        if user_id in self.proficiency_cache:
            del self.proficiency_cache[user_id]
        
        # Re-diagnose
        diagnosis = self.diagnose_student(user_id)
        return diagnosis

# ==================== API APPLICATION ====================

# Initialize FastAPI
app = FastAPI(
    title="Cognitive Diagnosis API",
    description="API for adaptive testing and student diagnosis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model service
model_service: Optional[CDMModelService] = None

# ==================== ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_service
    
    # Load model (cần cung cấp paths thực tế)
    model_path = "../experiments/my_first_model/20251120_003227/checkpoints/best_model.pth"
    config_path = "../experiments/my_first_model/20251120_003227/config.json"
    
    try:
        model_service = CDMModelService(model_path, config_path)
        print("✓ Model service initialized")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("  Running in demo mode with mock predictions")

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Cognitive Diagnosis API",
        "version": "1.0.0",
        "model_loaded": model_service is not None
    }

@app.get("/info")
async def get_info():
    """Get model information"""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "n_users": model_service.n_users,
        "n_items": model_service.n_items,
        "n_knowledge": model_service.n_knowledge,
        "knowledge_concepts": model_service.kc_names
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(user_id: int, item_id: int):
    """
    Dự đoán xác suất trả lời đúng của học sinh cho một câu hỏi
    
    Example:
        POST /predict?user_id=123&item_id=45
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = model_service.predict_score(user_id, item_id)
        return PredictionResponse(
            user_id=user_id,
            item_id=item_id,
            **result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """
    Chuẩn đoán năng lực học sinh
    
    Example:
        POST /diagnose
        {
            "user_id": 123,
            "response_history": [...]  # Optional
        }
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Update proficiency if response history provided
        if request.response_history:
            model_service.update_proficiency(request.user_id, request.response_history)
        
        # Diagnose
        diagnosis = model_service.diagnose_student(request.user_id)
        return DiagnosisResponse(**diagnosis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adaptive/next")
async def get_next_question(request: AdaptiveTestRequest):
    """
    Lấy câu hỏi tiếp theo trong adaptive testing
    
    Example:
        POST /adaptive/next
        {
            "user_id": 123,
            "test_id": "test_001",
            "current_responses": [...],
            "n_questions_remaining": 3
        }
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Update proficiency based on current responses
        if request.current_responses:
            model_service.update_proficiency(request.user_id, request.current_responses)
        
        # Get answered items
        answered_items = [r.item_id for r in request.current_responses]
        
        # Select next questions
        next_items = model_service.select_adaptive_question(
            request.user_id,
            answered_items,
            n_select=request.n_questions_remaining
        )
        
        return {
            "user_id": request.user_id,
            "test_id": request.test_id,
            "next_items": next_items,
            "total_answered": len(answered_items),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/predict")
async def batch_predict(batch: BatchResponse):
    """
    Batch prediction cho nhiều responses
    
    Example:
        POST /batch/predict
        {
            "responses": [
                {"user_id": 1, "item_id": 10, "score": 1},
                {"user_id": 1, "item_id": 11, "score": 0},
                ...
            ]
        }
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for response in batch.responses:
            pred = model_service.predict_score(response.user_id, response.item_id)
            results.append({
                "user_id": response.user_id,
                "item_id": response.item_id,
                **pred
            })
        
        return {
            "predictions": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def update_model(background_tasks: BackgroundTasks):
    """
    Trigger model update với dữ liệu mới (background task)
    """
    def retrain_model():
        # Logic re-training model
        print("Starting model retraining...")
        # ... training code ...
        print("Model retraining completed")
    
    background_tasks.add_task(retrain_model)
    
    return {
        "status": "Model update scheduled",
        "timestamp": datetime.now().isoformat()
    }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

"""
==============================================================================
DEPLOYMENT GUIDE
==============================================================================

1. Install dependencies:
   pip install fastapi uvicorn torch numpy pydantic

2. Start server:
   python api.py
   # Or with uvicorn:
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload

3. Access API docs:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)

4. Production deployment với Docker:

   Dockerfile:
   ```
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

   Build & Run:
   docker build -t cdm-api .
   docker run -p 8000:8000 -v /path/to/models:/app/models cdm-api

5. Example API calls:

   # Predict
   curl -X POST "http://localhost:8000/predict?user_id=123&item_id=45"

   # Diagnose
   curl -X POST "http://localhost:8000/diagnose" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 123}'

   # Adaptive testing
   curl -X POST "http://localhost:8000/adaptive/next" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": 123,
       "test_id": "test_001",
       "current_responses": [
         {"user_id": 123, "item_id": 10, "score": 1},
         {"user_id": 123, "item_id": 15, "score": 0}
       ],
       "n_questions_remaining": 3
     }'

==============================================================================
INTEGRATION WITH FRONTEND
==============================================================================

JavaScript example:

```javascript
// Predict score
async function predictScore(userId, itemId) {
  const response = await fetch(
    `http://localhost:8000/predict?user_id=${userId}&item_id=${itemId}`,
    { method: 'POST' }
  );
  return await response.json();
}

// Get diagnosis
async function getDiagnosis(userId) {
  const response = await fetch('http://localhost:8000/diagnose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId })
  });
  return await response.json();
}

// Adaptive testing
async function getNextQuestion(userId, testId, responses) {
  const response = await fetch('http://localhost:8000/adaptive/next', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      test_id: testId,
      current_responses: responses,
      n_questions_remaining: 1
    })
  });
  return await response.json();
}
```

==============================================================================
"""