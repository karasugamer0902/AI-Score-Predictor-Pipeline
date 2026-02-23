from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. 初始化 FastAPI 與 載入模型
app = FastAPI(title="學生分數預測系統 API")
model = joblib.load('../models/super_stable_model_v2.pkl')

# 2. 定義數據格式 (讓 API 知道要接收什麼)
class StudentData(BaseModel):
    age: float = None
    study_hours: float
    gender: str

# 3. 建立預測路徑 (Endpoint)
@app.post("/predict")
def predict(data: StudentData):
    # 轉換為 DataFrame
    input_df = pd.DataFrame({
        '年齡': [data.age],
        '讀書小時': [data.study_hours],
        '性別': [data.gender]
    })
    
    # 預測
    prediction = model.predict(input_df)[0]
    
    return {
        "status": "success",
        "predicted_score": round(prediction, 2),
        "message": "預測完成"
    }

# 4. 根目錄測試
@app.get("/")
def home():
    return {"message": "AI 預測伺服器已啟動"}