import joblib
import pandas as pd
import numpy as np
import os

def load_model(model_path):
    """載入訓練好的 Pipeline 模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    return joblib.load(model_path)

def predict_score(model, age, hours, gender):
    """
    接收原始參數並進行預測
    即使 age 是 None，Pipeline 內部的 SimpleImputer 也會處理
    """
    # 將輸入轉換為 DataFrame 格式
    input_df = pd.DataFrame({
        '年齡': [age],
        '讀書小時': [hours],
        '性別': [gender]
    })
    
    # 進行預測
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    # 1. 指定模型路徑 (假設你在專案根目錄執行)
    MODEL_FILE = '../models/super_stable_model_v2.pkl'
    
    try:
        # 2. 初始化模型
        model = load_model(MODEL_FILE)
        print("✅ 模型載入成功！")
        
        # 3. 測試各種情境
        test_cases = [
            {"age": 20, "hours": 15, "gender": "女", "desc": "一般情況"},
            {"age": None, "hours": 10, "gender": "男", "desc": "缺失年齡測試"},
            {"age": 50, "hours": 2, "gender": "女", "desc": "高齡低讀書時數"}
        ]
        
        print("\n--- 預測服務啟動 ---")
        for case in test_cases:
            res = predict_score(model, case['age'], case['hours'], case['gender'])
            print(f"[{case['desc']}] 輸入: {case['age'] if case['age'] else '缺失'}, {case['hours']}hr, {case['gender']}")
            print(f"預測成績: {res:.2f} 分\n")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")