# AI-Score-Predictor-Pipeline
# 端到端學生成績預測系統 (End-to-End Pipeline)

本專案展示了如何處理具有雜訊與非線性規律的數據，並建立一個自動化、高穩定性的機器學習預測工廠。

## 🚀 技術亮點
* **自動化 Pipeline**: 整合 `StandardScaler`、`SimpleImputer` 與 `OneHotEncoder`，實現從原始數據到預測的一鍵式處理。
* **非線性模型優化**: 對比線性回歸、隨機森林與 MLP 神經網路，最終隨機森林模型在交叉驗證中達到 **R² = 0.9906** 的準確度。
* **模型持久化**: 使用 `joblib` 封裝訓練好的「預訓練模型」，支援斷點部署與即時預測。
* **穩定性驗證**: 實作 5 折交叉驗證 (K-Fold CV)，確保模型在未知數據上的泛化能力。

## 🛠️ 開發工具
* **語言**: Python
* **核心庫**: Scikit-Learn, Pandas, NumPy, Joblib
* **自動化潛力**: 具備整合 Playwright 進行動態數據採集的能力。
