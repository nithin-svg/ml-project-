🧠 Cancer Diagnosis Predictor

An AI-powered web application that predicts cancer diagnosis using Machine Learning models. Built with Streamlit, this app allows users to input patient data and get real-time predictions.

🚀 Features
📊 Predicts cancer diagnosis (Positive / Negative)
🤖 Uses multiple ML models:
Logistic Regression
Random Forest
📈 Displays model performance:
Accuracy
Precision
Recall
F1 Score
📉 Visualization of prediction confidence
📋 Dataset preview and statistics
🎨 Clean and interactive UI
🛠️ Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

📦 Dependencies are listed in:
👉

📂 Project Structure
├── streamlit_app.py        # Main application
├── The_Cancer_data_1500_V2.csv  # Dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/cancer-predictor.git
cd cancer-predictor
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
streamlit run streamlit_app.py
📊 How It Works
The dataset is loaded and preprocessed
Data is split into training and testing sets
Two models are trained:
Logistic Regression
Random Forest
The best model is selected automatically based on accuracy
User inputs are taken through UI
Prediction is generated with confidence score

Implementation available in:
👉

🧪 Input Features

The model uses features such as:

Age
Gender
Smoking
Medical indicators (varies from dataset)
📈 Output
Cancer Positive → Warning with confidence level
Cancer Negative → Normal result
