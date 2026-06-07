import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# 2. Mengatur nama eksperimen agar rapi di dashboard
mlflow.set_experiment("Latihan_Diabetes_Prediction_nontuning") 
# Load data hasil preprocessing
df = pd.read_csv(r'C:\Users\achma\Downloads\ujian3\preprocessing\diabetes_preprocessing.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan Autologging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic"):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model dilatih dengan autolog.")