import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define X & Y
X = None 
y = None

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# define pipeline
pipe = Pipeline([
    None 
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (Train - Test): {acc:.3f}")
joblib.dump(pipe, 'traffic_model.pkl')

meta = {
    'predictive': (float(X_train),
                       float(X_train)),
    'default': float(X_train),
    'classes': pipe.classes_.tolist(),
    'test_accuracy': float(acc)                    
}

pd.Series(meta).to_json('traffic_meta.json')
