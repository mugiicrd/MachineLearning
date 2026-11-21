import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score

df = pd.read_csv("creditcard.csv")


y = df["Class"]

X = df.iloc[:, 0:30]

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)


params = {
    "objective":"binary:logistic",
    "eval_metric":"aucpr",
    "max_depth":4,
    "eta":0.1,
}
n=50
model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=n)

preds = model.predict(xgb_test)
predicted_classes = (preds >= 0.75).astype(int)
cm = confusion_matrix(y_test, predicted_classes)
print(cm)

accuracy = accuracy_score(y_test, predicted_classes)
print("Accuracy is the: ", accuracy * 100)
print("Preccision is: ", precision_score(y_test, predicted_classes) * 100)
print("Recall is: ", recall_score(y_test, predicted_classes) * 100)