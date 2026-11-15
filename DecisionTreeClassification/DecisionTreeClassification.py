# importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
# 1.Load dataset
df = pd.read_csv("salaries.csv")

# 2.Split dataset into X and y
y = df['salary_more_then_100k']

X = df.drop('salary_more_then_100k',axis='columns')
# 3.Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Identify categorical columns
categorical_features = ['company', 'job', 'degree']

# 5. Create preprocessing step
preprocessor = ColumnTransformer(
    transformers= [
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
)

# 6. Create ML pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])


# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate the model
print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# 9. Make a prediction
new_employee = pd.DataFrame({
    'company': ['Google'],
    'job': ['Data Scientist'],
    'degree': ['Masters']
})

prediction = model.predict(new_employee)
print("Prediction:", prediction)
