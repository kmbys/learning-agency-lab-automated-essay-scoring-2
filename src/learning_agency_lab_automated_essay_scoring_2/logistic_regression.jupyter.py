# %%
# load data
import pandas as pd

data = pd.read_csv('data/train.csv')
X_text = data['full_text']
y = data['score']

test = pd.read_csv('data/test.csv')
X_test_text = test['full_text']

# %%
data.head()

# %%
test.head()

# %%
# preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)
X_test = vectorizer.transform(X_test_text)

# %%
# split data
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
# valid
from sklearn.metrics import accuracy_score, cohen_kappa_score

y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
kappa = cohen_kappa_score(y_valid, y_pred, weights='quadratic')
print(f'Accuracy: {accuracy:.2f}')
print(f'Quadratic Weight Kappa: {kappa:.2f}')

# %%
# predict
model.predict(vectorizer.transform(['This is a sample essay for prediction.']))[0]

# %%
# submit
y_test = model.predict(X_test)
submission = pd.DataFrame({
    'essay_id': test['essay_id'],
    'score': y_test[0],
})

# %%
submission

# %%
submission.to_csv('submission.csv', index=False)
