import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your collected data
data = pd.read_csv(r"D:\New folder\3d_distances.csv")
data2 = pd.read_csv(r"D:\New folder\3d_distances.csv")
data3 = pd.read_csv(r"D:\New folder\3d_distances.csv")

data['label'] = 'squat'
data2['label'] = 'pushup'
data3['label'] = 'jumpingjack'

df = pd.concat([data, data2, data3])
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open('trainer_model.pkl', 'wb'))
