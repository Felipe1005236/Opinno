import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os 
import tensorflow.compat.v2 as tf
from tensorflow_text import SentencepieceTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from openpyxl.workbook import Workbook

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
model = hub.load(module_url)
print("module %s loaded" % module_url)


def embed_text(texts, batch_size=512):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        texts_tensor = tf.convert_to_tensor(batch_texts, dtype=tf.string)
        embeddings.append(model(texts_tensor))
    return tf.concat(embeddings, axis=0)

def load_limited_data(file_path, nrows=10000):
    return pd.read_csv(file_path, nrows=nrows)

data1 = load_limited_data('path')
data2 = load_limited_data('path')
data3 = load_limited_data('path')  
data4 = load_limited_data('/app/datos/processed_vestimenta.csv')  
data5 = load_limited_data('/app/datos/processed_EAC.csv')  

processed_data = pd.concat([data1,data2,data3,data4,data5], ignore_index=True)

processed_data = processed_data.dropna(subset=['Lemmas', 'Category'])

processed_data['Lemmas'] = processed_data['Lemmas'].astype(str)

nullos = processed_data[processed_data['Lemmas'].isnull()]
print (nullos[['Original','Lemmas']])

print(processed_data.info())

X = processed_data['Lemmas']
y = processed_data['Category']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_embeddings = embed_text(X_train.tolist())
X_test_embeddings = embed_text(X_test.tolist())

from sklearn.linear_model import LogisticRegression

#Linear regression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_embeddings, y_train)

y_pred_log_reg = log_reg.predict(X_test_embeddings)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))

from sklearn.tree import DecisionTreeClassifier
#Decision tree classifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train_embeddings, y_train)

y_pred_tree = tree_clf.predict(X_test_embeddings)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Neural Network

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_embeddings.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_embeddings, y_train, epochs=10, batch_size=32, validation_split=0.2)

y_pred_nn = np.argmax(nn_model.predict(X_test_embeddings), axis=1)
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn))
print("Accuracy:", accuracy_score(y_test, y_pred_nn))


from sklearn.ensemble import RandomForestClassifier
#Random Forest Classifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_embeddings, y_train)

y_pred_rf = rf_clf.predict(X_test_embeddings)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

df_rl = pd.DataFrame({'Lemmas': X_test, 'Predicted Category': label_encoder.inverse_transform(y_pred_log_reg)})
df_dt = pd.DataFrame({'Lemmas': X_test, 'Predicted Category': label_encoder.inverse_transform(y_pred_tree)})
df_nn = pd.DataFrame({'Lemmas': X_test, 'Predicted Category': label_encoder.inverse_transform(y_pred_nn)})
df_rf = pd.DataFrame({'Lemmas': X_test, 'Predicted Category': label_encoder.inverse_transform(y_pred_rf)})

df_rl.to_excel('/app/datos/logistic_regression.xlsx', index=False)
df_dt.to_excel('/app/datos/decision_tree.xlsx', index=False)
df_nn.to_excel('/app/datos/neural_network.xlsx', index=False)
df_rf.to_excel('/app/datos/random_forest.xlsx', index=False)

data_test_favorita = pd.read_csv('/app/datos/processed_test_favorita.csv')

lem = data_test_favorita['Lemmas']

lem = lem.astype(str)

test_embeddings_favorita = embed_text(lem.tolist())

y_pred_test_favorita = np.argmax(nn_model.predict(test_embeddings_favorita), axis=1)

df_test = pd.DataFrame({'Lemmas': lem, 'Predicted Category': label_encoder.inverse_transform(y_pred_test_favorita)})
df_test.to_excel('/app/datos/test_favorita_cate.xlsx', index=False)