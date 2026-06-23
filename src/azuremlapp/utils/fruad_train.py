import os
import pandas as pd 
from dotenv import load_dotenv 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

env_path=os.path.join(os.path.dirname(__file__),'..','.env')
load_dotenv(env_path)


def fraud_detection(file_path):
    df=pd.read_csv(file_path)

    df.dropna(inplace=True)
    
    df=pd.get_dummies(df, columns=["location"], drop_first=True)
    X=df.drop("label", axis=1)
    Y=df["label"]

    x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

    clf=DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)

    accuracy=(y_test==y_pred).mean()
    print(f"Accuracy of the model is: {accuracy:.2f}")

    f1=f1_score(y_test, y_pred)
    print(f"F1 score of the model is: {f1:.2f}")

    precision=precision_score(y_test, y_pred)
    print(f"Precision of the model is: {precision:.2f}")

    recall=recall_score(y_test, y_pred)
    print(f"Recall of the model is: {recall:.2f}")

    cm=confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__=="__main__":
    file_path=os.getenv("file_path")
    fraud_detection(file_path)