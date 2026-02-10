from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def evaluate(model, X, y):

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm
    }
