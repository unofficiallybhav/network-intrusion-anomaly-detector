import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def save_confusion_matrix(y_true, y_pred, model_name,filename,filepath=r"C:\Users\hp\OneDrive\Desktop\Python\Machine Learning\project\outputs\figures"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"[✔] Saved confusion matrix for {model_name} at: {filepath}")

def save_plot(fig, filepath=r"C:\Users\hp\OneDrive\Desktop\Python\Machine Learning\project\outputs\figures"):
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✔] Figure saved at: {filepath}")

def save_model(model,filepath=r"C:\Users\hp\OneDrive\Desktop\Python\Machine Learning\project\outputs\figures"):
    joblib.dump(model, filepath)
    print(f"[✔] Model saved at: {filepath}")

def load_model(filepath=r"C:\Users\hp\OneDrive\Desktop\Python\Machine Learning\project\outputs\figures"):
    model = joblib.load(filepath)
    print(f"[✔] Loaded model from: {filepath}")
    return model