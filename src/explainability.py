import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

def explain_with_shap(model, X_test, model_name="model", output_dir="../outputs/figures/"):

    print(f"Generating SHAP explanations for {model_name}...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Handle model types
    if hasattr(model, "predict_proba"):
        # Tree-based models
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        except Exception as e:
            print(f"TreeExplainer failed: {e}, switching to KernelExplainer...")
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
            shap_values = explainer.shap_values(shap.sample(X_test, 200))
    else:
        # Fallback for linear models
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)

    # 1 SHAP Summary Plot (Feature Importance)
    plt.title(f"SHAP Summary — {model_name}")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"), dpi=300)
    plt.close()

    # SHAP Bar Plot (Global Importance)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"Global Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_bar.png"), dpi=300)
    plt.close()

    # SHAP Force Plot (Top Sample)
    try:
        sample_idx = np.random.randint(0, len(X_test))
        shap.initjs()
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            shap_values[1][sample_idx] if isinstance(shap_values, list) else shap_values[sample_idx],
            X_test.iloc[sample_idx] if hasattr(X_test, "iloc") else X_test[sample_idx],
            matplotlib=True
        )
        plt.title(f"SHAP Force Plot — Sample #{sample_idx}")
        plt.savefig(os.path.join(output_dir, f"{model_name}_shap_force.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create SHAP force plot: {e}")

    print(f"SHAP plots saved to {output_dir}")

