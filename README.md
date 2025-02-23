Endpoint	Method	Description
/	            GET Health check
/regression	    GET	Get linear regression results
/random_forest	GET	Get Random Forest model predictions

📊 Model Performance
Metric	            Score
Accuracy	        90.1%
Cross-Validation	82.3%

Best Model Parameters:
    {
        "bootstrap": true,
        "max_depth": 15,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 500
    }


📌 Future Improvements
🔹 Hyperparameter Tuning – Test more tree depths & split thresholds.
🔹 Feature Selection – Try removing weak features like Histology.
🔹 Deep Learning Integration – Test Neural Networks for better performance.
🔹 Deploy to Cloud – Host API using FastAPI & AWS/GCP.

📜 License

🔓 MIT License – Free to use & modify.

💬 Contact

👤 Sebastian L. Carmagnola
📧 sebastian@example.com
🔗 GitHub: @sebastianlian

🔥 If you found this useful, drop a ⭐ on GitHub! 🚀