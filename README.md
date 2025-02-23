# **Predictive Oncology Insights**

## **Machine Learning-Based Insights for Oncology**

This project was born out of a desire to provide doctors with a **personal AI companion** that leverages **data analytics and machine learning** to improve breast cancer treatment recommendations. Inspired by **Hacklytics GT 2025**, we aimed to develop a highly accurate **predictive model** to assist clinicians in identifying key risk factors and optimizing treatment decisions. 

Through extensive **machine learning training and fine-tuning**, we developed an **optimized Random Forest model** that achieves **over 90% accuracy** in predicting patient survival outcomes, ensuring reliable and interpretable insights for clinicians.

---

## **What It Does**

This project **analyzes clinical and protein data** to provide insights into breast cancer patient outcomes. The model:

- **Predicts patient survival risk** based on various clinical features.
- **Identifies key contributing factors** using feature importance analysis.
- **Enhances clinical decision-making** by offering data-driven insights.

---

## **⚙️ How We Built It**

We focused on three **major components**:

### **Machine Learning Model Optimization**

- Developed an **advanced Random Forest Classifier**, fine-tuned via **GridSearchCV**.
- Engineered new features like **protein averages, histology-stage interaction, and age group bins** to improve accuracy.
- Applied **SMOTE (Synthetic Minority Oversampling)** to balance the dataset and prevent bias.
- Achieved **90%+ accuracy**, with cross-validation scores exceeding **82%**.
- Leveraged **feature importance analysis** to determine the most critical predictors in breast cancer outcomes.

### **Backend Development & Model Training**

- Designed the **Flask API** to serve machine learning predictions efficiently.
- Developed robust **data preprocessing pipelines** to handle missing values, outliers, and categorical variables.
- Integrated **MongoDB** to store and retrieve processed patient data for future enhancements.
- Ensured scalability by structuring the backend to accommodate additional models and real-time predictions.
- Implemented **RESTful API endpoints** for seamless integration with the frontend.

### **Beautiful Console Logs & Debugging Tools**

- Implemented **formatted, color-coded console outputs** for better readability using `colorama`.
- Added **structured logging** to ensure that **model performance metrics** are clearly displayed.
- Created detailed debugging tools to track model performance over different hyperparameter configurations.

---

## **Challenges We Overcame**

- **Balancing the Dataset:** Implementing **SMOTE** was crucial to handle class imbalances in patient outcomes.
- **Hyperparameter Tuning:** Optimizing **n_estimators, max_depth, and feature selection** was an iterative process.
- **Deploying the Backend API:** Ensuring smooth integration between **machine learning components and the frontend** required extensive testing.
- **Feature Engineering:** Experimented with multiple new features to determine the best combination for improving accuracy.

---

## **Accomplishments We're Proud Of**

- **Optimized Machine Learning Model:** Achieved **90%+ accuracy** with robust cross-validation performance.
- **Feature Engineering Success:** Created new features that significantly improved prediction accuracy.
- **Backend API Implementation:** Developed a seamless **Flask-based backend** to support real-time ML predictions.
- **Beautiful Console Logging:** Developed a clear, structured debugging process for model interpretation.
- **Interdisciplinary Collaboration:** Brought together **machine learning, software development, and UI design** to enhance the final product.

---

## **What We Learned**

- **Advanced ML Techniques:** Fine-tuning Random Forest models significantly impacts accuracy.
- **Backend Scalability:** Designing a scalable API ensures seamless model deployment.
- **Readable Debugging:** Clear console outputs improve workflow efficiency and model interpretability.
- **UI/UX for Machine Learning:** Making complex ML results accessible through intuitive visualizations enhances usability.
- **Importance of Data Cleaning:** Handling missing values and normalizing data improved model reliability.

---

## **What’s Next?**

- **Enhance the Backend:** Optimize API response times and introduce **real-time data updates**.
- **Deploy via Streamlit:** Provide a **fully interactive web dashboard** for clinicians.

---

## ** Backend Development Video Explanation**

[![Watch the Backend Explanation](https://img.youtube.com/vi/Nr8mY4-Lj20/0.jpg)](https://youtu.be/Nr8mY4-Lj20)

---

## **License**

This project is licensed under a proprietary license. The **machine learning model and training pipeline** developed in this project are **intellectual property** and may **not be copied, modified, or used** without explicit permission from the project maintainers. If you wish to use or reference any part of this model, please **contact us for authorization.**

---

## **🛠 Built With**

- **Flask**
- **Kaggle**
- **Matplotlib**
- **MongoDB**
- **Next.js**
- **NumPy**
- **Pandas**
- **Plotly**
- **Python**
- **Scikit-learn**
- **Streamlit**
- **TypeScript**

  
---
