# ☁️ QML-Based Cloud Cost Optimization Project

## 📌 Project Overview

This project demonstrates a **Cloud Cost Optimization system** using both **Classical Machine Learning (ML)** and **Quantum Machine Learning (QML)** techniques.
The objective is to predict cloud infrastructure costs based on resource usage and provide **actionable optimization recommendations**.

The project also includes an **interactive Streamlit dashboard** for real-time analysis and visualization.

---

## 🎯 Objectives

* Predict cloud usage cost based on resource consumption
* Compare **Classical ML** and **Quantum ML** approaches
* Analyze key **cost drivers** (CPU, Memory, Storage, Network)
* Provide **optimization recommendations**
* Deploy an interactive web application

---

## 🧠 Technologies & Libraries Used

### Programming Language

* **Python 3.11**

### Libraries

* **NumPy** – Numerical computations
* **Pandas** – Data handling and CSV processing
* **Scikit-Learn** – Classical ML (Random Forest Regressor)
* **PennyLane** – Quantum Machine Learning
* **Matplotlib** – Data visualization
* **Streamlit** – Interactive web dashboard

---

## 🧩 Project Architecture

```
Cloud Usage Data (CSV)
        ↓
Data Preprocessing
        ↓
Classical ML Model (Random Forest)
        ↓
Quantum ML Model (Variational Quantum Circuit)
        ↓
Cost Prediction
        ↓
Cost Driver Analysis
        ↓
Optimization Recommendations
        ↓
Streamlit Dashboard
```

---

## 📁 Project Structure

```
qml-cloud-cost-optimization/
├── app.py                  # Streamlit dashboard
├── notebooks/
│   └── qml_cloud_cost_notebook.ipynb
├── data/
│   └── sample datasets (CSV)
├── requirements.txt        # Deployment dependencies
├── environment.yml         # Local conda environment
├── README.md
```

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Salankara-Dey/qml-cloud-cost-optimization.git
cd qml-cloud-cost-optimization
```

### 2️⃣ Create & Activate Environment (Optional)

```bash
conda env create -f environment.yml
conda activate qml-cloud-cost
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Dataset Description

The dataset represents **cloud resource usage** with the following attributes:

| Feature      | Description                 |
| ------------ | --------------------------- |
| `cpu_hrs`    | CPU usage in hours          |
| `memory_gb`  | Memory usage in GB          |
| `storage_gb` | Storage usage in GB         |
| `network_gb` | Network data transfer in GB |

Both **small and large synthetic datasets** were used to test scalability.

---

## 🔬 Quantum Machine Learning Details

* Implemented using **PennyLane**
* Classical data embedded using **Angle Embedding**
* Variational quantum circuit with entangling layers
* Trained using **gradient descent optimization**
* Executed on a **quantum simulator**

---

## 📈 Results & Observations

* Classical ML models perform better on current datasets
* Quantum ML demonstrates learning capability but is limited by simulation constraints
* Feature importance analysis helps identify major cost drivers
* Optimization recommendations improve decision-making

---

## 🛠 Optimization Recommendations

Based on cost driver analysis:

* **CPU** → Auto-scaling and right-sizing VMs
* **Memory** → Avoid over-provisioning
* **Storage** → Move unused data to cold storage
* **Network** → Reduce cross-region data transfers

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**, allowing public access via a web interface.

---

## 🚀 Future Enhancements

* Integration with real AWS/Azure billing data
* Cost forecasting for future usage
* Hybrid classical-quantum optimization models
* Deployment on cloud platforms (AWS / Azure)
* Real-time monitoring dashboard

---

## 👨‍💻 Author

**Salankara Dey**
B.Tech Computer Science Engineering
KIIT
---

## 📄 License

This project is intended for **academic and learning purposes**.






