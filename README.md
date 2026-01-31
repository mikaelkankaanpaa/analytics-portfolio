# Data Analytics Portfolio - Mikael Kankaanpää
A small collection of example analytics projects (coursework + free time) that demonstrate my ability to turn data into **insight**, **usable tools**, and **decision support** — using Python, ML/AI methods, and Excel/VBA automation. 

## Projects

---

## 1) Premier League Player Performance Analytics (Python)
**Goal:** Analyze Premier League player performance to surface **top performers**, **over/under-performance**, and other relevant insights, following a typical, end-to-end data analysis workflow and by utilizing AI modeling. 

**What's Inside:**
- End-to-end analysis notebook: data collection > merging > cleaning > feature engineering > EDA > clustering > modeling.   
- Interactive visuals (Plotly) and a dynamic Dash component for exploring top scorers and distributions.   
- Modeling:
  - K-Means segmentation based on player performance.   
  - Predictive Machine Learning (Random Forest model) to estimate player ratings from performance metrics. 

**Links:**
- 📁 View the notebook in GitHub: [1-python-performance-analytics-project](https://github.com/mikaelkankaanpaa/analytics-portfolio/tree/main/1-python-performance-analytics-project/code)
- ▶️ Run the notebook online: [Project 1 - Google Colab](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/1-python-performance-analytics-project/code/analytics-project.ipynb)

**Tools:** Python with pandas, NumPy, scikit, Matplotlib, Plotly, Dash, Jupyter Widgets

---

## 2) AI & ML Model Implementations (Python)
**Goal:** Build practical intuition for core AI/ML approaches by implementing and evaluating models across supervised learning, reinforcement learning, and probabilistic methods.

**What's Inside:**
- **Supervised learning:** Neural network for MNIST image classification with optimizable hyperparameters.
- **Reinforcement learning:** Q-learning agent for OpenAI Gym Taxi-v3 using epsilon-greedy exploration.
- **Probabilistic models:** Naive Bayes and full Bayesian classifiers for MNIST/Fashion-MNIST, including robustness checks with noisy inputs.

**Links**
- 📁 [GitHub folder](https://github.com/mikaelkankaanpaa/analytics-portfolio/tree/main/2-ai-ml-models)
- ▶️ Runnable versions in Google Colab:
  - [Neural network](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/2-ai-ml-models/sequential-neural-network/mlp-image-classification.ipynb)
  - [Q-learning Agent](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/2-ai-ml-models/q-learning/openai-taxi-solved.ipynb)
  - [Bayesian Classifiers](TODO)

**Tools:** Python with TensorFlow/Keras, NumPy/SciPy, OpenAI Gym, Scikit. 

---


## 3) Sales Analytics Dashboard Automation (Excel + VBA)
**NOTE:** Running the file locally requires enabling macros for the file (right-clicking the downloaded file -> Properties -> General: Security and checking 'Unblock'). 

**Goal:** Build an interactive dashboard for sales analysis and pricing visualization using business simulator data (RealGame). 

**What's Inside**
- An Excel reporting workbook that turns raw simulator exports into a navigable dashboard view.
- VBA automation to update category-level pricing visuals and streamline reporting workflows.
- Pivot-based analysis with slicers to quickly filter and compare segments. 

**Tools:** Excel, Pivot Tables, Charts, Slicers, VBA macros. 

---

## Notes
- Projects included are based on public datasets (Kaggle / MNIST / OpenAI Gym) or course-provided simulator data (RealGame). 
- If you're reviewing this as part of an application and want a quick walkthrough, feel free to reach out.