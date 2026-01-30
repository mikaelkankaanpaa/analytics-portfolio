# analytics-portfolio
A small collection of analytics projects (coursework + free time) that demonstrate my ability to turn data into **insight**, **usable tools**, and **decision support** — using Python, ML/AI methods, and Excel/VBA automation. 

## Projects

---

## 1) Premier League Player Performance Analytics (Python)
**Goal:** Analyze Premier League 2023/24 player performance to surface **top performers**, **over/under-performance vs. expected goals (xG)**, and other decision-relevant insights, following a typical data analysis workflow (from data collection and cleaning to analysis and visualization).  

**What I built**
- A cleaned and feature-engineered dataFrame combining multiple CSV sources (single analysis-ready table). 
- Interactive visuals (incl. a small Dash app) to explore goal scoring and performance patterns. 
- Two analytical layers:
  - **Segmentation** (K-Means) to group players by xG vs. actual output.
  - **Predictive ML modeling** (Random Forest) to estimate player ratings from performance metrics. 

**Highlights**
- Advanced, interactive visualizations with Plotly and Dash 
- Clear over/under-performer identification using xG vs. goals
- ML model for predicting FotMob ratings

**Tools:** Python; pandas, NumPy, scikit, Matplotlib, Plotly, Dash 

---

## 2) AI & ML Model Implementations (Python)
**Goal:** Build practical intuition for core AI/ML approaches by implementing and evaluating models across supervised learning, reinforcement learning, and probabilistic methods. 

**What I built**
- **Supervised learning:** Neural network for MNIST image classification with optimizable hyperparameters.
- **Reinforcement learning:** Q-learning agent for OpenAI Gym Taxi-v3 using epsilon-greedy exploration.
- **Probabilistic models:** Naive Bayes and full Bayesian classifiers for MNIST/Fashion-MNIST, including robustness checks with noisy inputs.

**Highlights**
- MNIST classifier achieving >97% test accuracy (best run). 
- RL agent demonstrating learning progress via improved reward/steps over episodes.
- Probabilistic baselines with explicit assumptions and interpretable parameters (means/variances/covariances). 

**Tools:** Python; TensorFlow/Keras, NumPy/SciPy, OpenAI Gym. 

---

## 3) Sales Analytics Dashboard Automation (Excel + VBA)
**NOTE:** Running the file locally requires enabling macros for the file (right-clicking the downloaded file -> Properties -> General: Security and checking 'Unblock'). 

**Goal:** Build an interactive “mock dashboard” for sales analysis and pricing visualization using business simulator data (RealGame). 

**What I built**
- An Excel reporting workbook that turns raw simulator exports into a navigable dashboard view.
- VBA automation to update category-level pricing visuals and streamline reporting workflows.
- Pivot-based analysis with slicers to quickly filter and compare segments. 

**Highlights**
- Demonstrates practical business analytics: structured inputs → automated outputs → decision-friendly visuals.
- Focus on usability (fast filtering, clear visuals, reduced manual work). 

**Tools:** Excel, Pivot Tables, Charts, Slicers, VBA macros. 

---

## Notes
- Projects included are based on public datasets (Kaggle / MNIST / OpenAI Gym) or course-provided simulator data (RealGame). 
- If you're reviewing this as part of an application and want a quick walkthrough, feel free to reach out.
``