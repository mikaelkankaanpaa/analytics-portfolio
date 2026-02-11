# Data Analytics & Technologies Portfolio - Mikael Kankaanpää
A small collection of example projects that demonstrate my ability to leverage tech and turn data into **insight**, **usable tools**, and **decision support** — using Python, ML/AI methods, Excel/VBA automation, SPSS and n8n.

## Projects:

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
- 📁 View the notebook in GitHub: [1-python-performance-analytics-project](https://github.com/mikaelkankaanpaa/analytics-portfolio/blob/main/1-python-performance-analytics-project/code/analytics-project.ipynb)
- ▶️ Run the notebook online: [Project 1 - Google Colab](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/1-python-performance-analytics-project/code/analytics-project.ipynb)

**Tools:** Python with pandas, NumPy, scikit, Matplotlib, Plotly, Dash, Jupyter Widgets <br>
**Demonstrated Skills:**  
Data analysis workflow, data wrangling and feature engineering, exploratory data analysis (EDA), clustering and predictive modeling, model evaluation, interactive data visualization, dashboard prototyping, translating analytics into performance insights

---

## 2) Dynamic Sales Analytics Dashboard (Excel + VBA)
**Goal:** Build an interactive, lightweight Excel dashboard for competitor and sales analysis, using data from a gamified business simulation<sup>1</sup> [RealGame](https://www.realgame.fi/).

**What's Inside**
- An Excel reporting workbook that turns raw data exports into a navigable dashboard view.
- VBA automation to update segment-level KPIs, perfomance and pricing visuals and streamline reporting workflows.
- Concrete decision-support tools driven from a large set of transaction data.

**Links**
- 📁 GitHub folder: [2-excel-VBA-dashboard](https://github.com/mikaelkankaanpaa/analytics-portfolio/tree/main/2-excel-VBA-dashboard)
- ⬇️ [Click here to download the Excel-workbook](https://github.com/mikaelkankaanpaa/analytics-portfolio/raw/refs/heads/main/2-excel-VBA-dashboard/custom-vba-example.xlsm) *[Note, that full functionality requires enabling macros for the file: right-click the downloaded file -> Properties -> General: Security -> select 'Unblock']*

**Tools:** Excel, VBA macros, Pivot Tables, Charts, Functions <br>
**Demonstrated Skills:**  
Excel-based analytics, VBA scripting and workflow automation, pivot-based analysis, data visualization for decision support, raw data transformations

**<sup>1</sup>** *This is a small excerpt from analysis associated with the course "Business Decisions and Market Analytics", which was structured around the RealGame simulation.* 

---

## 3) AI & ML Model Implementations (Python)
**Goal:** Build practical intuition for core AI/ML approaches by implementing and evaluating models across supervised learning, reinforcement learning, and probabilistic methods.

**What's Inside:**
- **Supervised learning:** Neural network for MNIST image classification with optimizable hyperparameters.
- **Reinforcement learning:** Q-learning agent for OpenAI Gym Taxi-v3 using epsilon-greedy exploration.
- **Probabilistic models:** Naive Bayes and full Bayesian classifiers for MNIST/Fashion-MNIST, including robustness checks with noisy inputs.

**Links**
- 📁 GitHub folder: [3-ai-ml-models](https://github.com/mikaelkankaanpaa/analytics-portfolio/tree/main/3-ai-ml-models)
- ▶️ Run the scripts online in Google Colab:
  - [Neural network](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/3-ai-ml-models/sequential-neural-network/mlp-image-classification.ipynb)
  - [Q-learning Agent](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/3-ai-ml-models/q-learning/openai-taxi-solved.ipynb)
  - Bayesian Classifiers:
    - [Full Bayesian](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/3-ai-ml-models/bayesian-estimators/full-bayes-estimator.ipynb)
    - [Naive Bayesian](https://colab.research.google.com/github/mikaelkankaanpaa/analytics-portfolio/blob/main/3-ai-ml-models/bayesian-estimators/naive-bayes-estimator.ipynb)

**Tools:** Python with TensorFlow/Keras, NumPy/SciPy, OpenAI Gym, Scikit.<br>
**Demonstrated Skills:**  
Supervised and reinforcement learning implementation, neural network development and tuning, probabilistic modeling, algorithm evaluation and benchmarking, experimentation and robustness testing, working with ML frameworks, translating theoretical ML concepts into practical implementations

---

## 4) Health Care Cost Drivers Analysis (IBM SPSS)

**Goal:**  
Identify and quantify how lifestyle factors influence health care costs by building and validating a statistical regression model, demonstrating structured analytical problem-solving and statistical rigor.

**What's Inside:**
- End-to-end statistical analysis workflow including data preprocessing, transformation, exploratory correlation analysis, and model diagnostics.
- Multiple linear regression model estimating the impact of behavioral variables on annual health care costs.
- Full assumption validation to ensure model reliability:
  - Outlier detection and treatment via log transformations.
  - Linearity, homoscedasticity, and residual normality testing.
  - Multicollinearity evaluation using VIF metrics.
- Interpretation of statistical outputs into practical decision-support insights.

**Links:**
- 📁 GitHub folder: [4-SPSS-data-analysis](https://github.com/mikaelkankaanpaa/analytics-portfolio/tree/main/4-SPSS-data-analysis)
- ⬇️ [Click here to download the report as PDF](https://raw.githubusercontent.com/mikaelkankaanpaa/analytics-portfolio/main/4-SPSS-data-analysis/report/spss-data-analysis-project-report.pdf)

**Tools:** IBM SPSS <br>
**Demonstrated Skills:**  
Statistical regression modeling, diagnostic testing and assumption validation, data preprocessing and transformation, correlation analysis, statistical interpretation and reporting, reporting

---

## 5) News & Weather AI Assistant (n8n)

**Goal:** Build a chat-based AI agent that can retrieve real-time Finnish and global news as well as weather data using tool-based automation in n8n.

**What's Inside:**
- A n8n workflow with a LLM-chat UI (Google Gemini)
- Defined tools for fetching news from multiple RSS feeds and for retrieving weather data through API
- Dynamic tool selection based on user chat inputs
- Custom system instructions and prompt design

**Links:**
- 📁 GitHub folder: [5-n8n-ai-agent](https://github.com/mikaelkankaanpaa/analytics-portfolio/tree/main/5-n8n-ai-agent)
- 🌐 Try chatting with the agent online: [n8n Webhook](https://mikaelkankaanpaa.app.n8n.cloud/webhook/e5616171-e3b5-4c39-81d4-67409f9fa60a/chat) 

**Tools:** n8n, Google Gemini API, Openmeteo API, RSS feeds

**Demonstrated Skills:**  
AI agent design, workflow automation, low-code automation, prompt engineering, API integration, human-in-the-loop system design

---

## Notes
- Only projects based on public datasets (Kaggle / MNIST / OpenAI Gym / GSS) or course-provided simulator data (RealGame) are included in this portfolio. 
- If you're reviewing this as part of an application and want a quick walkthrough, feel free to reach out.
