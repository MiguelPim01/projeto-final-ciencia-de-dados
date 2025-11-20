# Soccer Match Result Prediction 🏆⚽

A machine learning project that predicts soccer match outcomes (win, draw, loss) using data scraped from Transfermarkt for Brazilian Serie A (Brasileirão) and Premier League matches.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [License](#license)

## 🎯 Overview

This project aims to predict the outcome of soccer matches from the perspective of the home team using supervised machine learning algorithms. The predictions are categorized into three classes:
- **Vitoria** (Victory): Home team wins
- **Empate** (Draw): Match ends in a draw
- **Derrota** (Defeat): Home team loses

The project was developed as the final assignment for the Introduction to Data Science course at UFES (Universidade Federal do Espírito Santo).

## 📊 Dataset

The dataset was collected through web scraping from [Transfermarkt](https://www.transfermarkt.com.br/) and contains match data from:
- **Brasileirão** (Brazilian Serie A): 2021-2024 seasons
- **Premier League**: 2021-2023 seasons

### Dataset Features (38 columns total)

**Match Information:**
- `capacidade_estadio`: Stadium capacity (int)
- `publico_estadio`: Match attendance (int)
- `ano`: Year of the match (int)
- `fase`: League round/matchday (str)

**Team Statistics (for both home and away teams):**
- `valor_mercado_*`: Total market value of team squad (float)
- `valor_mercado_media_*`: Average market value per player (float)
- `media_idade_*`: Average age of squad (float)
- `jogadores_de_selecao_*`: Number of players in national teams (float)
- `jogadores_de_sub_selecao_*`: Number of players in youth national teams (float)
- `estrangeiros_*`: Number of foreign players (float)
- `posse_*`: Ball possession percentage (int)
- `socios_torcedores_*`: Number of club members/supporters (float)
- `tentativas_*`: Total shots (int)
- `chutes_fora_*`: Shots off target (int)
- `defesa_*`: Goalkeeper saves (int)
- `escanteios_*`: Corner kicks (int)
- `cobrancas_falta_*`: Free kicks (int)
- `faltas_cometidas_*`: Fouls committed (int)
- `impedimentos_*`: Offsides (int)

*Note: Each feature marked with `*` has both `_casa` (home) and `_fora` (away) versions*

## ✨ Features

- **Web Scraping**: Automated data collection from Transfermarkt
- **Data Preprocessing**: Cleaning, feature engineering, and label creation
- **Multiple Classifiers**: Implementation of KNN, Random Forest, and SVM
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and ROC curves
- **Data Visualization**: Visual analysis of results and model performance

## 🔬 Methodology

1. **Data Collection**: Web scraping from Transfermarkt website
2. **Preprocessing**:
   - Label creation based on match results
   - Removal of unnecessary columns (team names, goal counts)
   - Feature normalization for distance-based algorithms
3. **Model Training**: Train/test split (70/30) with cross-validation
4. **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
5. **Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC-ROC

## 🤖 Models Used

### 1. K-Nearest Neighbors (KNN)
- Hyperparameter search: k ∈ [1, 50]
- Features normalized using MinMaxScaler (0-10 range)
- Cross-validation for optimal k selection

### 2. Random Forest Classifier
- Hyperparameters tuned:
  - `n_estimators`: [100, 500, 1000]
  - `max_depth`: [2, 4, 6, 8]
  - `max_features`: ['sqrt', 'log2', 1.0]
  - `max_samples`: [0.2, 0.6, 1.0]

### 3. Support Vector Machine (SVM)
- Hyperparameters tuned:
  - `kernel`: ['rbf', 'linear']
  - `gamma`: [0.01, 0.001, 0.0001]
  - `C`: [1, 10, 100, 1000]
- Probability estimates enabled for ROC curve analysis

## 📈 Results

The models showed moderate performance with the following key findings:

- **Best Overall Performance**: SVM classifier achieved slightly better accuracy
- **Class Imbalance Impact**: Models performed well on "Vitoria" (most frequent) but struggled with "Empate" (least frequent)
- **ROC Curves**: SVM showed the best ROC-AUC scores, especially for Victory prediction
- **Challenge Areas**: All models had difficulty predicting draws, likely due to class imbalance and the inherent unpredictability of drawn matches

Detailed results including confusion matrices, classification reports, and ROC curves are available in the `main.ipynb` notebook.

## 📁 Project Structure

```
projeto-final-ciencia-de-dados/
├── dados/
│   ├── brasileirao_partidas/
│   │   ├── partidas_2021.csv
│   │   ├── partidas_2022.csv
│   │   ├── partidas_2023.csv
│   │   └── partidas_2024.csv
│   └── premierleague_partidas/
│       ├── partidas_2021.csv
│       ├── partidas_2022.csv
│       └── partidas_2023.csv
├── main.ipynb              # Main analysis notebook
├── partidas.csv            # Combined dataset
├── scraper.ipynb           # Web scraping implementation
├── pre_processer.ipynb     # Data preprocessing
├── scraping_test.ipynb     # Scraping tests
├── Especificação.ipynb     # Project specification
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. **Clone the repository:**
```bash
git clone https://github.com/MiguelPim01/projeto-final-ciencia-de-dados.git
cd projeto-final-ciencia-de-dados
```

2. **Open the main notebook:**
```bash
jupyter notebook main.ipynb
```

3. **Run the cells sequentially** to:
   - Load and preprocess the data
   - Train the classification models
   - Evaluate model performance
   - Visualize results

## 👥 Authors

- [Alan Teixeira da Costa](https://github.com/Alan-TC)
  - Matrícula: 2022100831
  - Email: alan.costa@edu.ufes.br

- [Miguel Vieira Machado Pim](https://github.com/MiguelPim01)
  - Matrícula: 2022100894
  - Email: miguel.pim@edu.ufes.br

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data source: [Transfermarkt](https://www.transfermarkt.com.br/)
- Course: Introdução a Ciência de Dados - UFES
- Web scraping code repository: [Scraper-T2-CienciaDeDados](https://github.com/Alan-TC/Scraper-T2-CienciaDeDados) (This repository was also developed by the authors)

## 📚 References

- Scikit-learn Documentation
- Transfermarkt website for match and team statistics
- Course materials from Introduction to Data Science (UFES)

---

**Note**: This project is for educational purposes as part of a university course requirement.
