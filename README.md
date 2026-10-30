# Applied Statistical Models: Linear Regression

Dieses Projekt zeigt eine einfache lineare Regression mit Python und scikit-learn.

## Ziel
Das Ziel ist die Analyse der Beziehung zwischen einer unabhängigen Variable und einer abhängigen Variable.

Das Modell hat die Form:

y = β0 + β1x + ε

## Technologien
- Python
- NumPy
- Matplotlib
- scikit-learn
- SciPy

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ausführung
```bash
python main.py
```

## Ausgaben
Das Programm erzeugt:
- `linear_regression_plot.png`
- `residual_diagnostics.png`

## Beschreibung
Das Projekt zeigt:
- Modellschätzung
- Bewertung mit R², MAE, MSE und RMSE
- Visualisierung der Regressionsgeraden
- Residuenanalyse
- Anwendung eines grundlegenden statistischen Modells

## Referenzen
- James et al. – Introduction to Statistical Learning
- scikit-learn Documentation
- GeeksforGeeks – Linear Regression
- StatQuest – Linear Regression
