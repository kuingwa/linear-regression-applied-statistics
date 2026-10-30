# Applied Statistical Models: Linear Regression

Dieses Projekt zeigt eine einfache lineare Regression mit Python und scikit-learn.

## Ziel des Projekts

Das Ziel dieses Projekts ist es, die Beziehung zwischen zwei Variablen zu untersuchen und zu modellieren. Dabei wird eine unabhängige Variable verwendet, um eine abhängige Variable vorherzusagen. Die grundlegende Form des Modells lautet:

y = β0 + β1x + ε

Dabei bedeutet:
- y: abhängige Variable
- x: unabhängige Variable
- β0: Achsenabschnitt
- β1: Steigung
- ε: Fehlerterm

## Warum ist dieses Projekt relevant?

Die lineare Regression ist eines der wichtigsten Verfahren der Statistik und des maschinellen Lernens. Sie wird häufig in Forschung, Wirtschaft, Technik und Datenanalyse eingesetzt, um Zusammenhänge zwischen Variablen zu verstehen und Vorhersagen zu treffen.

## Verwendete Technologien

- Python
- NumPy
- Matplotlib
- scikit-learn
- SciPy

## Installation

So wird das Projekt lokal gestartet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ausführung

```bash
python main.py
```

## Ausgabe des Projekts

Beim Start des Programms werden folgende Dateien erzeugt:

- `linear_regression_plot.png`
- `residual_diagnostics.png`

Diese Dateien zeigen:
- die Regressionsgerade
- das Konfidenzintervall
- die Residuenanalyse

## Beschreibung der Analyse

Das Projekt zeigt die folgenden Schritte:

- Datengenerierung
- Modellanpassung
- Berechnung von MAE, MSE, RMSE und R²
- Darstellung der Regressionsgeraden
- Überprüfung der Modellannahmen mit Residuen

## Referenzen

- James et al. – Introduction to Statistical Learning
- scikit-learn Documentation
- GeeksforGeeks – Linear Regression
- StatQuest – Linear Regression
