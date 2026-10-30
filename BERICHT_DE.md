# Angewandte statistische Modelle: Lineare Regression

## Einleitung

Die lineare Regression ist eines der wichtigsten und am häufigsten verwendeten statistischen Modelle. Sie ermöglicht es, die Beziehung zwischen einer abhängigen Variable und einer oder mehreren unabhängigen Variablen zu untersuchen und zu beschreiben. Das Hauptziel dieses Modells besteht darin, reale Daten präzise abzubilden, die Modellparameter zu definieren und die Ergebnisse sinnvoll zu interpretieren.

In vielen Bereichen wie Wirtschaft, Biologie, Psychologie oder Soziologie ist die lineare Regression ein zentrales Werkzeug der Datenanalyse. Sie ermöglicht es, Zusammenhänge zu erkennen und zu quantifizieren. Ein typisches Beispiel ist die Beziehung zwischen Lernzeit und Prüfungsleistung oder zwischen Temperatur und Energieverbrauch.

## Forschungsvorhaben

Das zentrale Problem besteht darin, die Beziehung zwischen zwei quantitativen Variablen systematisch zu erfassen und zu beschreiben. Beispielsweise kann untersucht werden, ob eine Veränderung einer unabhängigen Variable mit einer Veränderung der abhängigen Variable zusammenhängt und in welchem Ausmaß diese Beziehung durch eine Gerade beschrieben werden kann.

## Ziele

- Definition des Modells
- Schätzung der Parameter
- Bewertung der Anpassungsgüte
- Analyse der Modellfehler
- Interpretation der Ergebnisse

## Theorie der linearen Regression

Das Grundmodell lautet:

y = β0 + β1x + ε

Dabei gilt:
- y: abhängige Variable
- x: unabhängige Variable
- β0: Intercept
- β1: Steigung
- ε: Fehlerterm

Die Methode der kleinsten Quadrate (OLS) minimiert die Summe der quadrierten Residuen und liefert die beste Gerade für die Daten.

## Methodik

Für dieses Projekt wurden synthetische Daten verwendet, um das Prinzip der linearen Regression klar darzustellen. Anschließend wurde der Datensatz in Trainings- und Testdaten aufgeteilt. Das Modell wurde auf dem Trainingssatz angepasst und auf dem Testsatz evaluiert.

## Modellspezifikation

Das Modell wird als:

y = β0 + β1x + ε

definiert. Die Parameter β0 und β1 werden mit OLS geschätzt. Der Koeffizient β1 beschreibt, wie stark sich y verändert, wenn x um eine Einheit steigt.

## Bewertung des Modells

Zur Bewertung der Modellgüte werden Kennzahlen wie R², MAE, MSE und RMSE verwendet. Ein hoher R²-Wert zeigt an, dass ein großer Teil der Varianz durch das Modell erklärt wird. MSE und RMSE geben zusätzlich Informationen über die Größe der Vorhersagefehler.

## Diagnostik und Annahmen

Zu den wichtigsten Annahmen der linearen Regression gehören:
- Linearität
- Unabhängigkeit der Fehler
- Homoskedastizität
- Normalverteilung der Residuen

Die Residuenanalyse hilft dabei, Muster zu erkennen und die Validität des Modells zu prüfen.

## Ergebnisse und Interpretation

Die Ergebnisse zeigen einen positiven Zusammenhang zwischen den Variablen. Das Modell weist eine solide Vorhersagekraft auf, obwohl die Grenzen linearer Modelle bei komplexeren Strukturen berücksichtigt werden müssen.

## Diskussion und Fazit

Die lineare Regression ist ein leistungsfähiges und leicht interpretierbares Werkzeug. Sie ist besonders nützlich für einfache quantitative Analysen, muss aber gleichzeitig auf ihre Annahmen und Grenzen geprüft werden. Diese Arbeit zeigt ihren fundamentalen Nutzen für die praktische Datenanalyse.

## Referenzen

- James et al. – Introduction to Statistical Learning
- scikit-learn Documentation
- GeeksforGeeks – Linear Regression
- StatQuest – Linear Regression
- Wikipedia – Lineare Regression
