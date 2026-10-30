# Angewandte statistische Modelle: Lineare Regression

## Einleitung

Die lineare Regression ist eines der wichtigsten und am häufigsten verwendeten statistischen Modelle. Sie hilft dabei, Zusammenhänge zwischen einer abhängigen Variable und einer oder mehreren unabhängigen Variablen zu modellieren. In diesem Projekt wird ein einfacher Fall untersucht, bei dem eine einzige unabhängige Variable verwendet wird, um eine abhängige Variable zu erklären.

Das Ziel besteht darin, die Beziehung zwischen den Variablen mathematisch zu beschreiben und zu beurteilen, wie gut das Modell zu den Daten passt. Solche Modelle werden in vielen Bereichen verwendet, zum Beispiel in der Wirtschaft, der Biologie, in der Technik oder im Sozialwesen.

## Forschungsfrage

In diesem Projekt geht es darum, die Beziehung zwischen zwei Größen zu untersuchen. Die Frage lautet: Wie gut lässt sich eine abhängige Variable mit einer linearen Funktion einer unabhängigen Variable erklären?

Wenn wir zum Beispiel die Geschwindigkeit eines Fahrzeugs und die zurückgelegte Strecke betrachten, dann kann die lineare Regression helfen, den Zusammenhang zwischen beiden Größen zu modellieren.

## Ziele der Arbeit

- das Modell mathematisch zu definieren
- die Parameter zu schätzen
- die Güte des Modells zu bewerten
- Fehler und Abweichungen zu analysieren
- die Ergebnisse verständlich zu interpretieren

## Theorie der linearen Regression

Das Grundmodell der linearen Regression hat die Form:

y = β0 + β1x + ε

Dabei gilt:
- y: abhängige Variable
- x: unabhängige Variable
- β0: Achsenabschnitt
- β1: Steigung
- ε: Fehlerterm

Die Parameter β0 und β1 werden mit der Methode der kleinsten Quadrate geschätzt. Ziel dieser Methode ist es, die Summe der quadrierten Abweichungen zwischen beobachteten und vorhergesagten Werten zu minimieren.

## Methodik

Für dieses Projekt wurden künstliche Daten erzeugt, die einen klaren linearen Zusammenhang zwischen x und y zeigen. Anschließend wurden die Daten in Trainings- und Testdaten aufgeteilt. Das Modell wurde auf den Trainingsdaten angepasst und anschließend auf den Testdaten bewertet.

## Modellbeschreibung

Das verwendete Modell lautet:

y = β0 + β1x + ε

Hierbei beschreibt β1, wie stark sich die abhängige Variable ändert, wenn die unabhängige Variable um eine Einheit steigt. β0 gibt den Wert von y an, wenn x gleich null ist.

## Bewertung des Modells

Die Modellgüte wurde mit mehreren Kennzahlen bewertet:

- MAE: durchschnittlicher absoluter Fehler
- MSE: mittlerer quadratischer Fehler
- RMSE: Wurzel aus dem mittleren quadratischen Fehler
- R²: Bestimmtheitsmaß

Diese Kennzahlen geben wichtige Hinweise darauf, wie gut das Modell die Daten erklärt und wie präzise die Vorhersagen sind.

## Residuenanalyse

Die Residuen sind die Unterschiede zwischen beobachteten und vorhergesagten Werten. Ihre Analyse ist wichtig, weil sie zeigt, ob die Annahmen der linearen Regression plausibel sind. Besonders interessant sind dabei:

- Linearität
- Unabhängigkeit der Fehler
- Homoskedastizität
- Normalverteilung der Residuen

Wenn die Residuen zufällig verteilt sind, spricht das dafür, dass das Modell gut passt.

## Ergebnisse und Interpretation

Die Ergebnisse zeigen einen klaren positiven Zusammenhang zwischen der unabhängigen und der abhängigen Variable. Die Regressionsgerade passt gut zu den Daten, und die Kennzahlen bestätigen, dass das Modell insgesamt eine starke Vorhersagekraft besitzt.

Die grafische Darstellung zeigt deutlich, dass die Daten auf einer nahezu linearen Struktur beruhen. Die Entscheidung für ein lineares Modell ist daher sinnvoll und gut nachvollziehbar.

## Diskussion

Die lineare Regression ist ein sehr nützliches und leicht interpretierbares Modell. Es eignet sich besonders für Datensätze mit einem klaren linearen Zusammenhang. Gleichzeitig ist es wichtig, die Annahmen zu prüfen und die Grenzen des Modells zu erkennen. In realen Anwendungen können komplexere Zusammenhänge existieren, dann sind andere Modelle oft besser geeignet.

## Fazit

Das Projekt zeigt, wie eine einfache lineare Regression dabei hilft, einen Zusammenhang zwischen zwei Variablen zu modellieren, zu bewerten und visuell zu interpretieren. Die Kombination aus mathematischer Modellierung, Kennzahlen und grafischer Darstellung macht die Methode besonders verständlich und für die Datenanalyse relevant.

## Referenzen

- James et al. – Introduction to Statistical Learning
- scikit-learn Documentation
- GeeksforGeeks – Linear Regression
- StatQuest – Linear Regression
- Wikipedia – Lineare Regression
