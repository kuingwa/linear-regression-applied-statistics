import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import t


# ----------------------------------------------------------
# Dieses Projekt zeigt eine kleine, aber klare Anwendung der
# linearen Regression mit Python. Das Ziel ist es, eine
# Beziehung zwischen zwei Variablen zu modellieren und dabei
# die wichtigsten Kennzahlen und Grafiken zu berechnen.
# ----------------------------------------------------------


def generate_data(n_samples: int = 120, random_seed: int = 42):
    """
    Diese Funktion erzeugt einen künstlichen Datensatz für die lineare Regression.

    Die unabhängige Variable x steht zum Beispiel für die Geschwindigkeit.
    Die abhängige Variable y steht zum Beispiel für die zurückgelegte Strecke.
    Die Daten enthalten einen klaren linearen Zusammenhang und einen zufälligen Fehler,
    damit sie realistisch wirken.
    """
    rng = np.random.default_rng(random_seed)

    # Die unabhängige Variable wird zufällig zwischen 5 und 25 erzeugt.
    speed = rng.uniform(5, 25, n_samples)

    # Ein kleiner Zufallsfehler simuliert reale Messabweichungen.
    noise = rng.normal(0, 3.5, n_samples)

    # Das wahre Modell ist: distance = 8 + 3.5 * speed + Fehler
    distance = 8 + 3.5 * speed + noise

    return speed, distance


def fit_linear_regression(x, y):
    """
    Diese Funktion passt ein lineares Modell an die Daten an.

    Das Modell hat die Form:
    y = beta_0 + beta_1 * x + epsilon

    beta_0 ist der Achsenabschnitt (Intercept), beta_1 ist die Steigung.
    """
    x_2d = x.reshape(-1, 1)

    # Wir verwenden die Standard-Linear-Regression aus scikit-learn.
    model = LinearRegression()

    # Das Modell wird mit den Trainingsdaten angepasst.
    model.fit(x_2d, y)

    # Vorhersagen für dieselben Daten werden berechnet.
    y_pred = model.predict(x_2d)

    return model, y_pred


def evaluate_model(y_true, y_pred):
    """
    Diese Funktion bewertet die Qualität des Modells mit wichtigen Kennzahlen.

    MAE  = durchschnittlicher absoluter Fehler
    MSE  = mittlerer quadratischer Fehler
    RMSE = Quadratwurzel aus MSE
    R²   = Bestimmtheitsmaß, misst, wie gut das Modell die Daten erklärt
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n=== Modellbewertung ===")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def confidence_interval(x, y, alpha: float = 0.05):
    """
    Diese Funktion berechnet ein 95%-Konfidenzintervall für die Regressionsgerade.

    Dadurch wird sichtbar, wie unsicher die geschätzte Gerade ist.
    Mit diesem Intervall können wir die Streuung der Schätzung beurteilen.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Das Modell wird als y = beta_0 + beta_1 * x geschrieben.
    X = np.column_stack([np.ones(len(x)), x])

    # Die kleinste-Quadrate-Schätzung liefert die Parameter beta_0 und beta_1.
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta

    # Die Residuen zeigen den Fehler zwischen beobachtet und vorhergesagt.
    residuals = y - y_hat
    n, p = X.shape
    sigma2 = np.sum(residuals ** 2) / (n - p)

    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)

    # Wir erzeugen einen feinen Raster für die Gerade, um das Intervall sauber darzustellen.
    x_grid = np.linspace(x.min(), x.max(), 300)
    X_grid = np.column_stack([np.ones(len(x_grid)), x_grid])

    y_line = X_grid @ beta
    se_pred = np.sqrt(sigma2 * (1 / n + (x_grid - x_mean) ** 2 / sxx))

    # Kritischer t-Wert für ein 95%-Intervall.
    t_critical = t.ppf(1 - alpha / 2, df=n - p)
    margin = t_critical * se_pred

    lower = y_line - margin
    upper = y_line + margin

    return x_grid, y_line, lower, upper


def residual_diagnostics(x, y, y_pred):
    """
    Diese Funktion erstellt eine einfache Residuenanalyse.

    Damit kann geprüft werden, ob die Residuen zufällig verteilt sind,
    was eine wichtige Annahme der linearen Regression ist.
    """
    residuals = y - y_pred

    print("\n=== Residuen-Diagnostik ===")
    print(f"Mittelwert Residuen : {residuals.mean():.4f}")
    print(f"Standardabweichung : {residuals.std():.4f}")

    # Wir teilen die Analyse in zwei Teilplots auf:
    # links: Residuen gegen Vorhersagen
    # rechts: Histogramm der Residuen
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_pred, residuals, color="royalblue", alpha=0.7)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residuen vs. vorhergesagte Werte")
    axes[0].set_xlabel("Vorhergesagte Werte")
    axes[0].set_ylabel("Residuen")

    axes[1].hist(residuals, bins=15, color="cornflowerblue", edgecolor="black")
    axes[1].set_title("Verteilung der Residuen")
    axes[1].set_xlabel("Residuen")
    axes[1].set_ylabel("Häufigkeit")

    plt.tight_layout()
    plt.savefig("residual_diagnostics.png", dpi=200)
    plt.show()


def plot_regression(x, y):
    """
    Diese Funktion zeichnet die Regressionsgerade und das Konfidenzintervall.

    So wird visuell deutlich, wie gut die Gerade zu den beobachteten Punkten passt.
    """
    x_grid, y_line, lower, upper = confidence_interval(x, y, alpha=0.05)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    plt.figure(figsize=(9, 6))

    # Die beobachteten Datenpunkte werden als Streudiagramm dargestellt.
    plt.scatter(x, y, color="navy", alpha=0.7, label="Beobachtete Daten")

    # Die Regressionsgerade wird als blaue Linie eingezeichnet.
    plt.plot(x_grid, y_line, color="darkorange", linewidth=2, label="Regressionsgerade")

    # Die graue Fläche zeigt das 95%-Konfidenzintervall an.
    plt.fill_between(x_grid, lower, upper, color="gray", alpha=0.25, label="95% Konfidenzintervall")

    plt.xlabel("Speed")
    plt.ylabel("Distance")
    plt.title("Applied Statistical Models: Linear Regression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Das Diagramm wird als Datei gespeichert, damit es auch außerhalb der Sitzung genutzt werden kann.
    plt.savefig("linear_regression_plot.png", dpi=200)
    plt.show()


def main():
    """
    Hauptfunktion des Projekts.

    Hier werden die Daten erzeugt, das Modell geschätzt, bewertet und grafisch dargestellt.
    """
    x, y = generate_data()

    # Aufteilung in Trainings- und Testdaten für eine realistische Bewertung.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Das Modell wird auf den Trainingsdaten angepasst.
    model, y_train_pred = fit_linear_regression(x_train, y_train)

    # Für die Bewertung werden Vorhersagen für den Testdatensatz erstellt.
    y_test_pred = model.predict(x_test.reshape(-1, 1))

    print(f"Intercept : {model.intercept_:.4f}")
    print(f"Slope     : {model.coef_[0]:.4f}")

    # Wir bewerten die Qualität des Modells mit Kennzahlen.
    evaluate_model(y_test, y_test_pred)

    # Die Regressionsgerade wird grafisch dargestellt.
    plot_regression(x, y)

    # Die Residuenanalyse hilft bei der Überprüfung der Modellannahmen.
    residual_diagnostics(x_test, y_test, y_test_pred)

    print("\nModellgleichung:")
    print(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x")


if __name__ == "__main__":
    main()
