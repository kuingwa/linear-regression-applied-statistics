import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import t


def generate_data(n_samples: int = 120, random_seed: int = 42):
    """
    Erzeugt einen einfachen Datensatz für die lineare Regression.

    Die Variable x stellt die unabhängige Variable dar, zum Beispiel die Geschwindigkeit.
    Die Variable y stellt die abhängige Variable dar, zum Beispiel die zurückgelegte Strecke.
    """
    rng = np.random.default_rng(random_seed)

    # Unabhängige Variable
    speed = rng.uniform(5, 25, n_samples)

    # Zufälliger Fehler, der die reale Variation widerspiegelt
    noise = rng.normal(0, 3.5, n_samples)

    # Modell: distance = 8 + 3.5 * speed + Fehler
    distance = 8 + 3.5 * speed + noise

    return speed, distance


def fit_linear_regression(x, y):
    """
    Passt ein lineares Regressionsmodell an die Daten an.

    Das Modell hat die Form:
    y = beta_0 + beta_1 * x + epsilon
    """
    x_2d = x.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_2d, y)
    y_pred = model.predict(x_2d)

    return model, y_pred


def evaluate_model(y_true, y_pred):
    """
    Bewertet das Modell mit wichtigen Kennzahlen.

    MAE: Durchschnittlicher absoluter Fehler
    MSE: Durchschnittlicher quadratischer Fehler
    RMSE: Wurzel aus dem MSE
    R²: Bestimmtheitsmaß
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
    Berechnet ein 95%-Konfidenzintervall für die Regressionsgerade.

    Damit wird gezeigt, wie unsicher die Schätzung der Regressionslinie ist.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta

    residuals = y - y_hat
    n, p = X.shape
    sigma2 = np.sum(residuals ** 2) / (n - p)

    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)

    x_grid = np.linspace(x.min(), x.max(), 300)
    X_grid = np.column_stack([np.ones(len(x_grid)), x_grid])

    y_line = X_grid @ beta
    se_pred = np.sqrt(sigma2 * (1 / n + (x_grid - x_mean) ** 2 / sxx))

    t_critical = t.ppf(1 - alpha / 2, df=n - p)
    margin = t_critical * se_pred

    lower = y_line - margin
    upper = y_line + margin

    return x_grid, y_line, lower, upper


def residual_diagnostics(x, y, y_pred):
    """
    Zeigt einfache Residuen-Diagnostiken an.

    Dadurch kann überprüft werden, ob die Annahmen der linearen Regression erfüllt sind.
    """
    residuals = y - y_pred

    print("\n=== Residuen-Diagnostik ===")
    print(f"Mittelwert Residuen : {residuals.mean():.4f}")
    print(f"Standardabweichung : {residuals.std():.4f}")

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
    Zeichnet die Regressionsgerade mit einem 95%-Konfidenzintervall.
    """
    x_grid, y_line, lower, upper = confidence_interval(x, y, alpha=0.05)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, color="navy", alpha=0.7, label="Beobachtete Daten")
    plt.plot(x_grid, y_line, color="darkorange", linewidth=2, label="Regressionsgerade")
    plt.fill_between(x_grid, lower, upper, color="gray", alpha=0.25, label="95% Konfidenzintervall")
    plt.xlabel("Speed")
    plt.ylabel("Distance")
    plt.title("Applied Statistical Models: Linear Regression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("linear_regression_plot.png", dpi=200)
    plt.show()


def main():
    """
    Hauptfunktion des Projekts.
    Sie erzeugt die Daten, passt das Modell an, bewertet es und zeigt die Ergebnisse an.
    """
    x, y = generate_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model, y_train_pred = fit_linear_regression(x_train, y_train)
    y_test_pred = model.predict(x_test.reshape(-1, 1))

    print(f"Intercept : {model.intercept_:.4f}")
    print(f"Slope     : {model.coef_[0]:.4f}")

    evaluate_model(y_test, y_test_pred)

    plot_regression(x, y)
    residual_diagnostics(x_test, y_test, y_test_pred)

    print("\nModellgleichung:")
    print(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x")


if __name__ == "__main__":
    main()
