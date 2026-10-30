import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t


# -----------------------------
# 1. Données synthétiques
# -----------------------------
def generate_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(5, 25, 120)
    noise = rng.normal(0, 3.5, 120)
    y = 8 + 3.5 * x + noise
    return x, y


# -----------------------------
# 2. Modèle de régression
# -----------------------------
def fit_regression_model(x, y):
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred


# -----------------------------
# 3. Intervalle de confiance
# -----------------------------
def confidence_interval(x, y, y_pred):
    x_sorted = np.sort(x)
    X_sorted = x_sorted.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_sorted, y)
    y_fit = model.predict(X_sorted)

    x_mean = x.mean()
    sxx = np.sum((x_sorted - x_mean) ** 2)

    sigma2 = np.sum((y - y_pred) ** 2) / (len(x) - 2)
    se_fit = np.sqrt(sigma2 * (1 / len(x) + (x_sorted - x_mean) ** 2 / sxx))

    t_crit = t.ppf(0.975, len(x) - 2)
    lower = y_fit - t_crit * se_fit
    upper = y_fit + t_crit * se_fit

    return x_sorted, y_fit, lower, upper


# -----------------------------
# 4. Figure style slide
# -----------------------------
def create_slide():
    x, y = generate_data()
    model, y_pred = fit_regression_model(x, y)
    x_sorted, y_fit, lower, upper = confidence_interval(x, y, y_pred)

    fig = plt.figure(figsize=(14, 8), facecolor="#0b4d6f")

    # Titre principal
    fig.text(
        0.055, 0.87,
        "Applied Statistical Models: Linear Regression",
        fontsize=33,
        fontweight="bold",
        color="white",
        ha="left",
        va="center"
    )

    # Sous-titres à gauche
    fig.text(
        0.055, 0.62,
        "Subtopics:",
        fontsize=26,
        fontweight="bold",
        color="white",
        ha="left",
        va="top"
    )

    subtopics = [
        "• model specification & design",
        "• fitting & loss functions",
        "• evaluation, explained variance",
        "• assumptions and diagnostics",
        "• scikit-learn, statsmodels",
    ]

    y_pos = 0.56
    for line in subtopics:
        fig.text(0.075, y_pos, line, fontsize=22, color="white", ha="left", va="center")
        y_pos -= 0.08

    # Références à gauche
    fig.text(
        0.055, 0.25,
        "References:",
        fontsize=26,
        fontweight="bold",
        color="white",
        ha="left",
        va="top"
    )

    refs = [
        "• James et al. An Introduction to Statistical Learning (Ch. 2.1, 3.1 and 3.6). Link",
        "• Linear Regression @ GeeksForGeeks Link",
        "• StatQuest: Linear Regression YouTube",
    ]

    y_ref = 0.20
    for line in refs:
        fig.text(0.075, y_ref, line, fontsize=17, color="white", ha="left", va="center")
        y_ref -= 0.06

    # Zone graphique à droite
    ax = fig.add_axes([0.49, 0.18, 0.43, 0.42], facecolor="#f3f3f3")

    # Scatter plot
    ax.scatter(x, y, s=18, color="navy", alpha=0.8)

    # Droite de régression
    ax.plot(x_sorted, y_fit, color="navy", linewidth=2.0)

    # Intervalle de confiance
    ax.fill_between(x_sorted, lower, upper, color="gray", alpha=0.25)

    # Titres et labels
    ax.set_title("95% Confidence Interval", fontsize=14, pad=8, color="black")
    ax.set_xlabel("speed", fontsize=12)
    ax.set_ylabel("demand", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)

    # Texte sous le graphique
    fig.text(
        0.58, 0.09,
        "dependent and independent variables",
        fontsize=20,
        color="white",
        ha="center",
        va="center"
    )

    # Enregistrement
    plt.savefig("linear_regression_slide.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_slide()
