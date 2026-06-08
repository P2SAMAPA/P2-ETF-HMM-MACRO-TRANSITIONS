import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def hmm_macro_score(returns, macro_df, n_states=3):
    """
    Compute probability that the ETF is in the best regime (highest mean return)
    using macro variables to predict the regime.
    """
    # Convert to numpy and clean NaNs
    returns = np.asarray(returns).flatten()
    macro_df = macro_df.copy()
    # Remove rows where any NaN in returns or macro
    mask = ~np.isnan(returns)
    if mask.sum() < 10:
        return 0.0
    returns = returns[mask]
    macro_df = macro_df.iloc[mask]
    if len(returns) < 20:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Cluster returns into regimes
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    states = kmeans.fit_predict(returns.reshape(-1, 1))
    # Compute regime means to identify best regime (highest return)
    means = [np.mean(returns[states == s]) for s in range(n_states)]
    best_state = np.argmax(means)
    # Train logistic regression to predict regime from macro
    X = macro_scaled
    y = states
    # Remove any rows with NaN in X (should not be any)
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]
    if len(y) < 10:
        return 0.0
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    log_reg.fit(X, y)
    # Predict probability of best state for the last macro observation
    last_macro = macro_scaled[-1].reshape(1, -1)
    probs = log_reg.predict_proba(last_macro)[0]
    classes = log_reg.classes_
    best_idx = np.where(classes == best_state)[0]
    if len(best_idx) == 0:
        return 0.0
    prob = probs[best_idx[0]]
    return float(prob)
