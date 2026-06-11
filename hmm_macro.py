import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def hmm_macro_score(returns, macro_df, n_states=3):
    """
    Compute probability that the ETF is in the best regime (highest mean return)
    using macro variables to predict the regime.
    """
    # Convert to numpy and clean NaNs
    returns = np.asarray(returns).flatten()
    macro_df = macro_df.copy()
    
    # Remove rows where any NaN in returns
    mask = ~np.isnan(returns)
    if mask.sum() < 10:
        return 0.0
    
    returns = returns[mask]
    macro_df = macro_df.iloc[mask]
    
    if len(returns) < 20:
        return 0.0
    
    # Handle NaN in macro columns - fill with column mean
    if macro_df.isnull().any().any():
        # Fill forward then backward for time series
        macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
    
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    
    # Impute any remaining NaNs
    imputer = SimpleImputer(strategy='mean')
    macro_scaled = imputer.fit_transform(macro_scaled)
    
    # Final safety check
    if np.any(np.isnan(macro_scaled)):
        macro_scaled = np.nan_to_num(macro_scaled, nan=0.0)
    
    # Cluster returns into regimes
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    states = kmeans.fit_predict(returns.reshape(-1, 1))
    
    # Compute regime means to identify best regime (highest return)
    means = [np.mean(returns[states == s]) for s in range(n_states)]
    best_state = np.argmax(means)
    
    # Train logistic regression to predict regime from macro
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(macro_scaled, states)
    
    # Predict probability of best state for the last macro observation
    last_macro = macro_scaled[-1].reshape(1, -1)
    probs = log_reg.predict_proba(last_macro)[0]
    
    # Find probability for best_state
    class_list = list(log_reg.classes_)
    if best_state not in class_list:
        return 0.0
    
    prob = probs[class_list.index(best_state)]
    return float(prob)
