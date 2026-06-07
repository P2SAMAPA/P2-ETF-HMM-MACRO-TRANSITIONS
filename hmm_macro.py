import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

class HMMMacro:
    def __init__(self, n_states=3, max_iter=50, tol=1e-4):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _log_likelihood(self, observations, gamma):
        # gamma: posterior probabilities (T x n_states)
        ll = 0.0
        for t in range(len(observations)):
            for s in range(self.n_states):
                if gamma[t, s] > 0:
                    ll += gamma[t, s] * np.log(gamma[t, s] + 1e-12)
        return ll

    def fit(self, returns, macro_df):
        """
        Fit HMM with macro‑dependent transitions using EM.
        returns: 1D array of returns (T)
        macro_df: DataFrame of macro variables aligned with returns (T x M)
        """
        T = len(returns)
        if T < 10:
            return None
        M = macro_df.shape[1]
        # Standardise macro
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
        # Initialise parameters
        # Emission: Gaussian with state‑specific mean and variance
        mu = np.linspace(-0.01, 0.01, self.n_states)  # initial means
        sigma = np.ones(self.n_states) * 0.02
        # Transition log‑linear coefficients: for each from‑state i, we have a vector [alpha_i0, alpha_i1,..., alpha_iM]
        # The transition to to‑state j: logit = beta_ij0 + beta_ij1 * macro1 + ...
        # We'll store a matrix of size (n_states, n_states, M+1)
        # For simplicity, we use a separate multinomial logistic for each from‑state.
        # Parameter vector: for each i, (n_states * (M+1)) parameters. We'll flatten.
        # To avoid over‑parameterisation, we set one reference state (e.g., j=0) coefficients to zero.
        # Thus for each i, we have (n_states-1)*(M+1) free parameters.
        # We'll use a simple initial: all transitions equal (1/n_states) independent of macro.
        # That corresponds to zero coefficients for macro.
        # For each from‑state i, we have coefficients for j=1..n_states-1 (excluding reference).
        # Let param_ij = [c0, c1, ..., cM] for j>0.
        # Total parameters per i: (n_states-1)*(M+1)
        # We'll store as a dict or array.
        # Instead of full EM with M‑step optimisation, we'll use a simplified approach:
        # We'll first estimate an ordinary HMM (constant transition) using Baum‑Welch,
        # then use the resulting soft assignments to estimate macro effects via multinomial regression.
        # This is not full EM but works and is stable.

        # Step 1: Fit ordinary HMM (constant transitions) using Baum‑Welch
        # Use hmmlearn if available; else implement simple version.
        # We'll implement a simple Baum‑Welch for Gaussian HMM.
        # Initial guesses
        from scipy.stats import norm
        # Forward‑backward algorithm
        def forward(obs, A, mu, sigma):
            T = len(obs)
            K = self.n_states
            alpha = np.zeros((T, K))
            # Initial distribution: uniform
            pi = np.ones(K) / K
            for s in range(K):
                alpha[0, s] = pi[s] * norm.pdf(obs[0], mu[s], sigma[s])
            alpha[0] /= np.sum(alpha[0])
            for t in range(1, T):
                for s in range(K):
                    alpha[t, s] = norm.pdf(obs[t], mu[s], sigma[s]) * np.sum(alpha[t-1] * A[:, s])
                alpha[t] /= np.sum(alpha[t])
            return alpha

        def backward(obs, A, mu, sigma):
            T = len(obs)
            K = self.n_states
            beta = np.zeros((T, K))
            beta[-1] = 1.0
            for t in range(T-2, -1, -1):
                for s in range(K):
                    beta[t, s] = np.sum(A[s, :] * norm.pdf(obs[t+1], mu, sigma) * beta[t+1])
                beta[t] /= np.sum(beta[t])
            return beta

        # Initial parameters
        A = np.ones((self.n_states, self.n_states)) / self.n_states
        mu = np.random.randn(self.n_states) * 0.01
        sigma = np.ones(self.n_states) * 0.02
        for _ in range(self.max_iter):
            # E‑step: forward‑backward
            alpha = forward(returns, A, mu, sigma)
            beta = backward(returns, A, mu, sigma)
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            # M‑step for emissions
            for s in range(self.n_states):
                total = np.sum(gamma[:, s])
                mu[s] = np.sum(gamma[:, s] * returns) / total
                sigma[s] = np.sqrt(np.sum(gamma[:, s] * (returns - mu[s])**2) / total)
            # M‑step for transitions (simple: empirical count)
            xi = np.zeros((self.n_states, self.n_states))
            for t in range(len(returns)-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i, j] += gamma[t, i] * A[i, j] * norm.pdf(returns[t+1], mu[j], sigma[j]) * beta[t+1, j]
            A = xi / np.sum(xi, axis=1, keepdims=True)

        # After obtaining ordinary HMM, we now model transition probabilities as functions of macro.
        # We'll use the smoothed state probabilities gamma to fit a multinomial logit for each from‑state.
        # For each time t, we have gamma[t, i] as soft assignment of state at time t.
        # We want to predict the next state (t+1) given macro at time t.
        # We'll create weighted regression: for each from‑state i, we have weights gamma[t, i].
        # The target is the state at t+1 (we can use softmax regression with weighted samples).
        # This is a weighted multinomial logistic regression.
        # We'll implement with scipy.optimize for each i.
        from sklearn.linear_model import LogisticRegression
        # Prepare data for regression
        X = macro_scaled[:-1]  # macro at time t
        # For each from‑state i, we create a dataset where each sample (t) has weight gamma[t, i]
        # and target = next state (t+1)
        y = np.argmax(gamma[1:], axis=1)  # most likely next state
        # But better: use soft targets: we can use the distribution gamma[t+1] as probabilities.
        # For simplicity, we'll use the hard assignment y above.
        # We'll fit a multinomial logistic regression with sample weights.
        # However, sklearn's LogisticRegression does not support multi‑class with sample weights? It does.
        log_regs = []
        for i in range(self.n_states):
            weights = gamma[:-1, i]  # weight for each time step where we are in state i
            # Remove near‑zero weights to avoid numerical issues
            valid = weights > 1e-6
            if np.sum(valid) < 10:
                # fallback: uniform transitions
                log_reg = None
            else:
                X_i = X[valid]
                y_i = y[valid]
                w_i = weights[valid]
                log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
                log_reg.fit(X_i, y_i, sample_weight=w_i)
            log_regs.append(log_reg)
        # Store log_regs for each from‑state
        self.log_regs = log_regs
        self.scaler = scaler
        self.emission_mu = mu
        self.emission_sigma = sigma
        # For scoring, compute filtered probability of the best state at the last time step given macro
        # We'll do a forward pass using macro‑dependent transitions.
        # Use the current macro to compute transition matrix at each step.
        T = len(returns)
        alpha = np.zeros((T, self.n_states))
        # Initial distribution uniform
        pi = np.ones(self.n_states) / self.n_states
        for s in range(self.n_states):
            alpha[0, s] = pi[s] * norm.pdf(returns[0], self.emission_mu[s], self.emission_sigma[s])
        alpha[0] /= np.sum(alpha[0])
        for t in range(1, T):
            # Compute transition matrix A_t dependent on macro at t-1
            macro_t = macro_scaled[t-1].reshape(1, -1)
            A_t = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if self.log_regs[i] is not None:
                    # Predict transition probabilities from state i
                    probs = self.log_regs[i].predict_proba(macro_t)[0]
                    # Map to states (the order of classes may not be sequential)
                    # We need to map to all states. Use the classes_ attribute
                    classes = self.log_regs[i].classes_
                    for idx, s in enumerate(classes):
                        A_t[i, s] = probs[idx]
                else:
                    # fallback uniform
                    A_t[i, :] = 1.0 / self.n_states
            for s in range(self.n_states):
                alpha[t, s] = norm.pdf(returns[t], self.emission_mu[s], self.emission_sigma[s]) * np.sum(alpha[t-1] * A_t[:, s])
            alpha[t] /= np.sum(alpha[t])
        self.alpha = alpha
        return self

    def score(self, returns, macro_df):
        """
        Returns the filtered probability of the highest‑mean regime at the last time step.
        """
        if self.log_regs is None:
            return 0.0
        T = len(returns)
        if T != len(macro_df):
            min_len = min(T, len(macro_df))
            returns = returns[:min_len]
            macro_df = macro_df.iloc[:min_len]
        macro_scaled = self.scaler.transform(macro_df)
        # Initial distribution uniform
        pi = np.ones(self.n_states) / self.n_states
        alpha = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            alpha[0, s] = pi[s] * norm.pdf(returns[0], self.emission_mu[s], self.emission_sigma[s])
        alpha[0] /= np.sum(alpha[0])
        for t in range(1, T):
            macro_t = macro_scaled[t-1].reshape(1, -1)
            A_t = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if self.log_regs[i] is not None:
                    probs = self.log_regs[i].predict_proba(macro_t)[0]
                    classes = self.log_regs[i].classes_
                    for idx, s in enumerate(classes):
                        A_t[i, s] = probs[idx]
                else:
                    A_t[i, :] = 1.0 / self.n_states
            for s in range(self.n_states):
                alpha[t, s] = norm.pdf(returns[t], self.emission_mu[s], self.emission_sigma[s]) * np.sum(alpha[t-1] * A_t[:, s])
            alpha[t] /= np.sum(alpha[t])
        # Identify state with highest mean
        best_state = np.argmax(self.emission_mu)
        return alpha[-1, best_state]
