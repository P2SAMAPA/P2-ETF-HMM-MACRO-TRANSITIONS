import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class HMMMacro:
    def __init__(self, n_states=3, max_iter=50):
        self.n_states = n_states
        self.max_iter = max_iter

    def fit(self, returns, macro_df):
        """
        Fit a standard HMM (constant transitions) and then train a logistic regression
        to predict the posterior state probabilities from macro variables.
        """
        T = len(returns)
        if T < 10:
            return self
        # Standard HMM with Baum-Welch
        # Initial parameters
        mu = np.linspace(-0.01, 0.01, self.n_states)
        sigma = np.ones(self.n_states) * 0.02
        A = np.ones((self.n_states, self.n_states)) / self.n_states
        pi = np.ones(self.n_states) / self.n_states

        def forward(obs):
            T = len(obs)
            alpha = np.zeros((T, self.n_states))
            for s in range(self.n_states):
                alpha[0, s] = pi[s] * norm.pdf(obs[0], mu[s], sigma[s])
            alpha[0] /= np.sum(alpha[0])
            for t in range(1, T):
                for s in range(self.n_states):
                    alpha[t, s] = norm.pdf(obs[t], mu[s], sigma[s]) * np.sum(alpha[t-1] * A[:, s])
                alpha[t] /= np.sum(alpha[t])
            return alpha

        def backward(obs):
            T = len(obs)
            beta = np.zeros((T, self.n_states))
            beta[-1] = 1.0
            for t in range(T-2, -1, -1):
                for s in range(self.n_states):
                    beta[t, s] = np.sum(A[s, :] * norm.pdf(obs[t+1], mu, sigma) * beta[t+1])
                beta[t] /= np.sum(beta[t])
            return beta

        for _ in range(self.max_iter):
            alpha = forward(returns)
            beta = backward(returns)
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            # Update mu, sigma
            for s in range(self.n_states):
                total = np.sum(gamma[:, s])
                if total > 0:
                    mu[s] = np.sum(gamma[:, s] * returns) / total
                    sigma[s] = np.sqrt(np.sum(gamma[:, s] * (returns - mu[s])**2) / total)
            # Update A
            xi = np.zeros((self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i, j] += gamma[t, i] * A[i, j] * norm.pdf(returns[t+1], mu[j], sigma[j]) * beta[t+1, j]
            A = xi / np.sum(xi, axis=1, keepdims=True)
        # Store emission parameters
        self.mu = mu
        self.sigma = sigma
        # Train logistic regression to predict state probabilities from macro at last step
        # Use the smoothed state probabilities gamma as soft targets
        # Use macro variables at each time step to predict the state distribution at that time
        # Then we can predict the state distribution for the current macro.
        macro_scaled = StandardScaler().fit_transform(macro_df)
        self.macro_scaler = StandardScaler()
        macro_scaled = self.macro_scaler.fit_transform(macro_df)
        # Targets: state probabilities at each time (gamma)
        # We'll use gamma as multi-label targets for multinomial regression
        # For each time step, we have a vector of probabilities
        # We'll fit a logistic regression with soft targets using iterative scaling.
        # For simplicity, we'll fit a multinomial logistic regression on the most likely state.
        y = np.argmax(gamma, axis=1)
        self.log_reg = LogisticRegression(multi_class='multinomial', max_iter=100)
        self.log_reg.fit(macro_scaled, y)
        return self

    def score(self, returns, macro_df):
        """
        Return the probability of being in the best (highest mean) state at the last time step,
        as predicted by macro variables.
        """
        if not hasattr(self, 'log_reg'):
            return 0.0
        # Align lengths
        min_len = min(len(returns), len(macro_df))
        returns = returns[:min_len]
        macro_df = macro_df.iloc[:min_len]
        # Use macro at the last time step to predict state probabilities
        macro_last = macro_df.iloc[-1].values.reshape(1, -1)
        macro_last_scaled = self.macro_scaler.transform(macro_last)
        state_probs = self.log_reg.predict_proba(macro_last_scaled)[0]
        # Find best state (highest mean)
        best_state = np.argmax(self.mu)
        # Ensure best_state is within the number of classes
        if best_state >= len(state_probs):
            best_state = 0
        prob = state_probs[best_state]
        return float(prob)
