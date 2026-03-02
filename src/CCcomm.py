#CCcomm.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

class NetworkLinearRegression(nn.Module):
    def __init__(self, n_features, alpha=0, lambda_=0.01, Omega=None, device=None):
        """
        Network Linear Regression: Linear Regression with L1 + Laplacian Regularization.
        """
        super(NetworkLinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # 线性层: X @ W + b
        self.alpha = alpha  # Balance coefficient between L1 and Laplacian regularization
        self.lambda_ = lambda_  # Regularization strength
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device)
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))
        I = sparse.eye(Omega.shape[0])
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt
        return L_sym

    def forward(self, X):
        return self.linear(X)  

    def loss(self, predictions, y):
        """ Compute mean squared error loss and add L1 and Laplacian regularization """
        mse_loss = nn.MSELoss()(predictions, y)

        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return mse_loss + l1_penalty + laplacian_penalty


def CCcommInfer_linear(X, y, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.001, n_epochs=500, verbose=False):#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]

    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X, feature_means = center_features(X)
    X, y = X.to(device), y.to(device)

    model = NetworkLinearRegression(n_features, alpha, lambda_, Omega, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions = model(X)
        loss = model.loss(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def CCcommInfer_multiview_linear(X1, X2, y, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.001, n_epochs=500, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert X1.shape[0] == X2.shape[0], "The sample sizes of X1 and X2 must be the same."
    assert X1.shape[1] == X2.shape[1], "In order to share the weights, the number of features of X1 and X2 must be the same."
    n_features = X1.shape[1]

    if not isinstance(X1, torch.Tensor):
        X1 = torch.from_numpy(X1).float()
    if not isinstance(X2, torch.Tensor):
        X2 = torch.from_numpy(X2).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X1, _ = center_features(X1)
    X2, _ = center_features(X2)
    X1, X2, y = X1.to(device), X2.to(device), y.to(device)  

    model = NetworkLinearRegression(n_features, alpha, lambda_, Omega, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions1 = model(X1)
        predictions2 = model(X2)
        loss1 = model.loss(predictions1, y)
        loss2 = model.loss(predictions2, y)
        loss_total = theta * loss1 + (1-theta) * loss2
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Total: {loss_total.item():.4f}")

    return model

class NetworkCoxRegression(nn.Module):
    def __init__(self, n_features, alpha=0, lambda_=0.01, Omega=None, device=None):
        """
        Network Cox Regression: Cox Regression with L1 + Laplacian Regularization.

        Parameters:
        - n_features: number of input features.
        - alpha: weight for L1 vs Laplacian regularization (0 <= alpha <= 1).
        - lambda_: regularization strength.
        - Omega: similarity matrix (scipy.sparse.csr_matrix) representing the network structure.
        - device: device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(NetworkCoxRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)  # Linear layer: X @ W (no bias for Cox regression)
        self.alpha = alpha                     # Balance between L1 and Laplacian regularization
        self.lambda_ = lambda_                 # Regularization strength
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)  # Compute Laplacian
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device)  # Convert to PyTorch sparse tensor and move to device
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        """
        Compute the symmetric normalized Laplacian matrix from the similarity matrix Omega.
        """
        # Compute degree matrix D
        D = sparse.diags(Omega.sum(axis=1).A1)  # D is a diagonal matrix with row sums of Omega

        # Compute D^{-1/2}
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))

        # Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} Omega D^{-1/2}
        I = sparse.eye(Omega.shape[0])  # Identity matrix
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt

        return L_sym

    def forward(self, X):
        """
        Forward pass: compute risk scores (log hazard ratio).
        """
        return self.linear(X)  # Output risk scores (no activation function)

    def loss(self, risk_scores, time, event):
        """
        Compute the Cox partial likelihood loss with L1 and Laplacian regularization.

        Parameters:
        - risk_scores: Model output (risk scores).
        - time: Survival times.
        - event: Event indicators (1 if event occurred, 0 if censored).
        """
        # Sort by time (ascending order)
        sorted_time, indices = torch.sort(time, descending=False)
        sorted_risk_scores = risk_scores[indices]
        sorted_event = event[indices]

        # Compute partial likelihood loss
        loss = 0.0
        for i in range(len(sorted_time)):
            if sorted_event[i] == 1:  # Only consider events (not censored)
                # Risk set: individuals still at risk at time sorted_time[i]
                risk_set = (sorted_time >= sorted_time[i]).nonzero(as_tuple=True)[0]
                # Log-sum-exp of risk scores in the risk set
                log_sum_exp = torch.logsumexp(sorted_risk_scores[risk_set], dim=0)
                # Partial likelihood contribution
                loss += sorted_risk_scores[i] - log_sum_exp

        # Negative log partial likelihood
        loss = -loss / torch.sum(event)  # Normalize by number of events

        # L1 regularization
        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        # Laplacian regularization
        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()  # Get weight vector (n_features,)
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()  # weight^T @ L_sym @ weight
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return loss + l1_penalty + laplacian_penalty


def CCcommInfer_cox(X, time, event, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.001, n_epochs=500, verbose=False):
    """
    Train the Network Cox Regression model.

    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - time: Survival times (torch.Tensor or numpy.ndarray).
    - event: Event indicators (torch.Tensor or numpy.ndarray).
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.

    Returns:
    - model: Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(time, torch.Tensor):
        time = torch.from_numpy(time).float()
    if not isinstance(event, torch.Tensor):
        event = torch.from_numpy(event).float()

    X, time, event = X.to(device), time.to(device), event.to(device)  # Move data to device
    model = NetworkCoxRegression(n_features, alpha, lambda_, Omega, device=device).to(device)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # Model prediction (risk scores)
        risk_scores = model(X)
        loss = model.loss(risk_scores, time, event)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def CCcommInfer_multiview_cox(
    X1, X2, time, event,
    alpha=0, lambda_=0.01, Omega=None, theta=0.5,
    learning_rate=0.001, n_epochs=500, verbose=False
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert X1.shape[0] == X2.shape[0], "The sample sizes of X1 and X2 must be the same."

    assert X1.shape[1] == X2.shape[1], "In order to share the weights, the number of features of X1 and X2 must be the same."

    n_features = X1.shape[1]

    def to_tensor(x):
        return torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()
    X1, X2 = to_tensor(X1).to(device), to_tensor(X2).to(device)
    time, event = to_tensor(time).to(device), to_tensor(event).to(device)

    model = NetworkCoxRegression(n_features, alpha, lambda_, Omega, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        risk_scores1 = model(X1)
        risk_scores2 = model(X2)

        loss1 = model.loss(risk_scores1, time, event)
        loss2 = model.loss(risk_scores2, time, event)

        loss_total = theta * loss1 + (1-theta) * loss2

        loss_total.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Total: {loss_total.item():.4f}")

    return model

class NetworkOrdinalLogit(nn.Module):
    def __init__(self, n_features, num_classes, alpha=0, lambda_=0.1, Omega=None, device=None):
        """
        Network Ordinal Logit Model: Ordinal Logistic Regression with L1 + Laplacian Regularization.
        
        Parameters:
        - n_features: number of input features.
        - num_classes: number of ordered classes.
        - alpha: weight for L1 vs Laplacian regularization (0 <= alpha <= 1).
        - lambda_: regularization strength.
        - Omega: similarity matrix (scipy.sparse.csr_matrix) representing the network structure.
        - device: device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(NetworkOrdinalLogit, self).__init__()
        self.linear = nn.Linear(n_features, 1) 
        self.num_classes = num_classes
        self.alpha = alpha                     
        self.lambda_ = lambda_                 
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        self.cutpoints = nn.Parameter(torch.arange(num_classes - 1).float() - num_classes / 2).to(self.device)

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)  # Compute Laplacian
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device) 
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))
        I = sparse.eye(Omega.shape[0])  
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt
        
        return L_sym

    def forward(self, X):
        """
        Forward pass: compute cumulative probabilities for ordinal classes.
        """
        logits = self.linear(X)  
        sigmoids = torch.sigmoid(self.cutpoints - logits) 
        return sigmoids
    

    def loss(self, predictions, y):
        """
        Compute the negative log-likelihood loss for ordinal logit model with L1 and Laplacian regularization.
        """
        sigmoids = predictions
        sigmoids = torch.cat([torch.zeros_like(sigmoids[:, [0]]), sigmoids, torch.ones_like(sigmoids[:, [0]])], dim=1)
        
        class_probs = sigmoids[:, 1:] - sigmoids[:, :-1]

        y_true = y.long().squeeze()
        likelihoods = torch.gather(class_probs, 1, y_true.unsqueeze(1))
        nll_loss = -torch.log(likelihoods + 1e-15).mean()

        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()  
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()  # weight^T @ L_sym @ weight
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return nll_loss + l1_penalty + laplacian_penalty


def CCcommInfer_ordinal_logit(X, y, alpha=0, lambda_=0.1, Omega=None, learning_rate=0.0001, n_epochs=500, tail = 'right', verbose=False):
    """
    Train the Network Ordinal Logit model.
    
    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - y: Training labels (torch.Tensor or numpy.ndarray).
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
    - model: Trained model.
    - feature_means: Means of each feature (used for centering).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    if tail == 'right':
        y = y.max() - y# Make sure y starts at 0
    elif tail == 'left':
        y = y - y.min()
    num_classes = len(np.unique(y))

    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X, feature_means = center_features(X)
    X, y = X.to(device), y.to(device)  

    model = NetworkOrdinalLogit(n_features, num_classes, alpha, lambda_, Omega, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions = model(X)
        loss = model.loss(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    return model

def CCcommInfer_multiview_ordinal_logit(X1, X2, y, alpha=0, lambda_=0.1, Omega=None, theta=0.5, learning_rate=0.0001, n_epochs=500, verbose=False):
    """
    Train the Network Ordinal Logit model.
    
    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - y: Training labels (torch.Tensor or numpy.ndarray).
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
    - model: Trained model.
    - feature_means: Means of each feature (used for centering).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert X1.shape[0] == X2.shape[0], "The sample sizes of X1 and X2 must be the same."

    assert X1.shape[1] == X2.shape[1], "In order to share the weights, the number of features of X1 and X2 must be the same."

    n_features = X1.shape[1]
    y = y.max() - y# Make sure y starts at 0
    num_classes = len(np.unique(y))

    if not isinstance(X1, torch.Tensor):
        X1 = torch.from_numpy(X1).float()
    if not isinstance(X2, torch.Tensor):
        X2 = torch.from_numpy(X2).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X1, _ = center_features(X1)
    X2, _ = center_features(X2)
    X1, X2, y = X1.to(device), X2.to(device), y.to(device)  

    model = NetworkOrdinalLogit(n_features, num_classes, alpha, lambda_, Omega, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions1 = model(X1)
        predictions2 = model(X2)
        loss1 = model.loss(predictions1, y)
        loss2 = model.loss(predictions2, y)
        loss_total = theta * loss1 + (1-theta) * loss2
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Total: {loss_total.item():.4f}")
    
    return model

class NetworkLogisticRegression(nn.Module):
    def __init__(self, n_features, alpha=0, lambda_=0.01, Omega=None, device=None):
        """
        Network Logistic Regression: Logistic Regression with L1 + Laplacian Regularization.

        Parameters:
        - n_features: number of input features.
        - alpha: weight for L1 vs Laplacian regularization (0 <= alpha <= 1).
        - lambda_: regularization strength.
        - Omega: similarity matrix (scipy.sparse.csr_matrix) representing the network structure.
        - device: device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(NetworkLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # Linear layer: X @ W + b
        self.sigmoid = nn.Sigmoid()            # Activation function: Sigmoid
        self.alpha = alpha                     # Balance between L1 and Laplacian regularization
        self.lambda_ = lambda_                 # Regularization strength
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)  # Compute Laplacian
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device)  
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))
        # Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} Omega D^{-1/2}
        I = sparse.eye(Omega.shape[0])  
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt

        return L_sym

    def forward(self, X):
        """
        Forward pass: compute logits and apply sigmoid.
        """
        logits = self.linear(X) 
        return self.sigmoid(logits)  

    def loss(self, predictions, y):
        """
        Compute the logistic loss with L1 and Laplacian regularization.
        """
        # Binary cross-entropy loss
        bce_loss = nn.BCELoss()(predictions, y)

        # L1 regularization
        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()  
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()# weight^T @ L_sym @ weight
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return bce_loss + l1_penalty + laplacian_penalty


def CCcommInfer_logit(X, y, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.0001, n_epochs=500, verbose=False):
    """
    Train the Network Logistic Regression model. 

    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - y: Training labels (torch.Tensor or numpy.ndarray).
    - n_features: Number of input features.
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.

    Returns:
    - model: Trained model.
    - feature_means: Means of each feature (used for centering).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X, feature_means = center_features(X)
    X, y = X.to(device), y.to(device)  

    model = NetworkLogisticRegression(n_features, alpha, lambda_, Omega, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions = model(X)
        loss = model.loss(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def CCcommInfer_multiview_logit(X1, X2, y, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.0001, n_epochs=500, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert X1.shape[0] == X2.shape[0], "The sample sizes of X1 and X2 must be the same."
    assert X1.shape[1] == X2.shape[1], "In order to share the weights, the number of features of X1 and X2 must be the same."
    n_features = X1.shape[1]

    if not isinstance(X1, torch.Tensor):
        X1 = torch.from_numpy(X1).float()
    if not isinstance(X2, torch.Tensor):
        X2 = torch.from_numpy(X2).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X1, _ = center_features(X1)
    X2, _ = center_features(X2)
    X1, X2, y = X1.to(device), X2.to(device), y.to(device)  

    model = NetworkLogisticRegression(n_features, alpha, lambda_, Omega, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions1 = model(X1)
        predictions2 = model(X2)
        loss1 = model.loss(predictions1, y)
        loss2 = model.loss(predictions2, y)
        loss_total = theta * loss1 + (1-theta) * loss2
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Total: {loss_total.item():.4f}")

    return model

def center_features(X):
    feature_means = X.mean(dim=0, keepdim=True)
    X_centered = X - feature_means
    return X_centered, feature_means

"""
class CCcommInfer:
    def __init__(self, method="linear", **kwargs):
        self.method = method.lower()
        self.model = self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        if self.method == "linear":
            return CCcommInfer_linear(**kwargs)
        elif self.method == "cox":
            return CCcommInfer_cox(**kwargs)
        elif self.method == "ordinal":
            return CCcommInfer_ordinal_logit(**kwargs)
        elif self.method == "binary":
            return CCcommInfer_logit(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
"""

class CCcommInfer_LRTF:
    def __init__(self, method="linear", **kwargs):
        self.method = method.lower()
        self.model = self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        if self.method == "linear":
            return CCcommInfer_multiview_linear(**kwargs)
        elif self.method == "cox":
            return CCcommInfer_multiview_cox(**kwargs)
        elif self.method == "ordinal":
            return CCcommInfer_multiview_ordinal_logit(**kwargs)
        elif self.method == "binary":
            return CCcommInfer_multiview_logit(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CCcommInfer:
    def __init__(
        self,
        method="linear",
        use_stability=False,
        n_repeat=1, 
        base_seed=0,
        **kwargs
    ):
        self.method = method.lower()
        self.use_stability = use_stability
        self.n_repeat = max(1, int(n_repeat))
        self.base_seed = base_seed
        self.kwargs = kwargs

        self.model = None           # A representative model (default is the 1st run)
        self.beta = None            # The recommended value is β (default is beta_mean)
        self.beta_first = None      # The first fitted value of β
        self.beta_mean = None 
        self.beta_std = None
        self.betas_all = None
        self.stability_pos = None
        self.stability_neg = None

        self.betas_bootstrap = None          # (B, p)， bootstrap  β
        self.beta_bootstrap_mean = None      # bootstrap β mean
        self.beta_bootstrap_std = None       # bootstrap β std
        self.bootstrap_support_pos = None    # P̂(β > 0)
        self.bootstrap_support_neg = None    # P̂(β < 0)

        self.model = self._initialize_model()

    def _fit_once(self, seed_offset: int):

        _set_global_seed(self.base_seed + seed_offset)

        m = self.method
        X = self.kwargs["X"]

        if m in ["linear", "binary", "ordinal"]:
            y = self.kwargs["y"]
        elif m == "cox":
            time = self.kwargs["time"]
            event = self.kwargs["event"]
        else:
            raise ValueError(f"Unknown method: {m}")

        alpha = self.kwargs.get("alpha", 0)
        lambda_ = self.kwargs.get("lambda_", 0.1)
        Omega = self.kwargs.get("Omega", None)
        learning_rate = self.kwargs.get("learning_rate", None)
        n_epochs = self.kwargs.get("n_epochs", 500)
        verbose = self.kwargs.get("verbose", False)

        extra_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k not in [
                "X", "y", "time", "event",
                "alpha", "lambda_", "Omega",
                "learning_rate", "n_epochs",
                "verbose", "device"
            ]
        }

        if m == "binary":
            model = CCcommInfer_logit(
                X=X,
                y=y,
                alpha=alpha,
                lambda_=lambda_,
                Omega=Omega,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=verbose
            )

        elif m == "linear":
            model = CCcommInfer_linear(
                X=X,
                y=y,
                alpha=alpha,
                lambda_=lambda_,
                Omega=Omega,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=verbose
            )

        elif m == "ordinal":
            model = CCcommInfer_ordinal_logit(
                X=X,
                y=y,
                alpha=alpha,
                lambda_=lambda_,
                Omega=Omega,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=verbose,
                **extra_kwargs
            )

        elif m == "cox":
            model = CCcommInfer_cox(
                X=X,
                time=time,
                event=event,
                alpha=alpha,
                lambda_=lambda_,
                Omega=Omega,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=verbose
            )

        else:
            raise ValueError(f"Unknown method: {m}")

        beta = model.linear.weight.detach().cpu().squeeze().numpy()
        return model, beta

    def sample_bootstrap(
        self,
        n_bootstrap=100,
        bootstrap_seed=0,
        verbose=True
    ):
       
        m = self.method

        X = self.kwargs["X"]

        if hasattr(X, "detach"):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        n_samples = X_np.shape[0]

        if m in ["linear", "binary", "ordinal"]:
            if "y" not in self.kwargs:
                raise ValueError(f"method='{m}' requests y to bootstrap")
            y = self.kwargs["y"]
            if hasattr(y, "detach"):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = np.asarray(y)
        elif m == "cox":
            if ("time" not in self.kwargs) or ("event" not in self.kwargs):
                raise ValueError("method='cox' requests time and event to bootstrap")
            time = np.asarray(self.kwargs["time"])
            event = np.asarray(self.kwargs["event"])
        else:
            raise ValueError(f"Unknown method: {m}")

        alpha = self.kwargs.get("alpha", 0)
        lambda_ = self.kwargs.get("lambda_", 0.1)
        Omega = self.kwargs.get("Omega", None)
        learning_rate = self.kwargs.get("learning_rate", 0.0001)
        n_epochs = self.kwargs.get("n_epochs", 500)
        verbose_flag = self.kwargs.get("verbose", False)

        extra_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k not in [
                "X", "y", "time", "event",
                "alpha", "lambda_", "Omega",
                "learning_rate", "n_epochs",
                "verbose", "device"
            ]
        }

        if self.beta is not None:
            p = np.asarray(self.beta).shape[0]
        elif self.beta_first is not None:
            p = np.asarray(self.beta_first).shape[0]
        else:
            raise ValueError("First, create the CCcommInfer object (which has undergone at least one fitting process), and then call the sample_bootstrap function.")

        B = int(n_bootstrap)
        betas_boot = np.zeros((B, p), dtype=float)

        rng = np.random.default_rng(bootstrap_seed)

        for b in tqdm(range(B)):
            if verbose or verbose_flag:
                if (b + 1) % max(1, B // 10) == 0:
                    print(f"[{m}] sample bootstrap {b+1}/{B} ...")

            _set_global_seed(bootstrap_seed + b)

            idx = rng.integers(0, n_samples, size=n_samples)

            X_b = X_np[idx]

            if m in ["linear", "binary", "ordinal"]:
                y_b = y_np[idx]
            else:
                time_b = time[idx]
                event_b = event[idx]
            if m == "binary":
                model_b = CCcommInfer_logit(
                    X=X_b,
                    y=y_b,
                    alpha=alpha,
                    lambda_=lambda_,
                    Omega=Omega,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    verbose=False
                )

            elif m == "linear":
                model_b = CCcommInfer_linear(
                    X=X_b,
                    y=y_b,
                    alpha=alpha,
                    lambda_=lambda_,
                    Omega=Omega,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    verbose=False
                )

            elif m == "ordinal":
                model_b = CCcommInfer_ordinal_logit(
                    X=X_b,
                    y=y_b,
                    alpha=alpha,
                    lambda_=lambda_,
                    Omega=Omega,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    verbose=False,
                    **extra_kwargs
                )

            elif m == "cox":
                model_b = CCcommInfer_cox(
                    X=X_b,
                    time=time_b,
                    event=event_b,
                    alpha=alpha,
                    lambda_=lambda_,
                    Omega=Omega,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    verbose=False
                )

            beta_b = model_b.linear.weight.detach().cpu().squeeze().numpy()
            betas_boot[b, :] = beta_b

        beta_boot_mean = betas_boot.mean(axis=0)
        beta_boot_std = betas_boot.std(axis=0, ddof=1)

        support_pos = np.mean(betas_boot > 0, axis=0)  
        support_neg = np.mean(betas_boot < 0, axis=0)  

        self.betas_bootstrap = betas_boot
        self.beta_bootstrap_mean = beta_boot_mean
        self.beta_bootstrap_std = beta_boot_std
        self.bootstrap_support_pos = support_pos
        self.bootstrap_support_neg = support_neg

        if verbose or verbose_flag:
            print(f"[{m}] sample bootstrap finished: B={B}")


    def _initialize_model(self):
        if (not self.use_stability) or (self.n_repeat <= 1):
            model, beta = self._fit_once(seed_offset=0)
            self.beta_first = beta
            self.beta_mean = beta
            self.beta_std = np.zeros_like(beta)
            self.beta = beta              
            self.betas_all = beta[None, :]
            self.stability_pos = np.mean(self.betas_all > 0, axis=0)
            self.stability_neg = np.mean(self.betas_all < 0, axis=0)
            return model

        if self.kwargs.get("verbose", False):
            print(f"[{self.method}] Stability mode ON: repeat {self.n_repeat} runs...")

        betas_all = None
        model_first, beta_first = None, None
    
        for r in tqdm(range(self.n_repeat)):
            if self.kwargs.get("verbose", False):
                print(f"[{self.method}] Repeat run {r+1}/{self.n_repeat} ...")
    
            model_r, beta_r = self._fit_once(seed_offset=r)

            if r == 0:
                model_first, beta_first = model_r, beta_r
                p = beta_r.shape[0]
                betas_all = np.zeros((self.n_repeat, p), dtype=float)
    
            betas_all[r, :] = beta_r
        
        beta_mean = betas_all.mean(axis=0)
        beta_std = betas_all.std(axis=0, ddof=1)

        stability_pos = np.mean(betas_all > 0, axis=0)
        stability_neg = np.mean(betas_all < 0, axis=0)

        self.beta_first = beta_first
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.beta = beta_first #beta_mean
        self.betas_all = betas_all
        self.stability_pos = stability_pos
        self.stability_neg = stability_neg

        return model_first




