#LRcomm.py

#Elastic net regularized H
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import NMF
from numpy.linalg import eig
from numpy.linalg import norm
from scipy.optimize import minimize_scalar

class UnsupervisedNMF(nn.Module):
    def __init__(self, X, n_components, beta=0.01, rho=0.5, device="cuda"):
        """
        Unsupervised NMF model with support for Elastic Net regularization of H.
        Args:
            X: Input matrix (samples × features)
            n_components: Dimension of the latent space
            beta: Elastic Net regularization coefficient
            rho: Balance coefficient between L1 and L2 regularization (0 ≤ rho ≤ 1)
            device: 'cuda' or 'cpu'
        """
        super(UnsupervisedNMF, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)

        m, n = X.shape
        self.n_components = n_components
        self.beta = beta
        self.rho = rho

        # Initialize W and H using sklearn's non-negative SVD
        #nmf_model = NMF(n_components=n_components, init='nndsvd', max_iter=100)
        nmf_model = NMF(n_components=n_components, init='random', max_iter=100)
        W_init = nmf_model.fit_transform(X)
        H_init = nmf_model.components_

        self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32, device=self.device))
        self.H = nn.Parameter(torch.tensor(H_init, dtype=torch.float32, device=self.device))

    def forward(self):
        """
        Reconstruct X
        """
        return torch.mm(self.W, self.H)

    def loss_function(self):
        """
        Loss function: includes reconstruction error and Elastic Net regularization term for H.
        """
        reconstruction_loss = torch.norm(self.X - self.forward(), p="fro") ** 2

        l1_regularization = torch.sum(torch.abs(self.H))  
        l2_regularization = torch.norm(self.H, p="fro") ** 2  
        elastic_net_regularization = self.beta * (self.rho * l1_regularization + (1 - self.rho) * l2_regularization)

        return reconstruction_loss + elastic_net_regularization


def LRcommDiscover(X, n_components, beta=0.01, rho=0.5, 
                                    maxiter=2000, lr=0.001, tol=1e-5, ica_tol=1e2,ica_maxiter=5e3, NNICA = False, device=None,non_negative = False):

    """
    Train the unsupervised NMF model.
    Args:
        X: Input matrix (samples × features)
        n_components: Dimension of the latent space
        beta: Elastic Net regularization coefficient
        rho: Balance coefficient between L1 and L2 regularization
        maxiter: Maximum number of iterations
        lr: Learning rate
        tol: Convergence threshold of NMF
        NNICA: Whether to use NNICA to optimize results
        ica_tol: The smaller the ica_tol parameter, the larger the non-negative constraint, the more difficult it is to converge
        ica_maxiter: Maximum number of iterations of NNICA
        device: 'cuda' or 'cpu'
        non_negative: Whether to force the result to be non-negative
    Returns:
        W_rotated: Left matrix (samples × latent dimension)
        H_ica: Right matrix (latent dimension × features)
        loss_history: Loss values at each iteration
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UnsupervisedNMF(X, n_components, beta, rho, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for iteration in range(maxiter):

        optimizer.zero_grad()
        loss = model.loss_function()
        loss.backward()
        optimizer.step()
        # Enforce non-negativity constraint
        model.W.data = torch.clamp(model.W.data, min=0)
        model.H.data = torch.clamp(model.H.data, min=0)
        loss_history.append(loss.item())

        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {iteration}")
            break

        if iteration % 50 == 0:
            print(f"Iteration {iteration}/{maxiter}, Loss: {loss.item():.4f}")

    W = model.W.detach().cpu().numpy()
    H = model.H.detach().cpu().numpy()

    if NNICA:
            H_ica, W_ica, Z, _ = run_nn_ica(H,t_tol=ica_tol,i_max=ica_maxiter)
            # Compute the inverse rotation of W
            W_rotated = np.dot(W, np.linalg.pinv(W_ica))
            if non_negative:
                W_rotated[W_rotated < 0] =0
                H_ica[H_ica < 0] =0
            return W_rotated, H_ica, loss_history
    else:
        return W, H, loss_history


def select_k_components(
    X, S, k_max=20, repeat_times=5,
    max_iter=2000, lr=1e-3,
    min_iters=200, rtol=1e-6, atol=1e-7, patience=20,
    eii_tol=0.05,
    lambda_pheno=0.10, lambda_en=0.05,
    seed=0, verbose=False, device=None
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Ks = list(range(2, k_max + 1))
    dist_K = []  
    best_K = Ks[0]

    for idx, K in enumerate(Ks):
        errs = []

        for rep in range(repeat_times):
            model = PhenotypeRegularizedNMF(X, S, K, device=device).to(device)

            with torch.no_grad():
                R = model.X - model.forward()
                R_tilde = torch.sum(R * R) / model._mn

                SW = torch.mm(model.S, model.W)
                P_tilde = torch.sum(SW * SW) / model._mk

                l1 = torch.sum(torch.abs(model.H)) / model._kn
                l2 = torch.sum(model.H * model.H) / model._kn
                E_tilde = model.rho * l1 + (1 - model.rho) * l2

                eps = 1e-12
                model.alpha = float(lambda_pheno) * float(R_tilde / (P_tilde + eps))
                model.beta  = float(lambda_en   ) * float(R_tilde / (E_tilde + eps))

            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_history = []
            no_improve = 0

            for it in range(max_iter):
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss_function()  
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.W.clamp_(min=0)
                    model.H.clamp_(min=0)

                cur = float(loss.item())
                loss_history.append(cur)

                if it >= max(min_iters, 1):
                    prev = loss_history[-2]
                    rel_impr = abs(prev - cur) / max(1.0, abs(prev))
                    abs_impr = abs(prev - cur)
                    if (rel_impr < rtol) or (abs_impr < atol):
                        no_improve += 1
                    else:
                        no_improve = 0
                    if no_improve >= patience:
                        break

            with torch.no_grad():
                R = model.X - model.forward()
                recon_err = float(torch.sum(R * R) / model._mn)
            errs.append(recon_err)

        mean_err = float(np.mean(errs))
        dist_K.append(mean_err)

        if verbose:
            print(f"[K={K}] recon_err={mean_err:.6f}")

        # eii = (E(K-1) - E(K)) / (E(2) - E(K))
        if idx >= 2:
            E2 = dist_K[0]      
            EKm1 = dist_K[-2]      
            EK   = dist_K[-1]      
            denom = max(1e-12, (E2 - EK))
            eii = (EKm1 - EK) / denom

            if verbose:
                print(f"   eii={eii:.4f}")

            if (EKm1 - EK <= 0) or (eii < eii_tol):
                best_K = Ks[idx - 1]
                if verbose:
                    print(f"[Elbow] Stop at K={K}, select K={best_K}")
                break
            else:
                best_K = K  

    return best_K


#Elastic net regularized H
class PhenotypeRegularizedNMF(nn.Module):
    def __init__(self, X, S, n_components, alpha=0.1, beta=0.1, rho=0.75, device="cuda"):
        """
        Phenotype-regularized NMF model with support for Elastic Net regularization of H.
        Args:
            X: Input matrix (samples × features)
            S: Phenotype information matrix (samples × samples)
            n_components: Dimension of the latent space
            alpha: Phenotype regularization coefficient
            beta: Elastic Net regularization coefficient
            rho: Balance coefficient between L1 and L2 regularization (0 ≤ rho ≤ 1)
            device: 'cuda' or 'cpu'
        """
        super(PhenotypeRegularizedNMF, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.S = torch.tensor(S, dtype=torch.float32, device=self.device)

        m, n = X.shape
        self.m, self.n = int(m), int(n)
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        nmf_model = NMF(n_components=n_components, init='nndsvd', max_iter=100, random_state=0)
        W_init = nmf_model.fit_transform(X)
        H_init = nmf_model.components_

        self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32, device=self.device))
        self.H = nn.Parameter(torch.tensor(H_init, dtype=torch.float32, device=self.device))

        self._mn = float(self.m * self.n)
        self._mk = float(self.m * self.n_components)
        self._kn = float(self.n_components * self.n)

    def forward(self):
        return torch.mm(self.W, self.H)

    def loss_function(self):
        R = self.X - self.forward()
        reconstruction_loss = torch.sum(R * R) / self._mn

        SW = torch.mm(self.S, self.W)
        phenotype_loss = self.alpha * (torch.sum(SW * SW) / self._mk)

        l1_regularization = torch.sum(torch.abs(self.H)) / self._kn
        l2_regularization = torch.sum(self.H * self.H) / self._kn
        elastic_net_regularization = self.beta * (self.rho * l1_regularization + (1 - self.rho) * l2_regularization)

        return reconstruction_loss + phenotype_loss + elastic_net_regularization


def LRcommMining(X, S, n_components, alpha=0.1, beta=0.1, rho=0.75,
                 maxiter=4000, lr=0.001, atol=1e-7, device=None,
                 min_iters=200, rtol=1e-6, patience=20, verbose=True,
                 lambda_pheno=None, lambda_en=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PhenotypeRegularizedNMF(X, S, n_components, alpha, beta, rho, device).to(device)

    if (lambda_pheno is not None) or (lambda_en is not None):
        with torch.no_grad():
            R = model.X - model.forward()
            R_tilde = torch.sum(R * R) / model._mn

            SW = torch.mm(model.S, model.W)
            P_tilde = torch.sum(SW * SW) / model._mk

            l1 = torch.sum(torch.abs(model.H)) / model._kn
            l2 = torch.sum(model.H * model.H) / model._kn
            E_tilde = model.rho * l1 + (1 - model.rho) * l2

            eps = 1e-12
            if lambda_pheno is not None:
                model.alpha = float(lambda_pheno) * float(R_tilde / (P_tilde + eps))
            if lambda_en is not None:
                model.beta  = float(lambda_en   ) * float(R_tilde / (E_tilde + eps))

        if verbose:
            print(f"[Balance] alpha={model.alpha:.6g}, beta={model.beta:.6g} "
                  f"(targets: pheno={lambda_pheno}, en={lambda_en})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    no_improve = 0

    for iteration in range(maxiter):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss_function()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.W.clamp_(min=0)
            model.H.clamp_(min=0)

        cur = float(loss.item())
        loss_history.append(cur)

        if iteration >= max(min_iters, 1):
            prev = loss_history[-2]
            rel_impr = abs(prev - cur) / max(1.0, abs(prev))
            abs_impr = abs(prev - cur)
            if (rel_impr < rtol) or (abs_impr < atol):
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopped at iter {iteration} (patience={patience})")
                break

        if verbose and iteration % 200 == 0:
            print(f"Iteration {iteration}/{maxiter}, Loss: {cur:.6f}")

    W = model.W.detach().cpu().numpy()
    H = model.H.detach().cpu().numpy()
    return W, H, loss_history

def select_k_nmf(X, S=None, k_max=20, repeat_times=10, max_iter=2000, lr=0.001, tol=1e1, eii_tol=0.05,
                 use_semi_supervised=False, semi_supervised_params=None, unsupervised_params=None,
                 seed=0, verbose=False):
    """
    Select the optimal number of components (K) for NMF using normalized error improvement rate (eii).
    This function supports both unsupervised and semi-supervised NMF.

    Args:
        X: Input matrix (samples × features)
        S: Phenotype information matrix (samples × samples) (required for semi-supervised NMF)
        k_max: Maximum value of K to test
        repeat_times: Number of repetitions for each K to compute average error
        max_iter: Maximum number of training iterations
        lr: Learning rate for optimization
        tol: Convergence threshold (change in loss)
        eii_tol: Convergence threshold of the rate of change of reconstruction error
        use_semi_supervised: Whether to use semi-supervised NMF
        semi_supervised_params: Additional parameters for semi-supervised NMF (e.g., alpha, beta, rho)
        unsupervised_params: Additional parameters for unsupervised NMF (e.g., beta, rho)
        seed: Random seed for reproducibility
        verbose: Whether to print detailed information

    Returns:
        optimal_K: The selected optimal number of components (K)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    semi_supervised_params = semi_supervised_params or {}
    unsupervised_params = unsupervised_params or {}

    K_all = range(2, k_max + 1)
    dist_K = []

    for Ki in K_all:
        errors = []
        for _ in range(repeat_times):
            if use_semi_supervised:
                model = PhenotypeRegularizedNMF(X, S, Ki, **semi_supervised_params, device=device).to(device)
            else:
                model = UnsupervisedNMF(X, Ki, **unsupervised_params, device=device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            prev_loss = float('inf')

            for iteration in range(max_iter):
                optimizer.zero_grad()
                loss = model.loss_function()
                loss.backward()
                optimizer.step()

                model.W.data = torch.clamp(model.W.data, min=0)
                model.H.data = torch.clamp(model.H.data, min=0)

                if abs(loss.item() - prev_loss) < tol:
                    break
                prev_loss = loss.item()

            errors.append(loss.item())

        mean_error = np.mean(errors)
        dist_K.append(mean_error)

        if verbose:
            print(f"K={Ki}, Reconstruction Error={mean_error:.4f}")

        if Ki > 3:  
            eii = (dist_K[Ki-3] - dist_K[Ki-2]) / (dist_K[0] - dist_K[Ki-2])

            if verbose:
                print(f"K={Ki}, eii={eii:.4f}")

            if (dist_K[Ki-3] - dist_K[Ki-2] <= 0) or (eii < eii_tol):
                break  

    return Ki - 1  


#NNICA
def whiten(X, use_np=True):
    """Utility function to whiten data without zero-mean centering. Whitening means removing correlation between
    features and making the individual features have unit variance.

    :param
    X : np.array
        Data matrix. This is assumed to be in the (uncommon) format n_features x n_samples
    use_np : bool, optional (default: `True`)
        Whether to use numpy to compute the covariance matrix

    :returns
    Z : np.array
        Data matrix, whitened to remove covariance between the features
    """

    if use_np:
        C_X = np.cov(X, rowvar=True)
    else:
        C_X = (X - X.mean(1)) @ (X - X.mean(1)).T
    D, E = eig(C_X)
    V = E @ np.diag(1 / np.sqrt(D)) @ E.T
    Z = V @ X

    return Z


def rotation(phi):
    """Create 2D rotation matrix

    :param
    phi : float
        The angle by which we want to rotate.

    :returns
    A : np.array
        A 2D rotation matrix
    """

    return np.array([[np.cos(phi), np.sin(phi)],
                     [-np.sin(phi), np.cos(phi)]])


def loss(Y):
    """Compute the loss for a given reconstruction Y.
    This will simply be the sum of squared elements in the restriction
    of Y to it's negative elements

    :param
    Y : np.array
        Data matrix, reconstrcution of the sources

    :returns
    l : np.float
        The loss
    """
    # restrict Y to it's negative elements
    n_samples = Y.shape[1]
    Y_neg = np.where(Y < 0, Y, 0)

    return 1 / (2 * n_samples) * norm(Y_neg, ord='fro') ** 2


def obj_fun(phi, Z):
    """Objective to be used for finding the optimum rotation angle

    :param
    phi : float
        Rotation angle for which we wish to compute the los
    Z : np.matrix
        Whitened data matrix. Must have two rows, each corresponding to one feature.

    :returns
    l : float
        loss corresponding to a rotation of Z in 2D around phi
    """

    # check input
    if Z.shape[0] != 2:
        raise ValueError('Z has more than two features.')

    # rotate the data
    W = rotation(phi)
    Y = W @ Z

    return loss(Y)


def givens(n, i, j, phi):
    """Compute n-dimensional givens rotation

    :param
    n : int
        Dimension of the rotation matrix to be computed
    i, j : int
        Dimensions i and j define the surface we wish to rotate in
    phi : float
        Rotation angle

    :returns
    R : np.array
        Given's rotation

    """
    R = np.eye(n)
    R[i, i], R[j, j] = np.cos(phi), np.cos(phi)
    R[i, j], R[j, i] = np.sin(phi), -np.sin(phi)

    return R


def torque(Y):
    """Compute torque values of Y.

    These correspond to the gradient if different directions, where
    each direction is a possible rotation in a surface defines by two axis. The resulting matrix of
    torque values will have zeroes on the diagonal and will by symmetric.

    :param
    Y : np.matrix
        Reconstruction of the sourdes

    :returns
    t_max : float
        Maximum torque value found
    ixs : tuple
        i-j coordinates corresponding to the max. This defines a hyperplane in n-dimensional space.
    G : np.array
        Matrix of torque values. Symmetric and zero on the diagonal. Only the upper half is computed.
    """

    # compute the rectified parts of Y
    Y_pos = np.where(Y > 0, Y, 0)
    Y_neg = np.where(Y < 0, Y, 0)

    # compute torque values
    n = Y.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            G[i, j] = np.dot(Y_pos[i, :], Y_neg[j, :]) - np.dot(Y_neg[i, :], Y_pos[j, :])

    # find max and corresponding indices
    t_max = np.amax(np.abs(G))
    result = np.where(np.abs(G) == t_max)
    ixs = [result[i][0] for i in range(len(result))]

    return t_max, ixs, G


def run_nn_ica(X, t_tol=1e-1, t_neg=None, verbose=1, i_max=1e3, print_all=100,
               whiten_mat=True, keep='last', return_all=False):
    """Algorithm to run non-negative indipendent component analysis.

    Given some data X, find matrices A and S such that
    X = A S,
    where S is non-negative and has indipendent rows. We pose no
    constraint on the matrix A.

    This algorithm is implemented as described in Plumbley, 2003, and
    relies upon whitening and rotating the data. This is guaranteed to
    converge only if the sources are 'well grounded', i.e. have probability
    down to zero. Note that this is the implementation for a square mixing
    matrix A.

    Parameters
    --------
    X : np.array,
        The data matrix of shape (n_features, n_samples)
    t_tol : float, optional (default: `1e-1`)
        Stopping tolerance. If the maximum torque falls below this
        value, stop.
    t_neg: float, optional (default: `None`)
        Stopping number of negative elements. If #negative elements crosses
        this threshold, stop.
    verbose : int, optional (default: `1`)
        How much output to give
    i_max : int, optional (default: `1e3`)
        Maximum number of iterations
    print_all : int, optional (default: `100`)
        Print every print_all iterations
    whiten : bool, optional (default: `True`)
        whether to whiten the input matrix
    keep: Str, optional (default: `'last'`)
        which reconstruction to keep, possible options are:
        `'last'`, `'best_neg'`, `'best_tol'`
        and correspond to last, smallest #negative elements and smallest tolerance
        respectively
    return_all: bool, optional (default: `False`)
        whether to return all of the progress of Y, W
        returns Y_best, W_best, Z, t_max_arr, ys, ws

    Returns
    --------
    Y : np.array
        The reconstructed sources, up to scaling and permutation
    W : np.array
        The final rotation matrix
    Z : np.array
        The whitened data
    t_max_arr : np.array
        The maximum torque values for each iteration
    """

    assert keep in ('last', 'best_neg', 'best_tol'), f'Unknown selection criterion `{keep}`.'


    def set_best(Y, W, tol):
        nonlocal Y_best, W_best, tol_best

        if return_all:
            ys.append(Y)
            ws.append(W)

        if keep == 'last':
            is_better = True
        if keep == 'best_neg':
            is_better = Y_best is None or np.sum(Y_best < 0) > np.sum(Y < 0)
        else:
            is_better = tol_best > tol

        if is_better:
            Y_best, W_best = Y, W
            tol_best = tol  # need to keep memory of this

    # lists that records the progress of Y and W
    ys = []
    ws = []

    # best values and tolerance so far
    Y_best, W_best = None, None
    tol_best = np.inf

    t_neg = -np.inf if t_neg is None else t_neg

    # initialise
    n = X.shape[0]
    W = np.eye(n)
    t_max_arr = []

    # whiten the data
    Z = whiten(X) if whiten_mat else X
    Y = W @ Z

    i = 0
    while True:
        # compute the max torque of Y and corresponding indices
        t_max, ixs, _ = torque(Y)
        t_max_arr.append(t_max)

        set_best(Y, W, t_max)

        if t_max < t_tol or np.sum(Y_best < 0) < t_neg: # converged
            print('=' * 10)
            print('i = {}, t_max = {:.2f}, ixs = {}, #negative = {}'.format(i, t_max, ixs, np.sum(Y_best < 0)))
            print('Converged. Returning the reconstruction.')
            if return_all:
                return Y_best, W_best, Z, t_max_arr, ys, ws

            return Y_best, W_best, Z, t_max_arr

        if i > i_max  and t_max > t_max_arr[-2]:  # failed to converge
            print('=' * 10)
            print('i = {}, t_max = {:.2f}, ixs = {}, #negative = {}'.format(i, t_max, ixs, np.sum(Y_best < 0)))
            print(f'Error: Failed to converge. Returning current matrices.')
            if return_all:
                return Y_best, W_best, Z, t_max_arr, ys, ws

            return Y_best, W_best, Z, t_max_arr

        # print some information
        if (verbose > 0) and (i % print_all == 0):
            print('i = {}, t_max = {:.2f}, ixs = {}, #negative = {}'.format(i, t_max, ixs, np.sum(Y < 0)))

        # reduce to axis pair, find rotation angle and construct givens matrix
        Y_red = Y[ixs, :]
        opt_res = minimize_scalar(fun=obj_fun, bounds=(0, 2 * np.pi), method='bounded', args=Y_red)
        R = givens(n, ixs[0], ixs[1], opt_res['x'])

        # update the rotation matrix W and the reconstruction matrix Y
        W = R @ W
        Y = R @ Y

        i += 1