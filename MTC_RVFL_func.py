import numpy as np
import time

def _one_hot_encode(labels, num_classes):
    """Convert 1-D label vector to (N, num_classes) one-hot matrix."""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1
    return one_hot


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


ACTIVATIONS = {
    1: _sigmoid,
    2: np.sin,
    'tribas': lambda x: np.maximum(0, 1 - np.abs(x)),
    3: lambda x: np.maximum(0, 1 - np.abs(x)),
    'radbas': lambda x: np.exp(-x**2),
    4: lambda x: np.exp(-x**2),
    'tansig': np.tanh,
    'tanh': np.tanh,
    5: np.tanh,
    'relu': lambda x: np.maximum(0.0, x),
    6: lambda x: np.maximum(0.0, x)
}


def _generate_hidden(X, W, b, activation='sigmoid'):
    """RVFL hidden representation: [X, act(X W + b), 1]."""
    act = ACTIVATIONS[activation]
    H_hidden = act(X @ W + b)
    return np.hstack([X, H_hidden, np.ones((X.shape[0], 1))])


def _huber_weight(e, p):
    w = np.ones_like(e)
    mask = np.abs(e) > p
    w[mask] = p / np.maximum(np.abs(e[mask]), 1e-12)
    return w


def _cauchy_weight(e, p):
    return 1.0 / (1.0 + (e / p) ** 2)


def _bisquare_weight(e, p):
    r = e / p
    w = (1.0 - r ** 2) ** 2
    w[np.abs(r) > 1.0] = 0.0
    return w


def _welsch_weight(e, p):
    return np.exp(-((e / p) ** 2))


M_ESTIMATORS = {
    'huber':    (_huber_weight,    1.345),
    'cauchy':   (_cauchy_weight,   2.385),
    'bisquare': (_bisquare_weight, 4.685),
    'welsch':   (_welsch_weight,   2.985),
}


def _mest_weights(residual_matrix, m_type):
    """Per-sample M-estimator weights θ_i, using adaptive threshold."""
    mag = np.linalg.norm(residual_matrix, axis=1)
    fn, tuning = M_ESTIMATORS[m_type]
    p = max(tuning * np.median(mag) / 0.6745, 1e-8)
    return fn(mag, p)


def _ridge_solve(H, Y, lam):
    N, D = H.shape
    if D < N:
        return np.linalg.solve(lam * np.eye(D) + H.T @ H, H.T @ Y)
    return H.T @ np.linalg.solve(lam * np.eye(N) + H @ H.T, Y)



def _mtc_fit(H, Y, lam, sigma, sigma_in, sigma_out, max_iter, tol,
             mode=None, m_estimator=None):
    N, D = H.shape

    beta = _ridge_solve(H, Y, lam)

    for t in range(1, max_iter + 1):
        w_bar_sq = (sigma_out ** 2) / (sigma_in ** 2) + float(np.sum(beta ** 2))

        E = H @ beta - Y
        r2 = np.sum(E ** 2, axis=1)
        lam_M = np.exp(-r2 / (2.0 * sigma ** 2 * w_bar_sq))

        if mode is None:
            combined = lam_M
            s_val = float(np.sum(r2 * lam_M / w_bar_sq))
        elif mode == 'add':
            theta = _mest_weights(E, m_estimator)
            combined = lam_M + theta
            s_val = float(np.sum(r2 * lam_M / w_bar_sq))
        elif mode == 'mult':
            theta = _mest_weights(E, m_estimator)
            combined = theta * lam_M
            s_val = float(np.sum(theta * r2 * lam_M / w_bar_sq))
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        gamma = lam * sigma ** 2 * w_bar_sq - s_val
        gamma = max(gamma, 1e-6) 
        if D < N:
            # (H^T @ diag(combined) @ H) + gamma * I
            M = (H.T * combined) @ H + gamma * np.eye(D)
            beta_new = np.linalg.solve(M, (H.T * combined) @ Y)
        else:
            M = (H * combined[:, None]) @ H.T + gamma * np.eye(N)
            beta_new = H.T @ np.linalg.solve(M, combined[:, None] * Y)

        diff = float(np.sum((beta_new - beta) ** 2))
        beta = beta_new
        if diff < tol:
            return beta, t

    return beta, max_iter


def _run_rvfl(trainX, trainY, testX, testY, option, No_of_class, solver):
    N = option['N']
    activation = option.get('activation', 'sigmoid')

    trainY_oh = _one_hot_encode(trainY, No_of_class)

    Nsample, Nfea = trainX.shape
    s = 1.0
    
    W = np.random.uniform(-s, s, (Nfea, N))
    b = np.random.uniform(-s, s, (1, N))

    start = time.time()
    H_train = _generate_hidden(trainX, W, b, activation)
    beta, n_iter = solver(H_train, trainY_oh, option)
    pred_train = H_train @ beta
    train_time = time.time() - start

    train_acc = np.mean(np.argmax(pred_train, axis=1) == trainY) * 100

    start = time.time()
    H_test = _generate_hidden(testX, W, b, activation)
    pred_test = H_test @ beta
    test_time = time.time() - start

    test_acc = np.mean(np.argmax(pred_test, axis=1) == testY) * 100
    return train_acc, test_acc, train_time, test_time, n_iter


def RVFL_Model(trainX, trainY, testX, testY, option, No_of_class):
    def solver(H, Y, opt):
        return _ridge_solve(H, Y, 1.0 / opt['C']), 0

    train_acc, test_acc, t_tr, t_te, _ = _run_rvfl(
        trainX, trainY, testX, testY, option, No_of_class, solver)
    return train_acc, test_acc, t_tr, t_te


def MTC_RVFL_Model(trainX, trainY, testX, testY, option, No_of_class):
    def solver(H, Y, opt):
        return _mtc_fit(
            H, Y,
            lam=1.0 / opt['C'],
            sigma=opt.get('sigma', 0.5),
            sigma_in=opt.get('sigma_in', 0.1),
            sigma_out=opt.get('sigma_out', 0.1),
            max_iter=opt.get('max_iter', 30),
            tol=opt.get('tol', 1e-3),
            mode=None,
        )

    train_acc, test_acc, t_tr, t_te, _ = _run_rvfl(
        trainX, trainY, testX, testY, option, No_of_class, solver)
    return train_acc, test_acc, t_tr, t_te


def MMTC_RVFLa_Model(trainX, trainY, testX, testY, option, No_of_class):
    def solver(H, Y, opt):
        return _mtc_fit(
            H, Y,
            lam=1.0 / opt['C'],
            sigma=opt.get('sigma', 0.5),
            sigma_in=opt.get('sigma_in', 0.1),
            sigma_out=opt.get('sigma_out', 0.1),
            max_iter=opt.get('max_iter', 30),
            tol=opt.get('tol', 1e-3),
            mode='add',
            m_estimator=opt.get('m_estimator', 'huber'),
        )

    train_acc, test_acc, t_tr, t_te, _ = _run_rvfl(
        trainX, trainY, testX, testY, option, No_of_class, solver)
    return train_acc, test_acc, t_tr, t_te


def MMTC_RVFLb_Model(trainX, trainY, testX, testY, option, No_of_class):
    def solver(H, Y, opt):
        return _mtc_fit(
            H, Y,
            lam=1.0 / opt['C'],
            sigma=opt.get('sigma', 0.5),
            sigma_in=opt.get('sigma_in', 0.1),
            sigma_out=opt.get('sigma_out', 0.1),
            max_iter=opt.get('max_iter', 30),
            tol=opt.get('tol', 1e-3),
            mode='mult',
            m_estimator=opt.get('m_estimator', 'welsch'),
        )

    train_acc, test_acc, t_tr, t_te, _ = _run_rvfl(
        trainX, trainY, testX, testY, option, No_of_class, solver)
    return train_acc, test_acc, t_tr, t_te