from sklearn.svm import SVR
import pickle


def SVRModel(cfg, load_model=False, filename=None, X=None, y=None):
    if load_model:
        loaded_model = pickle.load(open(filename, "rb"))
        return loaded_model

    else:
        model = SVR(
            kernel=cfg.params.kernel, degree=cfg.params.degree,
            gamma=cfg.params.gamma, coef0=cfg.params.coef0,
            tol=cfg.params.tol, C=cfg.params.C,
            epsilon=cfg.params.epsilon, shrinking=cfg.params.shrinking,
            cache_size=cfg.params.cache_size, verbose=cfg.params.verbose,
            max_iter=cfg.params.max_iter,
        )
        model.fit(X, y)
        return model
