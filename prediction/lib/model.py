# -*- coding: utf-8 -*-
"""Hand-rolled gradient boosted decision trees (logistic loss) + a LogReg control.

Why hand-rolled: on this machine both sklearn and lightgbm are unusable because of a
pyarrow-NumPy ABI conflict, and the whole POC has so far been dependency-free and runs
straight under `python3`. At 600 training pairs, GBDT itself is simple enough that a
hand-rolled version stays controllable and yields clean gain-based feature importances.

⚠️ A hand-rolled implementation risks hidden bugs. Mitigations:
  1. `selftest()` checks on synthetic data that it learns a known signal and that
     train/holdout performance is sane
  2. a logistic regression is trained alongside as a control -- if GBDT cannot beat it,
     the implementation is broken
numpy is the only dependency.
"""
import json

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _Node:
    __slots__ = ('feat', 'thr', 'left', 'right', 'value')

    def __init__(self):
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.value = 0.0


class _Tree:
    """Regression tree on the second-order approximation over (grad, hess), XGBoost style."""

    def __init__(self, max_depth=3, min_child_weight=1.0, lam=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lam = lam
        self.gamma = gamma
        self.root = None
        self.gain_by_feature = None

    def _leaf_value(self, g, h):
        return -g.sum() / (h.sum() + self.lam)

    def _best_split(self, X, g, h, feats):
        G, H = g.sum(), h.sum()
        base = G * G / (H + self.lam)
        best = (0.0, None, None)
        for j in feats:
            order = np.argsort(X[:, j], kind='mergesort')
            xs, gs, hs = X[order, j], g[order], h[order]
            cg = np.cumsum(gs)
            ch = np.cumsum(hs)
            # Only split where the value actually changes
            valid = np.nonzero(xs[1:] != xs[:-1])[0]
            if valid.size == 0:
                continue
            GL, HL = cg[valid], ch[valid]
            GR, HR = G - GL, H - HL
            ok = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            if not ok.any():
                continue
            gain = (GL * GL / (HL + self.lam) + GR * GR / (HR + self.lam) - base) / 2.0
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                i = valid[k]
                best = (float(gain[k]), j, float((xs[i] + xs[i + 1]) / 2.0))
        return best

    def _build(self, X, g, h, depth, feats):
        node = _Node()
        if depth >= self.max_depth or X.shape[0] < 2:
            node.value = self._leaf_value(g, h)
            return node
        gain, j, thr = self._best_split(X, g, h, feats)
        if j is None or gain <= self.gamma:
            node.value = self._leaf_value(g, h)
            return node
        mask = X[:, j] <= thr
        if mask.all() or (~mask).all():
            node.value = self._leaf_value(g, h)
            return node
        self.gain_by_feature[j] += gain
        node.feat, node.thr = j, thr
        node.left = self._build(X[mask], g[mask], h[mask], depth + 1, feats)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth + 1, feats)
        return node

    def fit(self, X, g, h, feats):
        self.gain_by_feature = np.zeros(X.shape[1])
        self.root = self._build(X, g, h, 0, feats)
        return self

    def predict(self, X):
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            nd = self.root
            while nd.feat is not None:
                nd = nd.left if X[i, nd.feat] <= nd.thr else nd.right
            out[i] = nd.value
        return out

    @staticmethod
    def _node_to_dict(node):
        if node.feat is None:
            return {'value': float(node.value)}
        return {
            'feat': int(node.feat), 'thr': float(node.thr),
            'left': _Tree._node_to_dict(node.left),
            'right': _Tree._node_to_dict(node.right),
        }

    @staticmethod
    def _node_from_dict(data):
        node = _Node()
        if 'feat' not in data:
            node.value = float(data['value'])
            return node
        node.feat, node.thr = int(data['feat']), float(data['thr'])
        node.left = _Tree._node_from_dict(data['left'])
        node.right = _Tree._node_from_dict(data['right'])
        return node

    def to_dict(self):
        return {'root': self._node_to_dict(self.root)}

    @classmethod
    def from_dict(cls, data):
        tree = cls()
        tree.root = cls._node_from_dict(data['root'])
        return tree


class GBDT:
    def __init__(self, n_trees=120, lr=0.1, max_depth=3, min_child_weight=1.0,
                 lam=1.0, subsample=0.8, colsample=0.8, seed=0):
        self.n_trees = n_trees
        self.lr = lr
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lam = lam
        self.subsample = subsample
        self.colsample = colsample
        self.seed = seed
        self.trees = []
        self.base = 0.0
        self.importance_ = None

    def fit(self, X, y, w=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
        rng = np.random.default_rng(self.seed)
        p0 = np.clip((y * w).sum() / w.sum(), 1e-6, 1 - 1e-6)
        self.base = float(np.log(p0 / (1 - p0)))
        F = np.full(y.shape, self.base)
        self.trees = []
        self.importance_ = np.zeros(X.shape[1])
        n, m = X.shape
        for _ in range(self.n_trees):
            p = sigmoid(F)
            g = w * (p - y)
            h = w * np.clip(p * (1 - p), 1e-6, None)
            ridx = rng.choice(n, max(2, int(n * self.subsample)), replace=False)
            feats = rng.choice(m, max(1, int(m * self.colsample)), replace=False)
            t = _Tree(self.max_depth, self.min_child_weight, self.lam)
            t.fit(X[ridx], g[ridx], h[ridx], feats)
            F += self.lr * t.predict(X)
            self.importance_ += t.gain_by_feature
            self.trees.append(t)
        s = self.importance_.sum()
        if s > 0:
            self.importance_ = self.importance_ / s
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.base)
        for t in self.trees:
            F += self.lr * t.predict(X)
        return F

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def to_dict(self):
        """Serialize the fitted scorer, not merely its feature importance."""
        return {
            'format': 'hktv-gbdt-v1',
            'params': {
                'n_trees': self.n_trees, 'lr': self.lr, 'max_depth': self.max_depth,
                'min_child_weight': self.min_child_weight, 'lam': self.lam,
                'subsample': self.subsample, 'colsample': self.colsample, 'seed': self.seed,
            },
            'base': float(self.base),
            'importance': self.importance_.tolist() if self.importance_ is not None else None,
            'trees': [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get('format') != 'hktv-gbdt-v1':
            raise ValueError('unsupported GBDT format')
        model = cls(**data['params'])
        model.base = float(data['base'])
        model.importance_ = (np.asarray(data['importance'], dtype=float)
                             if data.get('importance') is not None else None)
        model.trees = [_Tree.from_dict(t) for t in data['trees']]
        return model

    def save(self, path, metadata=None):
        payload = self.to_dict()
        payload['metadata'] = metadata or {}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


class LogReg:
    """L2-regularised logistic regression (the control).

    Simple enough to be almost impossible to get wrong, which is what makes it a valid
    check that the GBDT is not broken.
    """

    def __init__(self, l2=1.0, iters=300, lr=0.5, seed=0):
        self.l2, self.iters, self.lr = l2, iters, lr
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def fit(self, X, y, w=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        sw = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1])
        self.b = 0.0
        for _ in range(self.iters):
            p = sigmoid(Z @ self.w + self.b)
            gw = Z.T @ (sw * (p - y)) / sw.sum() + self.l2 * self.w
            gb = float((sw * (p - y)).sum() / sw.sum())
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return sigmoid(Z @ self.w + self.b)


def auc(y, p):
    """ROC-AUC in Mann-Whitney U form, with tie handling."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Tie handling: equal values share the mean rank
    allv = np.concatenate([pos, neg])[order]
    i = 0
    r = ranks[order]
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        if j > i:
            r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def selftest():
    """Verify on synthetic data that GBDT learns a known non-linear signal without degrading."""
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 6))
    # Ground-truth rule: XOR-shaped non-linearity (logistic regression cannot learn it,
    # GBDT should)
    logit = 3.0 * (X[:, 0] > 0) * (X[:, 1] > 0) - 1.5
    y = (rng.uniform(size=n) < sigmoid(logit)).astype(float)
    tr, te = slice(0, 1400), slice(1400, n)
    g = GBDT(n_trees=150, max_depth=3, seed=1).fit(X[tr], y[tr])
    lr = LogReg().fit(X[tr], y[tr])
    a_g = auc(y[te], g.predict_proba(X[te]))
    a_l = auc(y[te], lr.predict_proba(X[te]))
    a_bayes = auc(y[te], sigmoid(logit[te]))      # from true probs -> AUC ceiling here
    imp = g.importance_
    true_share = imp[0] + imp[1]
    # Criteria: (1) reach at least 97% of the Bayes ceiling  (2) clearly beat the logistic
    #           regression, which cannot learn XOR  (3) the true-signal features take
    #           clearly more importance than chance (6 features, chance baseline 2/6=0.33;
    #           colsample=0.8 leaks some gain to the noise features, so the bar is 1.5x
    #           chance)
    ok = (a_g >= 0.97 * a_bayes) and (a_g > a_l + 0.05) and (true_share > 1.5 * 2 / 6)
    print('selftest: GBDT AUC=%.3f | Bayes ceiling=%.3f (reached %.1f%%) | LogReg AUC=%.3f'
          % (a_g, a_bayes, a_g / a_bayes * 100, a_l))
    print('          true-signal importance share=%.2f (chance baseline 0.33)  ->  %s'
          % (true_share, 'PASS' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if selftest() else 1)
