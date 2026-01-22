import numpy as np

class MyDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, y, X_column, threshold):
        # حساب الـ Entropy للأب
        parent_entropy = self._entropy(y)

        # تقسيم البيانات
        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # حساب الـ Entropy للأبناء
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # الـ Information Gain هو الفرق
        return parent_entropy - child_entropy

    def fit(self, X, y):
        # هنا بنبني الشجرة (تبسيط للكود بتاعك)
        print("Training Decision Tree using Information Gain...")
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # منطق بناء الشجرة (Recursion)
        num_samples, num_features = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))

        # اختيار أفضل Split بناءً على الـ Information Gain
        best_feat, best_thresh, best_gain = None, None, -1
        for feat_idx in range(num_features):
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], threshold)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat_idx, threshold

        left_idx = np.where(X[:, best_feat] <= best_thresh)[0]
        right_idx = np.where(X[:, best_feat] > best_thresh)[0]
        
        left = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return (best_feat, best_thresh, left, right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat_idx, threshold, left, right = tree
        if x[feat_idx] <= threshold:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)