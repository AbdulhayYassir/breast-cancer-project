import numpy as np
import joblib

model = joblib.load('models/breast_cancer_model.pkl')

def print_tree(tree, indent=" "):
    if not isinstance(tree, tuple):
        print(indent + "Predict:", tree)
        return
    
    feat, thresh, left, right = tree
    print(f"{indent}[Feature {feat} <= {thresh:.2f}]")
    print(indent + "  L->", end="")
    print_tree(left, indent + "    ")
    print(indent + "  R->", end="")
    print_tree(right, indent + "    ")

print_tree(model.tree)


from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42) # عشان نضمن إن الداتا مش مترتبة (خبيث ورا بعضه)
# وبعدين نادى على model.fit(X, y)