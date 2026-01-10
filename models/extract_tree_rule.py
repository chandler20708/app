from sklearn.tree import _tree

def extract_rules(tree, feature_names, class_names, breach_label="Breach", min_samples=15):
    tree_ = tree.tree_

    feature_name = [
        feature_names[i] if i >= 0 else "undefined"
        for i in tree_.feature
    ]

    breach_idx = class_names.index(breach_label)
    rules = []

    def recurse(node, conditions):
        is_leaf = (
            tree_.children_left[node] == _tree.TREE_LEAF and
            tree_.children_right[node] == _tree.TREE_LEAF
        )

        if not is_leaf:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            recurse(tree_.children_left[node],  conditions + [f"{name} ≤ {threshold:.1f}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.1f}"])
        else:
            samples = tree_.n_node_samples[node]
            value = tree_.value[node][0]
            total = value.sum()

            if total == 0 or samples < min_samples:
                return

            pred_idx = value.argmax()

            rules.append({
                "conditions": conditions,
                "predicted_class": class_names[pred_idx],
                "breach_rate": value[breach_idx] / total,  # overwrite meaning
                "samples": samples,
            })


    recurse(0, [])
    return rules
