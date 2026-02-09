import pandas as pd
import numpy as np
import os

# --- MATH FUNCTIONS: Calculate Entropy and Information Gain ---

def calculate_entropy(target_column):
    """Calculates the entropy of the target column."""
    counts = target_column.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

def calculate_gain(df, feature, target='Keputusan'):
    """Calculates the Information Gain of a feature relative to the target."""
    total_entropy = calculate_entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        weighted_entropy += (len(subset) / len(df)) * calculate_entropy(subset[target])
    return total_entropy - weighted_entropy

# --- ID3 LOGIC: Recursive Tree Building ---

def build_tree(df, remaining_features, target='Keputusan', is_root=False):
    """Recursively builds the decision tree using the ID3 algorithm."""
    target_data = df[target]
    
    # Base Case 1: If all data points have the same answer (pure node)
    if len(target_data.unique()) <= 1:
        return target_data.iloc[0]
    
    # Base Case 2: If no features are left but node isn't pure, return majority vote
    if not remaining_features:
        return target_data.mode()[0]
    
    # Calculate Gain for all remaining features
    gain_scores = []
    for f in remaining_features:
        gain = calculate_gain(df, f, target)
        gain_scores.append((f, gain))
        
    # [FEATURE] Show Root Selection Logic
    if is_root:
        print("\n=== ROOT SELECTION (Why is this feature the root?) ===")
        print(f"Target Entropy: {calculate_entropy(target_data):.4f}\n")
        print(f"{'Feature':<15} | {'Information Gain':<15}")
        print("-" * 35)
        # Sort by gain descending for better readability
        sorted_scores = sorted(gain_scores, key=lambda x: x[1], reverse=True)
        for f, g in sorted_scores:
            print(f"{f:<15} | {g:.4f}")
        print("-" * 35)
        best_feature = sorted_scores[0][0]
        print(f">>> WINNER: '{best_feature}' becomes the Root.\n")
    else:
        # Normal selection (find max gain)
        best_feature = max(gain_scores, key=lambda x: x[1])[0]
    
    # Create the tree structure (Node)
    tree = {best_feature: {}}
    
    # Remove the used feature from the list for the next recursion
    new_features = [f for f in remaining_features if f != best_feature]
    
    # Create branches for each unique value of the best feature
    for value in df[best_feature].unique():
        subset = df[df[best_feature] == value]
        # Recursively build subtrees
        tree[best_feature][value] = build_tree(subset, new_features, target)
        
    return tree

# --- PREDICTION FUNCTION ---

def predict(tree, new_data):
    """Traverses the decision tree to predict the result for new data."""
    # If the current node is not a dictionary, it's a leaf node (the result)
    if not isinstance(tree, dict):
        return tree
    
    # Get the feature name for the current node
    feature = list(tree.keys())[0]
    
    # Get the value from the new data for this feature
    data_value = new_data.get(feature)
    
    # Traverse the branch that matches the value
    if data_value in tree[feature]:
        next_branch = tree[feature][data_value]
        return predict(next_branch, new_data)
    else:
        # Handle cases where the value was not present in the training data
        return "Unrecognized Data (Not found in training)"

# --- EXECUTION ---

# Load dataset from CSV
df = pd.read_csv('data/data.csv')

# Define target and initial features
target_column = 'Keputusan'
initial_features = [col for col in df.columns if col != target_column]

# Build the tree
decision_tree = build_tree(df, initial_features, target_column, is_root=True)

# Result Visualization: ASCII Tree
def display_tree(tree, indent="  "):
    """Prints the tree structure in a readable format."""
    if not isinstance(tree, dict):
        print(f" --> {tree}")
    else:
        feature = list(tree.keys())[0]
        print(f"\n{indent}[ {feature} ]")
        for value, subtree in tree[feature].items():
            print(f"{indent}  |-- {value}", end="")
            display_tree(subtree, indent + "    ")

print("=========================")
print("DECISION TREE STRUCTURE:")
display_tree(decision_tree)
print("\n=========================")

# --- PREDICTION SIMULATION ---
print("\nPREDICTION SIMULATION:")

# Sample new data (Scenario: Rainy, Cool, Normal, No)
test_data = {
    'Cuaca': 'Hujan',
    'Suhu': 'Dingin',
    'Kelembapan': 'Normal',
    'Berangin': 'Tidak'
}

prediction_result = predict(decision_tree, test_data)

print(f"Input Data: {test_data}")
print(f"Result: {prediction_result}")
