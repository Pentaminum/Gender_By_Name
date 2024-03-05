import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt


"""
Part 1: Preliminaries
"""

# Load the dataset
df = pd.read_csv("spam_messages.csv")

# Features and Labels
X = df["Contents"] # Features(names)
y = df["Labels"] # Labels(genders)

# Split the dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)   
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Map labels to binary values
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_test_binary = y_test.map({'ham': 0, 'spam': 1})

# Plot precision-recall curve
precision, recall, _ = precision_recall_curve(y_test_binary, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png') 


# """
# Part 2: Brute force LOO influence
# """

# Function to compute LOO influence
def compute_loo_influence(X_train, y_train, X_test, y_test, pipeline):
    loo_influences = []
    
    for i in range(10):  # From 10 training data points
        # Remove i-th data point from the training set
        X_train_loo = X_train.drop(X_train.index[i])
        y_train_loo = y_train.drop(y_train.index[i])
        
        # Retrain the model on the modified training set
        pipeline.fit(X_train_loo, y_train_loo)
        
        # Evaluate the model's performance on the test set
        y_pred_loo = pipeline.predict(X_test)
        
        # Compute the difference in accuracy
        accuracy_loo = accuracy_score(y_test, y_pred_loo)
        loo_influence = accuracy - accuracy_loo
        
        loo_influences.append(loo_influence)
    
    return loo_influences

# Select 10 random training data points that contain both ham and spam messages
def select_training_data(X_train, y_train, num_samples=10):
    ham_indices = y_train[y_train == 'ham'].index
    spam_indices = y_train[y_train == 'spam'].index
    selected_indices = np.concatenate([np.random.choice(ham_indices, size=num_samples // 2, replace=False),
                                       np.random.choice(spam_indices, size=num_samples // 2, replace=False)])
    X_train_selected = X_train.loc[selected_indices]
    y_train_selected = y_train.loc[selected_indices]
    return X_train_selected, y_train_selected

X_train_selected, y_train_selected = select_training_data(X_train, y_train, num_samples=10)

# Compute LOO influence
loo_influences = compute_loo_influence(X_train_selected, y_train_selected, X_test, y_test, pipeline)

# Report influence scores
print("LOO Influence Scores:")
for i, influence in enumerate(loo_influences):
    print(f"Observation {i+1}: {influence}")

"""
Part 3: Group-level influence
"""

# Function to compute LEGO influence for different group sizes
def compute_lego_influence(X_train, y_train, X_test, y_test, pipeline):
    lego_influences = []
    
    # Different group sizes (e.g., 10%, 20%, ..., 100% of the training data)
    group_sizes = np.arange(0.1, 1.1, 0.1)
    
    for size in group_sizes:
        # Determine the number of data points to select for the current group size
        num_samples = int(len(X_train) * size)
        
        # Select data points for the current group size
        random_indices = np.random.choice(X_train.index, size=num_samples, replace=False)
        X_train_group = X_train.loc[random_indices]
        y_train_group = y_train.loc[random_indices]

        # Remove the entire group from the training set
        X_train_removed = X_train.drop(X_train_group.index)
        y_train_removed = y_train.drop(y_train_group.index)

        if len(X_train_removed) == 0:  # Skip if the training set is empty
            lego_influences.append(accuracy)
            continue
        
        # Retrain the model on the modified training set
        pipeline.fit(X_train_removed, y_train_removed)
        
        # Evaluate the model's performance on the test set
        y_pred_removed = pipeline.predict(X_test)
        
        # Compute the difference in accuracy
        accuracy_removed = accuracy_score(y_test, y_pred_removed)
        influence = accuracy - accuracy_removed
        
        lego_influences.append(influence)
    
    return group_sizes, lego_influences

# Compute LEGO influence for different group sizes
group_sizes, lego_influences = compute_lego_influence(X_train, y_train, X_test, y_test, pipeline)

# Report influence scores for each group
print("LEGO Influence Scores for Different Group Sizes:")
for size, influence in zip(group_sizes, lego_influences):
    print(f"Group Size: {size * 100}%\t Influence Score: {influence}")

# Plot group size compared with influence
plt.figure(figsize=(8, 6))
plt.plot(group_sizes * 100, lego_influences, marker='o', color='blue')
plt.xlabel('Group Size (%)')
plt.ylabel('Influence Score')
plt.title('Group Size vs. LEGO Influence')
plt.grid(True)
plt.savefig('LEGO_curve.png') 

"""
Part 4: Shapley Values
"""
def estimate_shapley_values(X_train, y_train, pipeline):
    num_permutations = 10
    shapley_values = []

    for _ in range(num_permutations):
        # Shuffle the training data
        shuffled_indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train.iloc[shuffled_indices]
        y_train_shuffled = y_train.iloc[shuffled_indices]

        # Initialize the marginal contributions for this permutation
        marginal_contributions = []

        # Train the model on the entire shuffled training data
        pipeline.fit(X_train_shuffled, y_train_shuffled)
        y_pred_full = pipeline.predict(X_train_shuffled)
        accuracy_full = accuracy_score(y_train_shuffled, y_pred_full)

        # Iterate over the shuffled data to compute marginal contributions
        for i in range(len(X_train)):
            print(str(_) + ", " + str(i))
            # Train the model on a subset of the shuffled training data
            X_subset = X_train_shuffled.iloc[:i+1]
            y_subset = y_train_shuffled.iloc[:i+1]
            if len(set(y_subset)) == 1:
                continue
            pipeline.fit(X_subset, y_subset)

            # Predict on the subset
            y_pred_subset = pipeline.predict(X_subset)
            
            # Compute accuracy on the subset
            accuracy_subset = accuracy_score(y_subset, y_pred_subset)
            
            # Compute the marginal contribution
            marginal_contribution = accuracy_subset - accuracy_full
            
            marginal_contributions.append(marginal_contribution)

        # Average the marginal contributions across the shuffled permutations
        shapley_values.append(np.mean(marginal_contributions))

    return shapley_values

# Estimate Shapley values
shapley_values = estimate_shapley_values(X_train, y_train, pipeline)

# Plot the distribution of Shapley values
plt.figure(figsize=(8, 6))
plt.hist(shapley_values, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Shapley Value')
plt.ylabel('Frequency')
plt.title('Distribution of Shapley Values')
plt.grid(True)
plt.savefig('Shapley_plot.png') 
