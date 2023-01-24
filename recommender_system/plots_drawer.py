import matplotlib.pyplot as plt

# For user-based k-nearest neighbours
# train test split 100k ratings
RMSE = [1.0060]
MAE = [0.7775]

# For Singular Value Decomposition
# train test split 100k ratings
RMSE = [0.8817]
MAE = [0.6777]

# For average-based with social regularization
# train test split 100k ratings
RMSE = [0.8880]
MAE = [0.6780]

# For individual-based with social regularization
# train test split 100k ratings
RMSE = [0.8798]
MAE = [0.6764]

# For user-based k-nearest neighbours
# 5-fold cross validation 100k ratings
RMSE = [0.9987]
MAE = [0.7731]

# For Singular Value Decomposition
# 5-fold cross validation 100k ratings
RMSE = [0.8782]
MAE = [0.6752]

# For average-based with social regularization
# 5-fold cross validation 100k ratings
RMSE = [0.8783]
MAE = [0.6748]

# For individual-based with social regularization
# 5-fold cross validation 100k ratings
RMSE = [0.8768]
MAE = [0.6741]

# Bar chart for average RMSE and MAE for each of the KNN, SVD, ABSR, IBSR (train-test split at 0.25)
left = [1, 2, 4, 5, 7, 8, 10, 11]

height = [1.0060, 0.7775, 0.8817, 0.6777, 0.8880, 0.6780, 0.8798, 0.6764]

tick_label = ['RMSE', 'MAE', 'RMSE', 'MAE', 'RMSE', 'MAE', 'RMSE', 'MAE']

plt.bar(left, height, tick_label = tick_label,
        width = 0.8)
plt.xlabel('         k-nn                        SVD              average-based      individual-based', loc='left')
plt.title('Accuracy metrics, using a train-test split at 0.25')
count = 0
for i in left:
    plt.text(i - 0.5, height[count], height[count])
    count += 1

plt.show()

# Bar chart for average RMSE and MAE for each of the KNN, SVD, ABSR, IBSR (5-fold cross validation)
left = [1, 2, 4, 5, 7, 8, 10, 11]

height = [0.9987, 0.7731, 0.8782, 0.6752, 0.8783, 0.6748, 0.8768, 0.6741]

tick_label = ['RMSE', 'MAE', 'RMSE', 'MAE', 'RMSE', 'MAE', 'RMSE', 'MAE']

plt.bar(left, height, tick_label = tick_label,
        width = 0.8)

plt.xlabel('         k-nn                        SVD              average-based      individual-based', loc='left')
plt.title('Accuracy metrics, using 5-fold cross validation')
count = 0
for i in left:
    plt.text(i - 0.5, height[count], height[count])
    count += 1
plt.show()

if __name__ == '__main__':
    print("Plotting...")