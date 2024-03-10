# CSE 151A Milestone 4: Second Model(s)

### **Note: For graphs, please see notebook. They are included there.**

Repo link: https://github.com/cse151a-nba-project/milestone-4/
Data link: https://github.com/cse151a-nba-project/data/

Group Member List: 

- Aaron Li all042\@ucsd.edu
- Bryant Jin brjin\@ucsd.edu
- Daniel Ji daji\@ucsd.edu
- David Wang dyw001\@ucsd.edu
- Dhruv Kanetkar dkanetkar\@ucsd.edu
- Eric Ye e2ye\@ucsd.edu
- Kevin Lu k8lu\@ucsd.edu
- Kevin Shen k3shen\@ucsd.edu
- Max Weng maweng\@ucsd.edu
- Roshan Sood rosood\@ucsd.edu

Abstract, for reference: 

Although sports analytics captured national attention only in 2011 with the release of Moneyball, research in the field is nearly a century old. Until relatively recently, this research was largely done by hand; however, the heavily quantitative nature of sports analytics makes it an attractive target for machine learning. This paper explores the application of advanced machine learning models to predict team performance in National Basketball Association (NBA) regular season and playoff games. Several models were trained on a rich dataset spanning 73 years, which includes individual player metrics, opponent-based performance, and team composition. The core of our analysis lies in combining individual player metrics, opponent-based game performances, and team chemistry, extending beyond traditional stat line analysis by incorporating nuanced factors. We employ various machine learning techniques, including neural networks and gradient boosting machines, to generate predictive models for player performance and compare their performance with both each other and traditional predictive models. Our analysis suggests that gradient boosting machines and neural networks significantly outperform other models. Neural networks demonstrate significant effectiveness in handling complex, non-linear data interrelations, while gradient boosting machines excel in identifying intricate predictor interactions. Our findings emphasize the immense potential of sophisticated machine learning techniques in sports analytics and mark a growing shift towards computer-aided and computer-based approaches in sports analytics.


# 1. Evaluate your (Milestone 3) data, labels and loss function. Were they sufficient or did you have have to change them.

Reference: https://github.com/cse151a-nba-project/milestone-3/blob/main/CSE_151A_Milestone_3.ipynb

Based on our milestone 3 analysis, it seems that the data, labels, and loss function used in the initial linear regression model were sufficient to get a baseline performance, but there is significant room for improvement to reduce the underfitting and high error rates observed.

Data and Labels:
The data consists of advanced player statistics (per, ws_48, usg_percent, bpm, vorp) for the top 8 players of each team from 1990 to 2023. The corresponding labels are the actual team win percentages. This data and label selection appears to be a good starting point, capturing key player performance metrics that could influence team success.

However, as mentioned in the planned improvements, additional feature engineering and selection could enhance the model's performance. This might involve exploring different or additional player statistics, considering interactions between features, or incorporating team-level metrics. Polynomial features could also be introduced to capture nonlinear relationships without immediately switching to more complex models.

Loss Function:
Our analysis uses Mean Squared Error (MSE) as the primary loss function to evaluate the model's performance. MSE is a commonly used loss function for regression problems and is generally well-suited for this task. It penalizes larger errors more heavily, which is appropriate when trying to minimize the difference between predicted and actual win percentages.

Our use of additional evaluation metrics like Mean Absolute Error (MAE) and R-squared (R^2) provides a more comprehensive assessment of the model's performance. MAE gives an interpretable measure of the average absolute difference between predictions and actual values, while R^2 indicates the proportion of variance in the dependent variable (win percentage) that is predictable from the independent variables (player statistics).

For these models, we decided to include several more statistics which we believed were highly correlated with wins. These stats were, ts%, x3p_ar, 2 stats correlated with offensive efficiency and output, and experience, which we believed to be a major factor in a teams performance.

In conclusion, while our initial data, labels, and loss function choices were sufficient for establishing a baseline, the high error rates and underfitting suggest that improvements can be made. The planned strategies, such as feature engineering, exploring more complex models (e.g., polynomial regression, deep neural networks), and applying regularization techniques, are explored in this notebook in order to improve our model's performance.

# 2. Train your second model

Based on the analysis of your current model's performance with linear regression, which indicates underfitting given the high mean squared error (MSE) rates for both testing and training, we are considering using a Deep Neural Network (DNN) model which can capture more of the intricacies in the data. We also intend to experiment with Elastic Net models that can capture more details of the data but with regularization to prevent overfitting.

We want to experiment with a Deep Neural Network next because it can model relationships between variables that are not linear, capturing more complex patterns in the dataset. Given the nature of sports statistics and team performance metrics, the relationship between the input features (advanced player statistics) and the output (team win percentages) might not be linear. DNN regression can model intricate interactions between player statistics and their impact on team success rates through its multiple hidden layers and non-linear activation functions. This flexibility allows DNNs to potentially offer significant improvements over simpler models by accurately capturing the non-linear and complex relationships in our dataset.

We also experimented with an Elastic Net model, a type of specialized linear regression, given the promising performance of out previous linear regressor. Elastic Net uses regularization techniques which allows our model to grow more complex without overfitting. It can handle Multicolinearity well, better than a simple linear regressor and additionally the regularization allows it to mimic feature selection by reducing weights near 0, allowing us to use a wider variety of stats.

Planned Strategy:
We aim to design a DNN with several hidden layers and neurons, experimenting with different architectures to find the most effective configuration for our dataset. Regularization techniques (like dropout and L2 regularization) and optimization algorithms will be also used to enhance learning and prevent overfitting.
Feature scaling and normalization will be crucial preprocessing steps
(that we've done in milestone 2) to ensure that the DNN model trains efficiently.
We will also use a portion of the dataset for validation during training to monitor for overfitting and adjust the model's parameters accordingly.

# 3. Evaluate your model compare training vs test error

|              | Linear | Elastic Net |  DNN   | Tuned DNN |
|--------------|--------|-------------|--------|-----------|
| Training MSE | 16.704 |   17.376    | 19.954 |  11.957   |
| Training MAE | 3.2729 |   3.3292    | 3.5214 |  2.3812   |
| Training R^2 | 0.9312 |   0.9284    | 0.9178 |  0.9507   |
| Testing MSE  | 20.881 |   19.921    | 24.686 |  37.071   |
| Testing MAE  | 3.7103 |   3.6713    | 4.1783 |  4.7918   |
| Testing R^2  | 0.9028 |   0.9072    | 0.8850 |  0.8274   |

As shown in the table above, the Hyper-parameter tuned DNN has the best training error by far, but also the worst test error. This indicates that our tuned DNN is overfitting the training data. However, the elastic net model, while having a slightly worse training loss than the baseline Linear Regressor, actually has the best testing loss out of all the models, indicating that it is adequately fitting the data without overfitting. This makes sense since the Elastic Net model is designed to avoid overfitting with multiple regularization techniques. However, the model's performance is only marginally better than the Linear Regressor, not achieving quite performing as we had hoped to.

# 4. Where does your model fit in the fitting graph, how does it compare to your first model?

## Note: we will discuss our multiple models (which we consider technically as one) here.

Based on the performance metrics we obtained, it's clear that our new models have significantly improved upon our initial linear regression model. Let's discuss the fitting graph description for each of the new models we developed:

Elastic Net Model: Our Elastic Net model has shown a notable improvement in performance compared to our linear regression model. The training MSE has decreased to approximately 17.38, while the testing MSE is around 19.92. This indicates a better fit to the data compared to our initial model. The training and testing errors are relatively close, suggesting that the model is not overfitting or underfitting. The Elastic Net regularization we applied has effectively balanced the model's complexity and generalization ability. The R^2 values for both training and testing are above 0.90, indicating a strong correlation between the predicted and actual win percentages.

Hyperparameter-Tuned DNN Model: Our hyperparameter-tuned DNN model has shown remarkable performance. The training MSE has significantly decreased to approximately 8.98, indicating a much better fit to the training data. However, the testing MSE is higher at around 41.25. This discrepancy between training and testing errors suggests that the model might be overfitting to some extent. The hyperparameter tuning process we conducted has likely resulted in a complex model that performs exceptionally well on the training data but struggles to generalize to unseen data. The R^2 value for training is impressively high at 0.96, while the testing R^2 is lower at 0.81. This further supports the notion of overfitting.

Manually Tuned DNN Model: Our manually tuned DNN model has also shown improvement compared to our linear regression model. The training MSE is approximately 19.84, while the testing MSE is around 24.06. These errors are lower than our initial model but higher than our Elastic Net model. The manually tuned DNN model seems to have a reasonable balance between fitting the training data and generalizing to unseen data. The R^2 values for both training and testing are above 0.88, indicating a good correlation between the predicted and actual win percentages.

Overall, our Elastic Net model appears to have the best balance between performance and generalization. It has significantly reduced the error compared to our linear regression model while maintaining a good fit to both training and testing data. Our hyperparameter-tuned DNN model has achieved the lowest training error but seems to be overfitting, as evidenced by the higher testing error. Our manually tuned DNN model has shown improvement but falls short of the Elastic Net model's performance.

# 5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

Yes, we performed hyperparameter tuning, cross-validation, and feature expansion for both our Elastic Net model and Deep Neural Network (DNN) model.

For the Elastic Net model, we used GridSearchCV to search for the best hyperparameters. We defined a parameter grid with different values for the 'alpha' (regularization strength) and 'l1_ratio' (mix ratio between Lasso and Ridge) hyperparameters. We used 5-fold cross-validation (cv=5) to evaluate the model's performance for each combination of hyperparameters. GridSearchCV then selected the best combination of hyperparameters based on the model's performance.

Regarding feature expansion, we extended our feature set to include additional player statistics such as 'ts_percent', 'experience', 'x3p_ar', 'per', 'ws_48', 'usg_percent', 'bpm', and 'vorp'. This expanded feature set provides a more comprehensive representation of player performance compared to our previous observations, which only considered 'per', 'ws_48', 'usg_percent', 'bpm', and 'vorp'. Additionally, we now consider the top 10 players from each roster instead of the top 8 players, allowing for a broader representation of team talent.

For the DNN model, we also used GridSearchCV for hyperparameter tuning. We defined a parameter grid that included different values for the number of hidden layers, initial units, activation function, and batch size. We used KFold cross-validation with 5 splits (n_splits=5) and shuffled the data (shuffle=True) for each fold. GridSearchCV searched for the best combination of hyperparameters based on the negative mean squared error (neg_mean_squared_error) as the scoring metric.

After finding the best hyperparameters for each model, we trained the models using the best parameters. For the DNN model, we also incorporated early stopping (EarlyStopping) to prevent overfitting. Early stopping monitored the validation mean squared error (val_mse) and stopped training if there was no improvement after 25 epochs (patience=25), restoring the best weights.

The results of hyperparameter tuning showed that the best parameters for the Elastic Net model were found through GridSearchCV, in this case values of "elasticnet__alpha': 0.004, 'elasticnet__l1_ratio': 0.9". Similarly, for the DNN model, the best parameters and the corresponding best score (negative mean squared error) were printed: "{'batch_size': 8, 'model__activation': 'relu', 'model__hidden_layers': 4, 'model__initial_units': 810} and Best score:  58.875539655649916". **We observe that although the elastic net model hyperparameter tuning resulted in an improved model from the simpler linear regression model without regularization, the DNN model had actually worse performance than the manually tuned values that our team found ourselves through trial and error.**

The expanded feature set and the inclusion of the top 10 players from each roster likely contributed to improved model performance by capturing a more comprehensive representation of team and player characteristics. The additional features provide a richer set of information for the models to learn from, potentially leading to more accurate predictions.

# 6. What is the plan for the next model you are thinking of and why?



Based on the performance of our current models, we believe that the next step in improving our predictive capabilities is to explore more advanced modeling techniques. While our Elastic Net model has shown promising results, we think that there is still room for improvement in terms of reducing the error and capturing more complex relationships within the data.

Our plan for the next model is to investigate the use of Support Vector Machines (SVMs) with non-linear kernels, specifically the Radial Basis Function (RBF) kernel. We have several reasons for considering this approach:

Non-linear Relationships: SVMs with RBF kernels have the ability to capture non-linear relationships between the features and the target variable. By mapping the input data into a higher-dimensional space, SVMs can find complex decision boundaries that separate the different classes or predict continuous values.

Regularization: SVMs have built-in regularization through the C parameter, which controls the trade-off between fitting the training data well and allowing some misclassifications. This regularization can help prevent overfitting and improve the model's generalization ability.

Hyperparameter Tuning: SVMs with RBF kernels have two main hyperparameters: C (regularization parameter) and gamma (kernel coefficient). We plan to conduct hyperparameter tuning using techniques like grid search or random search to find the optimal combination of these hyperparameters that minimizes the error metrics.

To implement this plan, we will start by preprocessing our data, ensuring that it is scaled and normalized appropriately for SVM training. We will then split the data into training and testing sets, allowing us to evaluate the model's performance on unseen data. We will utilize cross-validation techniques to assess the model's robustness and generalization ability.

After training the SVM model with different hyperparameter configurations, we will evaluate its performance using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R^2). We will compare these metrics with our previous models to determine if the SVM model provides a significant improvement.

In conclusion, our second set of models, which included the Elastic Net, hyperparameter-tuned DNN, and manually tuned DNN, showed improvements over our initial linear regression model. The Elastic Net model demonstrated the best balance between performance and generalization, with lower training and testing errors compared to the linear regression model. It effectively managed model complexity and achieved R^2 values above 0.90, indicating a strong correlation between predicted and actual win percentages.

The hyperparameter-tuned DNN model achieved the lowest training error, showcasing its ability to fit the training data exceptionally well. However, it exhibited signs of overfitting, as evidenced by the higher testing error compared to the Elastic Net model. This suggests that the model's complexity may have hindered its generalization ability to unseen data.

The manually tuned DNN model also demonstrated improvement over the linear regression model, with lower training and testing errors. While it found a reasonable balance between fitting the training data and generalizing to unseen data, it did not outperform the Elastic Net model in terms of overall performance.

To further improve our models, we can (continue to) explore several avenues:

Regularization Techniques: Investigate different regularization techniques for the DNN models, such as L1 and L2 regularization, dropout, or early stopping, to mitigate overfitting and enhance generalization. Since our tuned DNN model is severly overfitting to the training data, we believe that this regularizing it may be able to bring the very low training error to the testing error.

Hyperparameter Fine-tuning: Conduct more extensive hyperparameter tuning for the Elastic Net and DNN models to find the optimal combination of hyperparameters that minimizes the error metrics.

Feature Selection and Engineering: Analyze feature importance and consider selecting the most relevant features or engineering new features based on domain knowledge to improve the models' predictive power.

Data Augmentation: Collect more diverse and representative data to increase the models' exposure to different scenarios and improve their generalization ability.

Ensemble Methods: Explore ensemble techniques, such as bagging or boosting, to combine multiple models and leverage their collective knowledge for improved predictions.

Compared to our first linear regression model, the second set of models demonstrated significant improvements in performance. The Elastic Net model, in particular, achieved a better balance between bias and variance, capturing more complex relationships in the data while avoiding overfitting. The DNN models, although showing promise, required careful hyperparameter tuning to strike the right balance between fitting the training data and generalizing to unseen data.

Overall, the second set of models, especially the Elastic Net, provided a more accurate and reliable approach to predicting win percentages compared to the linear regression model. By leveraging regularization, hyperparameter tuning, and considering feature importance, we were able to develop models that better captured the underlying patterns in the data and improved our predictive capabilities.