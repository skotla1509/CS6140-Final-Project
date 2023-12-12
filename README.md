# Project Overview
## The Challenge
Customer relationship management (CRM) is becoming the focus of an increasing number of businesses these days. An example of this trend can be seen with major telecom giants like Verizon and AT&T, who invest heavily in CRM to enhance customer loyalty and reduce attrition. Given the saturated markets and fierce competition, retaining, and satisfying existing customers has proven to be more cost-effective than continuously targeting new ones who may have higher churn rates. Churn, in this context, refers to the rate at which customers discontinue utilizing a business's service or product over a given period.

The critical challenge is to construct a model to identify customers with a higher likelihood of churning. There are numerous predictive modeling techniques available to accomplish this. Our project delves into analyzing four of these techniques namely, Logistic Regression (LR), Support Vector Machines (SVM), Random Forests (RF), and Gradient Boosting Machines (GBM), comparing their efficacy using performance measures such as Accuracy, Precision, Recall, F1 score, AUC, and ROC.

## Significance
The problem of customer churn in the context of CRM is of vital interest for several compelling reasons. Firstly, it directly impacts a company's sustainability and profitability. In highly competitive markets like telecommunications, retaining existing customers is more cost-efficient than acquiring new ones. This is especially relevant in saturated markets where the potential for new customer acquisition is limited. Additionally, understanding and predicting churn helps businesses improve their services and offerings, leading to enhanced customer satisfaction and loyalty. This not only benefits the companies but also contributes to a healthier, more competitive market environment, ultimately benefiting consumers.

## Proposed Approach
To tackle the problem of predicting customer churn, a methodical and layered approach is taken. The proposed strategy involves initially employing simpler algorithms such as Logistic Regression (LR) and Support Vector Machines (SVM) before progressing to complex models like Artificial Neural Networks (ANN). 

## Rationale behind the Proposed Approach
**Baseline Establishment with LR and SVM**: Influenced by the findings in "Customer churn prediction in telecommunications" by Bingquan Huang, M. T. (2012), starting with Logistic Regression and SVM helps establish a baseline performance for the churn prediction model. These algorithms are not only simpler and faster to implement but also provide valuable insights into the data's behavior and key influencing factors. Logistic Regression, with its ease of interpretation and implementation, is particularly useful for understanding the relationship between independent variables and the churn likelihood. Similarly, SVM, known for its effectiveness in higher-dimensional spaces, can handle non-linear relationships in the data more adeptly.

**Progression to Artificial Neural Networks (ANN):** The move towards ANNs is backed by the paper "Customer Churn Prediction Modelling Based on Behavioural Patterns Analysis using Deep Learning" by S. Agrawal, A. Das, A. Gaikwad, and S. Dhage (2018). The study’s emphasis on deep learning for analyzing behavioral patterns aligns with the use of ANNs, which are adept at capturing complex, non-linear relationships in large datasets. Moreover, ANNs can be tuned and scaled to improve performance as needed.

## Formal Problem Statement
The problem of customer churn in the telecom industry can be formally defined as a binary classification task. The primary goal is to develop a predictive model that can accurately forecast whether a customer is likely to leave the company within a specified timeframe. This task can be represented as follows: 

Given a dataset $\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$, where:
- $\mathbf{x}_i$ represents the feature vector for the $i$-th customer, encompassing various customer-related attributes.
- $y_i$ is the binary target variable for the $i$-th customer, with $y_i = 1$ indicating churn (customer leaving) and $y_i = 0$ indicating non-churn (customer staying).

The objective is to construct a predictive machine learning model $f: \mathbf{X} \rightarrow Y$, where $f$ is a function mapping from the space of input features $\mathbf{X}$ to the output space $Y$. That is, the model $f$ aims to predict the churn status $y_i$ based on the input features $\mathbf{x}_i$. 

The effectiveness of this model is measured based on its accuracy in classifying customers into churn and non-churn categories, thereby enabling the telecom company to implement targeted retention strategies.

## Evaluation Criteria
In this section, we thoroughly discuss the evaluation metrics selected to assess the performance of our machine learning models in predicting customer churn. Each metric has been chosen for its relevance and ability to provide insight into different aspects of model performance, crucial for a nuanced understanding of churn prediction in the telecom sector.
### Accuracy
Accuracy measures the overall effectiveness of the model in correctly predicting both churn and non-churn cases.

$\text{Accuracy} = \frac{\text{True Positives (TP)} + \text{True Negatives (TN)}}{\text{Total Predictions}}$

In telecom, high accuracy means the model is generally reliable in identifying churners and non-churners. However, due to potential class imbalance (fewer churners than non-churners), accuracy alone might not be a sufficient indicator of a good model.

### Precision
Precision in the telecom sector assesses the model's ability to correctly identify customers who will churn.

$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{False Positives (FP)}}$

High precision is vital for telecom companies as it ensures that resources aimed at retaining customers are not wasted on those unlikely to churn.

### Recall
Recall is crucial for identifying as many actual churners as possible, which is essential for telecom companies to prevent revenue loss.

$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{False Negatives (FN)}}$

In telecom churn prediction, a high recall indicates the model is effective in capturing a significant portion of customers who are at risk of leaving.

### F1 Score
The F1 Score is particularly important in the telecom sector where both identifying churners (Recall) and avoiding false alarms (Precision) are crucial.

$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

It provides a balanced metric, especially useful in scenarios where the cost of false positives and the risk of missing actual churners are both high.

### ROC Curve and AUC-ROC Score
The ROC Curve and the AUC-ROC Score are vital for telecom companies to understand the trade-offs between identifying true churners and avoiding false positives.

$\text{TPR (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$

$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$

$\text{AUC-ROC Score} = \text{Area under the ROC Curve}$

A high AUC-ROC Score indicates a model's strong ability in distinguishing between potential churners and loyal customers, which is crucial in making informed decisions for customer retention strategies.


# Experiment setup
## Workflow
Our churn prediction model workflow comprises Data Collection, Data Cleaning, Feature Engineering, and Normalization. The data cleaning step involves refining this dataset by addressing missing values, duplicates, and irrelevant information. Following this, feature engineering is crucial for identifying key customer characteristics that may influence churn. This step is vital as the right features can significantly enhance model accuracy. Finally,  the normalization step standardizes the feature values, typically scaling them between 0 and 1, ensuring uniformity across all data inputs. Detailed explanations of these processes are provided in the subsequent sections.

### Data collection
We start with data collection and consolidation of the Telco customer churn dataset from IBM. This comprehensive dataset details 7043 customers of a fictional telecom company in California and is categorized into five distinct tables, each capturing different facets of customer information:

- **Demographics:** This section includes essential demographic details like age, gender, marital status, and dependents, providing a baseline understanding of the customer profile.
- **Location:** It encompasses geographical data such as country, state, and zip code, offering insights into regional variations in customer behavior and preferences.
- **Population:** This table provides the population statistics for different zip codes, aiding in understanding the market reach and penetration of the telecom services.
- **Services:** This crucial segment covers the range of services availed by customers, including data on total charges, tenure, and referral activity, shedding light on service usage patterns and customer engagement.
- **Status:** This includes key data on customer satisfaction, churn status, churn reasons, and customer lifetime value, which are pivotal in understanding churn dynamics and customer value to the company.
    
These tables are merged to create a comprehensive dataset with unique features, encompassing the aspects mentioned above. This diverse set of features provides a holistic view of the factors that might influence customer churn.

### Data Cleaning
During the Data Cleaning phase, standard procedures were applied to enhance the dataset's quality and usability. Notably, it was observed that the "Churn Reason" and "Churn Category" columns contained missing values. This observation was expected since customers who had not churned naturally lacked reasons for doing so, and consequently, no high-level category was associated with them. To address this, a considered approach was taken by filling these missing values with either "N/A" or "Not Churned," ensuring consistent data representation. Additionally, we conducted a thorough check for duplicate rows. This step ensures that each data point is unique and contributes to a more robust dataset for subsequent analysis.


### Feature Engineering
In this section, we delve into the experiments conducted on the dataset and our approach to feature extraction. Our process commences with the creation of a comprehensive feature summary, enabling us to gain a detailed understanding of each attribute. This summary encompasses crucial information such as feature names, the count of unique values, data types, and an enumeration of all distinct values within each feature. 

Based on these insights, we draw several key observations, prompting subsequent actions for feature extraction:

- **Redundant Features:** Notably, certain features such as `CustomerID,' `Country,' and `State' exhibit redundancy. They are either unique for each observation or maintain uniformity across all observations. Additionally, the `Lat Long' feature appears to be a composite of `Latitude' and `Longitude.' These redundant features contribute minimally to our prediction task and are candidates for removal to streamline our model.

- **Binary Categorical Features:** We identify several binary features such as `Gender,' `Senior Citizen,' `Married,' `Dependents,' `Referred a Friend,' and so on. To prepare them for modeling, we opt for label encoding, converting them into numerical inputs.

- **Multi-valued Categorical Features:** Features with multiple categories, including `Offer', `Internet Type,' `Contract,' and `Payment Method,' are identified. For compatibility with modeling algorithms, we employ one-hot encoding, transforming these attributes into numerical representations.

- **High-Cardinality Categorical Features:** Certain categorical features like `City' and `Zipcode' exhibit a high cardinality, implying a substantial number of unique values. Applying one-hot encoding to such features is typically discouraged, as it would significantly expand the total number of features and thus computationally inefficient. To address this challenge, we opt for frequency encoding on these features.

- **Continuous Numerical Features:** Attributes such as `Age,' `Number of Dependents,' `Number of Referrals,' `Tenure in Months,' `Monthly Charge,' `Total Charges,' and so on are recognized as continuous numerical features. To ensure compatibility with certain algorithms like Logistic Regression, SVMs, and Neural Networks, which assume standardized features, these attributes may require scaling.

- **Target Variable Selection:** In line with the official documentation from IBM, we identify several features, including `Churn Label,' `Churn Value,' `Churn Score,' `Customer Status,' `Churn Category,' and `Churn Reason,' as directly associated with the customer churn outcome. `Churn Value' is chosen as our target variable, while the remaining features are slated for removal to prevent data leakage during modeling.

### Normalisation
In alignment with our feature extraction process, our dataset contains several continuous numerical features. Notably, certain algorithms, such as Logistic Regression and Support Vector Machines (SVMs), operate under the assumption that all features should be on the same scale to prevent one feature from dominating the others during the learning process. This can occur when features are on significantly different scales, leading to suboptimal model performance and convergence issues. To address this, we adapt to normalization of these features by scaling them to have a mean of 0 and a standard deviation of 1 (standardization). 

## Model Implementation
### Logistic Regression
Logistic Regression is a fundamental machine learning algorithm primarily employed for binary classification tasks. It is celebrated for its simplicity and efficacy in modeling the probability of a binary outcome. At the heart of Logistic Regression resides the sigmoid function, denoted as $\sigma(z)$. This function serves a pivotal role in mapping a linear combination of inputs to a probability value within the range of 0 and 1. The sigmoid function is defined as follows:

$\sigma(z) = \frac{1}{1 + e^{-z}}$
Where $z = w^Tx + b$ signifies the linear combination of the weight vector ($w$), input feature vector ($x$), and bias ($b$).

The model from sklearn library has been used. Decisions regarding the number of iterations and the learning rate ($\alpha$) are made, with these parameters typically treated as hyperparameters. The max number of iterations is set to 1000 to guarantee convergence and default learning rate is used. Once the model is trained, a prediction function is established. This function employs the sigmoid transformation on the data, converting the resulting probabilities into class labels. The choice of a threshold, often set at 0.5, dictates the class assignment.

Finally metrics such as accuracy, precision, recall, and the F1-score are commonly employed to gauge how effectively the model generalizes to new data. These metrics provide valuable insights into the model's overall performance.

### Support Vector Machines
Support Vector Machines (SVM) is a powerful machine learning algorithm commonly used for classification tasks. SVM aims to find the maximal margin hyperplane that effectively separates observations into different classes. A hyperplane in SVM is a decision surface that effectively segregates data points into different classes. The optimal hyperplane is the one that has the largest distance, or margin, to the nearest training data points of any class. This margin is a buffer zone, and maximizing it leads to better generalization of the classifier. The classification in SVMs hinges on the equation of the hyperplane, \(w \cdot x - b = 0\), where \(w\) is the weight vector determining the orientation of the hyperplane, \(x\) represents individual data points, \(b\) is the bias term, offsetting the hyperplane from the origin. Data points are classified based on the sign of the expression \(w \cdot x - b\). 

The model from sklearn library has been used. In addition to the learning rate \(\alpha\), kernel, and the number of iterations, the regularization parameter \(C\) need to be set appropriately. \(C\) plays a vital role in the SVM formulation, as it balances the margin maximization and misclassification penalty. Default parameters are used for the sake of simplicity. We later perform cross-validation to choose the best hyperparameters.

### Artificial Neural Networks
Artificial Neural Networks (ANNs) are computational models inspired by the human brain's neural networks. They are part of a broader machine learning field known as deep learning. ANNs are composed of interconnected groups of nodes or "neurons," which are organized in layers. These layers include an input layer, one or more hidden layers, and an output layer. Here's a brief overview of how ANNs function and their usage:
- Input Layer: Receives the raw data similar to the sensory input in humans.
- Hidden Layers: Perform computations through neurons that apply weights to the inputs and direct them through an activation function as the output.
- Output Layer: Produces the final outcome, which can be a continuous value, a binary value, or a probability distribution over different classes depending on the problem.

Each neuron in a network applies a non-linear transformation to the input data and uses optimization algorithms, like gradient descent, to update the weights during the training process based on the error of the output.

#### Architecture
The network is designed with a sequential architecture, where each layer's output serves as the input for the subsequent layer, creating a linear flow of data processing.  The model begins with a simple single-layer network to understand the dataset's fundamental structure and then progresses through more complex architectures. After evaluating the single-layer network, a two-layer network is constructed to observe improvements or changes in predictive capabilities. Finally, a three-layer net is implemented, further enhancing the network's ability to capture intricate patterns in the data.

**Dense Layers:** These networks consist of fully connected (dense) layers, which are fundamental in learning from the entire dataset. They are capable of identifying complex patterns by allowing every neuron in one layer to connect to every neuron in the subsequent layer, thus enabling the network to learn intricate details from the data.

**Network Weights Initialization:** Weights in the network are initialized with small random values, which is a standard practice to break symmetry and ensure that the learning process does not stall. This randomness helps the network in developing a robust set of predictions as it trains.

**Activation Functions:** Activation functions like ReLU (Rectified Linear Unit) are employed in the early layers of the network. ReLU adds non-linearity to the model, allowing it to learn more complex relationships within the data. This function is particularly favored for its simplicity and efficiency in training deep networks.

**Dropout Layers:** As a regularization strategy to combat overfitting, dropout layers are interspersed between the dense layers. Dropout randomly disables a fraction of neuron connections during training phases, which encourages the network to learn more robust features that are not reliant on any small set of neurons.

**Output Layer:** The final layer of the network uses a sigmoid activation function. This function is suitable for binary classification tasks like churn prediction, as it outputs a probability value between 0 and 1, representing the likelihood of a customer churning.

The below images describe how each of the layers are structured with respect to the input dataset

![image](https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/d8285f08-9b5c-4ddb-a8de-2d78d529e038)
![image](https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/60d1d7ac-6af1-41bf-9e8c-fd0a0847f267)
![image](https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/e79aeefe-a489-4566-a95a-e89a5ae47d5e)

# Experiment results
## Logistic Regression Results
Logistic Regression (LR) is a simple yet effective approach, especially when dealing with linearly separable data. This characteristic makes LR an excellent starting point for analyzing datasets. If LR performs well, it suggests that the relationships between the features and the target variable (customer churn) are predominantly linear. This understanding can guide further data exploration and modeling efforts. For instance, if LR had shown poor performance, it would indicate the need to explore models capable of capturing more complex, non-linear relationships. 

The LR model implemented using the sklearn library achieved an impressive accuracy of 95.45\%. However, in the context of our imbalanced dataset, this metric might be somewhat misleading, as it could be predominantly reflecting the model's ability to identify the majority class (non-churned customers). In terms of class-specific performance, it demonstrated high precision (0.96 for class 0 and 0.93 for class 1), indicating a strong ability to correctly identify actual instances of each class. The recall scores (0.98 for class 0 and 0.89 for class 1) suggest that the model was particularly effective at identifying non-churned customers, while still maintaining good performance on churned customers. The f1-scores, balancing precision and recall, were 0.97 and 0.91 for classes 0 and 1, respectively. The ROC-AUC score of 0.992 indicates an excellent capacity of the model to discriminate between churned and non-churned customers at various thresholds.

<img width="363" alt="image" src="https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/ec5cf6ea-38df-418b-82f7-aae0160a81b1">


## Support Vector Machines Results
Support Vector Machine (SVM) is renowned for its effectiveness in classification tasks, especially in datasets with clear margins of separation between classes. 

The library SVM model using the default kernel achieved an accuracy of 95.24\%, with precision scores of 0.95 for non-churned and 0.95 for churned customers. The f1-scores were 0.97 and 0.91 for non-churned and churned customers, respectively, showing a good balance between precision and recall. The ROC-AUC score of 0.9897 suggests a high ability of the model to distinguish between the two classes across different thresholds. This behavior is very close to that of the LR model. Cross-validation has been performed to choose the best kernel and the results suggest that linear kernel to be optimal choice with the best model having 95.95\% accuracy.

<img width="389" alt="image" src="https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/d0ecb850-fd39-4948-8cf9-4119420231df">


## Artificial Neural Net Results
The three-layer ANN model's high accuracy of 95.67% is indicative of its strong predictive capability, closely aligning with the accuracy observed in the simpler single-layer model. Precision scores of 0.96 for non-churned customers and 0.94 for churned customers demonstrate the model's consistent reliability in its predictions across both classes. Recall scores of 0.98 for non-churned customers and 0.89 for churned customers show the model's strength in identifying true non-churned customers, though it indicates a slightly lower, yet still impressive, ability to identify true churned customers. The F1 scores of 0.97 for the non-churned class and 0.92 for the churned class further reinforce the model's balanced performance in terms of precision and recall.

The ROC-AUC score of 0.988 confirms the model's exceptional discriminative power between churned and non-churned customers. Such a high score suggests that the model is highly effective at ranking predictions with a high degree of confidence across various thresholds.

<img width="305" alt="image" src="https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/239eda3f-e4f1-43fc-8da4-3a04b2a4fb9a">
<img width="322" alt="image" src="https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/20d5cbe4-f81c-4a3f-bcd5-08063a383354">
<img width="316" alt="image" src="https://github.com/skotla1509/CS6140-Final-Project/assets/94617217/066de3d1-00b8-42d7-aa2d-847fce6eeaf9">




# Discussion
The progression from a single-layer to a three-layer Artificial Neural Network (ANN) did not result in a substantial change in performance metrics. This observation suggests that adding complexity to the model, in this case through additional layers, did not significantly enhance the ANN's ability to predict customer churn. This could imply that the relationships within the dataset may not be excessively complex, or that the single-layer network was already sufficient to capture the relevant patterns needed for an accurate prediction. This might also steer future efforts towards other model enhancements or feature engineering rather than increasing network depth. These results and observations can provide valuable insights into the modeling approach for similar datasets and predictive tasks.


# Conclusion
This project's use of IBM's Telco customer churn dataset was pivotal in uncovering the multifaceted nature of customer churn in the telecom sector. The comprehensive range of customer attributes, from demographics to service usage, allowed for a deep understanding of the various factors that contribute to churn. The rigorous data preprocessing, including meticulous cleaning and feature engineering, ensured the integrity and relevance of the data, which was fundamental to the study's success.

The comparative analysis of Logistic Regression, Support Vector Machines, and Artificial Neural Nets provided valuable insights into the suitability and effectiveness of different models in churn prediction. Logistic Regression and SVM, effective in linearly separable scenarios, affirmed the presence of linear relationships in the dataset. 

The importance of quality data preprocessing and the impact of selecting appropriate models based on data characteristics were key takeaways. Balancing model complexity with interpretability was a crucial aspect, particularly in a business setting where explainability is essential. Additionally, the project highlighted the need for careful consideration of different evaluation metrics, especially in imbalanced datasets, to accurately assess model performance.

# References
1. Bingquan Huang, M. T. (2012, January). Customer churn prediction in telecommunications. Expert Systems with Applications, 39(1), 1414-1425. doi:https://doi.org/10.1016/j.eswa.2011.08.024
2. S. Agrawal, A. Das, A. Gaikwad and S. Dhage, "Customer Churn Prediction Modelling Based on Behavioural Patterns Analysis using Deep Learning," 2018 International Conference on Smart Computing and Electronic Enterprise (ICSCEE), Shah Alam, Malaysia, 2018, pp. 1-6, doi: 10.1109/ICSCEE.2018.8538420
3. Coussement, K., & Poel, D. V. (2008). Churn prediction in subscription services: An application of support vector machines while comparing two parameter-selection techniques. Expert Systems with Applications, 34(1), 313-327. doi:https://doi.org/10.1016/j.eswa.2006.09.038
4. IBM. (2019, July 11). Telco customer churn (11.1.3+). Retrieved from https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
