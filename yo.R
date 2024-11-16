library(forcats)
library(caret)
library(dplyr)
library(metrica)
library(rpart.plot)         #    for    plotting    decision    trees
library(vip)                                           #    for    feature    importance
library(pdp)
library(tensorflow)
install_tensorflow()
library(readr)
df <- read_csv("bluechip-summit-credit-worthiness-prediction/Train.csv")

df$Dependents <- df$Dependents %>%
  fct_recode(
    "3" = "3+")

df$Gender <- factor(train$Gender)  
df$Married <- factor(df$Married)
df$Education <- factor(df$Education)
df$Self_Employed <- factor(df$Self_Employed)
df$Credit_History <- factor(df$Credit_History)
df$Property_Area <- factor(df$Property_Area)


sum(is.na(df))

# Let's split our data, 80% for the training and 20% for testing
split_size	=	0.8
sample_size	=	floor(split_size	*	nrow(df)) 
set.seed(123)
train_indices	<-	sample(seq_len(nrow(df)),	size	=	
                          sample_size)
train	<-	df[train_indices,	] 
test	<-	df[-train_indices,	]

train	<-	train[,c(3:15)]
test	<-	test[,c(3:15)	]

colnames(train)

train$Loan_Status <- factor(train$Loan_Status)
test$Loan_Status <- factor(test$Loan_Status)



X =train[,-12]
y= train[,12]
######## Test
X_test=test[,-12]
y_test= as.factor(as.matrix(test[,12]))
##############################


mthod=c("rpart","knn","cforest","svmLinear","svmRadial","naive_bayes",
        "nnet","xgbTree","glm")

################################################## DecisionTree
#Decision Tree Classification

## Controle SVM
ctrl1    <-    trainControl(
  method    =    "cv", 
  number    =    10
)
library(kernlab)
sigma_est <- sigest(Loan_Status~. , data=train)

svmgrid <- expand.grid(sigma=seq(0.03037965,0.31450595 , by = 0.01),
                       C=2^(-2:2))

ctrl2    <-    trainControl(
  method    =    "repeatedcv", 
  number    =    10
)
# ###########  SVM 1

set.seed(5628)         #    for    reproducibility 
classifierSvm1 =  train(Loan_Status~. , data=train , 
  method    =    "svmRadial",
  preProcess    =    c("center",    "scale"),
  metric    =    "Accuracy",     
  trControl    =    ctrl1,
  tuneLength    =    10 ,
  tuneGrid= svmgrid
)
ggplot(classifierSvm1)
y_predsvm1 =factor(predict(classifierSvm1, newdata = X_test), levels = c(0,1))

cm=confusionMatrix(data = y_predsvm1, reference = y_test)
cm$overall[1]

### SVM 2

classifierSvm2 =  train(Loan_Status~. , data=train , 
                        method    =    "svmRadial",
                        preProcess    =    c("center",    "scale"),
                        metric    =    "Accuracy", 
                        maximize= T,
                        trControl    =    ctrl2,
                        tuneLength    =    10,
                        tuneGrid= svmgrid 
)
ggplot(classifierSvm2)
y_predsvm2 =factor(predict(classifierSvm2, newdata = X_test), levels = c(0,1))

cm1=confusionMatrix(data = y_predsvm2, reference = y_test)
cm1$overall[1]


#################### Neural


# Installer et charger le package keras si nécessaire
if(!require(keras)) {
  install.packages("keras")
  library(keras)
}

# # Génération des données synthétiques
# set.seed(123)
# n <- 1000

df <- read_csv("bluechip-summit-credit-worthiness-prediction/Train.csv")

df$Dependents <- df$Dependents %>%
  fct_recode(
    "3" = "3+")

df$Gender <- factor(train$Gender)  
df$Married <- factor(df$Married)
df$Education <- factor(df$Education)
df$Self_Employed <- factor(df$Self_Employed)
df$Credit_History <- factor(df$Credit_History)
df$Property_Area <- factor(df$Property_Area)


# Séparation des variables explicatives (X) et de la cible (y)
X <- as.matrix(df%>% select(-Loan_Status))
y <- as.numeric(df$Loan_Status)

# Division en données d'entraînement et de test
set.seed(123)
n <- dim(df)[1]
train_index <- sample(1:n, size = 0.8 * n)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Construction du modèle
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")  # Sigmoid pour la classification binaire

# Compilation du modèle
model %>% compile(
  optimizer = optimizer_adam(lr = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Entraînement du modèle
history <- model %>% fit(
  X_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

# Évaluation sur les données de test
score <- model %>% evaluate(X_test, y_test)
cat("Test Loss:", score$loss, "\n")
cat("Test Accuracy:", score$accuracy, "\n")




# Fitting Decision TreeClassification to the Training set
library(rpart)

acc=c()

for (i in 1:length(mthod)) {}
  
### Svm

classifier = train(Loan_Status~. , data=train ,
                   method    =    "svmRadial" , metric="Accuracy",
                   trControl = cv)


# Predicting the Test set results
y_pred =factor(predict(classifier, newdata = X_test), levels = c(0,1)) 

ggplot(classifier)
# Making the Confusion Matrix
cm=confusionMatrix(data = y_pred, reference = y_test)
cm$overall[1]

###
classifier1 = train(Loan_Status~. , data=train ,
                   method    =    "nnet"  , metric="Accuracy",
                   trControl = cv)


ggplot(aes(x=mthod, y= acc))+geom_bar()

# Visualising the Training set results #no overfitting bcoz of rpart lib.(better than python results-had overfitting)
library(ElemStatLearn)
set = train
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3], main = 'Decision Tree (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#plotting the decision tree # to run this code,dont use feature scaling step
plot(classifier)
text(classifier)
library(rpart.plot) 
fancyRpartPlot(classifier)





###########################################
#    training    a    k-nearest    neighbors    model
knn_model    <-    knn(train_data,    test_data,    train_labels,    k=1) 
#    creating    a    confusion    matrix
cm    <-    table(test_labels,    knn_model) 
print("Confusion    matrix")
print(cm)170                Text Mining  with  Machine  Learning:  Principles  and Techniques
#    number    of    instances 
n    <-    sum(cm)
#    number    of    correctly    classified    instances    for    each    class 
correct    <-    diag(cm)
#    numbers    of    instances    in    each    class 
instances_in_classes    <-    apply(cm,    1,    sum)
#    numbers    of    each    class    predictions 
class_predictions    <-    apply(cm,    2,    sum)
#    accuracy
accuracy    <-    sum(correct)/n 
#    precision    per    class
precision    <-    correct/class_predictions 
#    recall    per    class
recall    <-    correct/instances_in_classes 
#    F1-measure    per    class
f1    <-    2    *    precision    *    recall    /    (precision    +    recall) 
#    printing    summary    information    for    all    classes
df    <-    data.frame(precision,    recall,    f1)
print("Detailed    classification    metrics")
print(df)
print(paste("Accuracy:",    accuracy))
#    macroaveraging
print("Macro-averaged    metrics") 
print(colMeans(df))
#    microaveraging
print("Micro-averaged    metrics")
print(apply(df,    2,    function    (x)
  weighted.mean(x,    w=instances_in_classes)))

#####################

table(df$Loan_Status)
table(test$Loan_Status)
4913 /(4913 +985)
984/(984+196)
#######################

# Charger les packages nécessaires
library(caret)

# Exemple de données - utiliser iris pour la classification
data(iris)

# Configurer la validation croisée
train_control <- trainControl(
  method = "cv",         # Validation croisée
  number = 5,            # Nombre de plis (folds)
  summaryFunction = defaultSummary  # Fonction de résumé pour l'accuracy
)

# Définir les algorithmes et leurs paramètres
models <- list(
  rf = caretModelSpec(method = "rf", tuneGrid = expand.grid(mtry = c(1, 2, 3))),
  svm = caretModelSpec(method = "svmLinear", tuneGrid = expand.grid(C = c(0.1, 1, 10))),
  knn = caretModelSpec(method = "knn", tuneGrid = expand.grid(k = c(3, 5, 7)))
)

# Initialiser un data frame pour stocker les accuracies
results <- data.frame(Model = character(), Accuracy = numeric(), stringsAsFactors = FALSE)

# Boucle pour entraîner chaque modèle et enregistrer les résultats
for (model_name in names(models)) {
  set.seed(123)  # Fixer le générateur pour la reproductibilité
  
  # Entraîner le modèle
  model <- train(
    Species ~ .,           # Formule (dépendante ~ indépendantes)
    data = iris,           # Données d'entraînement
    method = models[[model_name]]$method,  # Méthode de modélisation
    tuneGrid = models[[model_name]]$tuneGrid,  # Grid des hyperparamètres
    trControl = train_control  # Configuration de la validation croisée
  )
  
  # Récupérer l'accuracy moyenne du meilleur modèle
  best_accuracy <- max(model$results$Accuracy)
  
  # Enregistrer les résultats
  results <- rbind(results, data.frame(Model = model_name, Accuracy = best_accuracy))
}

# Afficher les résultats
print(results)
