library(caret)
library(xgboost)
library(doParallel)
library(forcats)
library(caret)
library(dplyr)
library(metrica)
library(rpart.plot)         #    for    plotting    decision    trees
library(vip)                                           #    for    feature    importance
library(pdp)
library(parallel)
library(doParallel)
library(caret)
library(readr)
df <- read_csv("bluechip-summit-credit-worthiness-prediction/Train.csv")

df$Dependents <- df$Dependents %>%
  fct_recode(
    "3" = "3+")

df$Gender <- as.numeric(df$Gender)  
df$Married <- as.numeric(df$Married)
df$Education <- as.numeric(df$Education)
df$Self_Employed <- as.numeric(df$Self_Employed)
df$Credit_History <- as.numeric(df$Credit_History)
df$Property_Area <- factor(df$Property_Area)
df$Dependents <- factor(df$Dependents)
df$Loan_Status <- as.numeric(df$Loan_Status )
str(df)

######### One hot encoded pour les variables quali

one_hot_encoded1 <- model.matrix(~ df$Property_Area- 1, df)
one_hot_encoded2 <- model.matrix(~ df$Dependents - 1, df)
# one_hot_encoded3 <- model.matrix(~ df$Loan_ID - 1, df)

colnames(one_hot_encoded2) <- c("Dependents0", "Dependents1", 
                                "Dependents2",
                                "Dependents3")

colnames(one_hot_encoded1) <-c("Property_Area0", "Property_Area1", 
                               "Property_Area2")

# name <- unique(df$Loan_ID)
# colnames(one_hot_encoded3) <- sort(name)

data <- cbind(df[,c(-2,-5,-13)],one_hot_encoded1, one_hot_encoded2)

data <- data[,-1]
#################
# Calcul des proportions de chaque classe
class_proportions <- prop.table(table(data$Loan_Status))

# Calcul des poids inverses des proportions pour donner plus de poids à la classe minoritaire
# On inverse les proportions pour que la classe minoritaire ait un poids plus élevé
weights <- 1 / class_proportions
weights <- weights / sum(weights)  # Normalisation pour que la somme des poids soit égale à 1

# Afficher les poids
print(weights[train_data$Loan_Status])


str(data)
########

# Diviser les données en entraînement et test
set.seed(123)
train_index <- createDataPartition(data$Loan_Status, p = 0.8, list = FALSE)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Configurer le cluster pour utiliser 3 cœurs
cluster <- makeCluster(3)
registerDoParallel(cluster)

# Préparation des poids
class_weights <- table(train_data$Loan_Status)
weights <- ifelse(train_data$Loan_Status == "1", 
                  class_weights["0"] / class_weights["1"], 1)

# Conversion des données en format xgboost
xgb_train <- xgb.DMatrix(
  data = as.matrix(train_data[, -which(names(train_data) == "Loan_Status")]),
  label = as.numeric(train_data$Loan_Status == "1"),
  weight = weights[train_data$Loan_Status]
)

# Configuration des paramètres pour xgboost
params <- list(
  objective = "binary:logistic",   # Log-loss pour classification binaire
  eval_metric = "auc",            # AUC comme métrique
  eta = 0.1,                      # Taux d'apprentissage
  max_depth = 6,                  # Profondeur maximale des arbres
  subsample = 0.8,                # Sous-échantillonnage des données
  colsample_bytree = 0.8          # Sous-échantillonnage des colonnes
)

# Entraînement du modèle
xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 100,                 # Nombre d'itérations
  verbose = 1,
  nthread = 3                    # Parallélisation
)

# Stopper le cluster
stopCluster(cluster)
registerDoSEQ()

# Prédictions
xgb_test <- xgb.DMatrix(as.matrix(test_data[, -which(names(test_data) == "Loan_Status")]))
predictions <- predict(xgb_model, xgb_test)

# Seuil de classification
predictions_class <- factor(ifelse(predictions > 0.5, "1", "0"))

# Évaluer les performances
confusionMatrix(predictions_class, factor(test_data$Loan_Status))
