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

df$Gender <- factor(df$Gender)  
df$Married <- factor(df$Married)
df$Education <- factor(df$Education)
df$Self_Employed <- factor(df$Self_Employed)
df$Credit_History <- factor(df$Credit_History)
df$Property_Area <- factor(df$Property_Area)
df$Dependents <- factor(df$Dependents)
df$Loan_Status <- factor(df$Loan_Status )
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
# Assigner les poids pour chaque observation en fonction de sa classe





########

# Diviser les données en entraînement et test
set.seed(123)
train_index <- createDataPartition(data$Loan_Status, p = 0.8, list = FALSE)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Convertir les niveaux en noms valides en utilisant make.names()
levels(train_data$Loan_Status) <- make.names(levels(train_data$Loan_Status))


weights_per_observation <- ifelse(train_data$Loan_Status == "1", 
                                  weights["1"], 
                                  weights["0"])

# Afficher les poids
print(weights_per_observation)
# Configurer le cluster pour utiliser 3 cœurs
cluster <- makeCluster(3)
registerDoParallel(cluster)


# Configurer trainControl avec parallélisation
train_control <- trainControl(
  method = "repeatedcv",    # Validation croisée répétée
  number = 5,               # 5 plis
  repeats = 3,              # Répétée 3 fois
  classProbs = TRUE,        # Calculer les probabilités pour les métriques comme ROC
  summaryFunction = twoClassSummary,  # Optimiser sur AUC
  allowParallel = TRUE      # Activer la parallélisation
)


# Configuration de la grille d'hyperparamètres pour xgboost
tune_grid <- expand.grid(
  nrounds = c(50, 100),            # Nombre d'itérations
  max_depth = c(3, 6),             # Profondeur maximale
  eta = c(0.01, 0.1),              # Taux d'apprentissage
  gamma = c(0, 1),                 # Régularisation
  colsample_bytree = c(0.8, 1),    # Sous-échantillonnage des colonnes
  min_child_weight = c(1, 3),      # Poids minimum pour un nœud
  subsample = c(0.8, 1)            # Sous-échantillonnage des observations
)

# Entraînement du modèle avec caret
xgb_model <- train(
  Loan_Status ~ ., 
  data = train_data,weights = weights_per_observation,
  method = "xgbTree",              # Utiliser l'intégration xgboost dans caret
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "Accuracy"                   # Optimisation basée sur l'AUC
)


# Stopper le cluster
stopCluster(cluster)
registerDoSEQ()

predictions <- predict(xgb_model, newdata = test_data)
levels(predictions)
# Créez un vecteur de correspondance entre les niveaux "X1", "X0" et "1", "0"
pred_levels <- c("X0" = "0", "X1" = "1")

# Modifier les prédictions pour les rendre compatibles avec les niveaux d'origine
predictions <- recode(predictions, !!!pred_levels)

# Vérifier les nouvelles valeurs de prédiction
print(predictions)


confusionMatrix(predictions, test_data$Loan_Status)
