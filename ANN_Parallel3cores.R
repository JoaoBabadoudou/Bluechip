# lET'S IMPORT THE DATA
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
library(neuralnet)
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

# Afficher les poids
print(weights)



########

# Diviser les données en entraînement et test
set.seed(123)
train_index <- createDataPartition(data$Loan_Status, p = 0.8, list = FALSE)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# # Convertir les niveaux en noms valides en utilisant make.names()
levels(train_data$Loan_Status) <- make.names(levels(train_data$Loan_Status))


# Configurer le cluster pour utiliser 3 cœurs
cluster <- makeCluster(3)  # Utiliser 3 cœurs
registerDoParallel(cluster)

# Convertir toutes les colonnes de vos données en numériques si ce n'est pas déjà le cas
train_data[] <- lapply(train_data, function(x) as.numeric(as.character(x)))

test_data[] <- lapply(test_data, function(x) as.numeric(as.character(x)))


# Vérifiez si des valeurs manquantes sont présentes
sum(is.na(train_data))  # Compte les valeurs manquantes

# Définir une formule de modèle
formula <- Loan_Status ~ .  # Votre variable dépendante

# Normaliser les données pour éviter des poids excessifs
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

train_data_normalized <- as.data.frame(lapply(train_data, normalize))

# Normaliser les données de test avec les mêmes min/max que les données d'entraînement
normalize_with_train <- function(x, min_train, max_train) {
  return((x - min_train) / (max_train - min_train))
}

# Normalisation des données de test en utilisant les min/max des données d'entraînement
test_data_normalized <- as.data.frame(mapply(
  normalize_with_train,
  x = test_data,
  min_train = sapply(train_data, min),
  max_train = sapply(train_data, max)
))

# S'assurer que les colonnes correspondent exactement
test_data_normalized <- test_data_normalized[names(train_data_normalized)]


# Définir une architecture adaptée
hidden_layers <- c(8, 6, 4)  # Trois couches avec des tailles décroissantes pour optimiser les calculs

# Entraîner le modèle
nn_model <- neuralnet(
  formula,                    # Formule avec la variable cible et les prédicteurs
  data = train_data_normalized,  # Données normalisées
  hidden = hidden_layers,        # Architecture des couches cachées
  linear.output = FALSE,         # Classification
  threshold = 0.01,              # Critère d'arrêt
  stepmax = 1e5,                 # Limiter le nombre d'itérations pour éviter une surcharge
  lifesign = "minimal"           # Afficher des mises à jour minimales pendant l'entraînement
)
# Résumé du modèle
summary(nn_model)


# Arrêter le cluster après entraînement
stopCluster(cluster)
registerDoSEQ()


predictions <- predict(nn_model, newdata =test_data_normalized)

str(test_data_normalized)
str(train_data_normalized)

predictions <-  factor( ifelse(predictions > 0.5, 1, 0))

confusionMatrix(predictions, factor(test_data$Loan_Status))

##############  Prediction  ##########################

df1 <- read_csv("bluechip-summit-credit-worthiness-prediction/Test.csv")

df1$Dependents <- df1$Dependents %>%
  fct_recode(
    "3" = "3+")

df1$Gender <- factor(df1$Gender)  
df1$Married <- factor(df1$Married)
df1$Education <- factor(df1$Education)
df1$Self_Employed <- factor(df1$Self_Employed)
df1$Credit_History <- factor(df1$Credit_History)
df1$Property_Area <- factor(df1$Property_Area)
df1$Dependents <- factor(df1$Dependents)

######### One hot encoded pour les variables quali

one_hot_encoded11 <- model.matrix(~ df1$Property_Area- 1, df1)
one_hot_encoded21 <- model.matrix(~ df1$Dependents - 1, df1)


colnames(one_hot_encoded21) <- c("Dependents0", "Dependents1", 
                                 "Dependents2",
                                 "Dependents3")

colnames(one_hot_encoded11) <-c("Property_Area0", "Property_Area1", 
                                "Property_Area2")


data1 <- cbind(df1[,c(-2,-5,-13)],one_hot_encoded11, one_hot_encoded21)

data1 <- data1[,-1]

data1[] <- lapply(data1, function(x) as.numeric(as.character(x)))

predictions_test <- predict(nn_model , newdata = data1 )
predictions_test <-  factor( ifelse(predictions_test> 0.5, 1, 0))
submission <- data.frame(ID = df1$ID, Loan_Status = predictions_test)
colnames(submission) <- c("ID", "Loan_Status")  # Adapter aux exigences

write.csv(submission, "submission5.csv", row.names = FALSE)

