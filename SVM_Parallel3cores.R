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

# Configurer le cluster pour utiliser 3 cœurs
cluster <- makeCluster(3)  # Utiliser 3 cœurs
registerDoParallel(cluster)



# Configurer trainControl
train_control <- trainControl(
  method = "repeatedcv",  # Validation croisée répétée
  number = 10,             # 5 plis
  repeats = 3,            # 3 répétitions
  allowParallel = TRUE,   # Activer la parallélisation
  search = "grid"         # Recherche systématique
)

# Définir une grille de recherche optimale
tune_grid <- expand.grid(
  sigma = c(0.005, 0.1),  # Grille pour sigma
  C = c(0.5,1,) )              # Grille pour C


# Entraîner le modèle SVM radial
svm_model2 <- train(
  Loan_Status ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = tune_grid,
  weights = weights[train_data$Loan_Status]
)

# Arrêter le cluster après entraînement
stopCluster(cluster)
registerDoSEQ()


predictions <- predict(svm_model2, newdata = test_data)

confusionMatrix(predictions, test_data$Loan_Status)

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


predictions_test <- predict(svm_model2, newdata = data1 )

submission <- data.frame(ID = df1$ID, Loan_Status = predictions_test)
colnames(submission) <- c("ID", "Loan_Status")  # Adapter aux exigences

write.csv(submission, "submission2.csv", row.names = FALSE)

