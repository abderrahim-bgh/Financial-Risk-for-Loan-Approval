######### AOT8130 - Analyse prédictive ~ Travail de session #########
############ Financial Risk for Loan Approval équipe 02 ############
#
#
# Installer les packages nécessaires (à exécuter une seule fois)
packages <- c("dplyr", "ggplot2", "gplots", "skimr", "viridis", "caret")
installed_packages <- packages %in% installed.packages()[, "Package"]
if (any(!installed_packages)) {
  install.packages(packages[!installed_packages])
}
install.packages("caret")

# Charger les packages
library(dplyr)
library(ggplot2)
library(gplots)
library(skimr)
library(viridis)
library(caret)

# Charger les données à partir d'un fichier CSV
loan_data <- read.csv("Loan.csv")

# Vérification de la qualité des données
dim(loan_data)                          # Dimensions des données
names(loan_data)                        # Noms des colonnes
str(loan_data)                          # Structure des D
skim(loan_data)                         # Résumé complet des données

# Compter les valeurs manquantes par colonne
na_counts <- setNames(as.data.frame(colSums(is.na(loan_data))), "Count_of_NA")
print(na_counts)

# Compter les lignes dupliquées
cat("Nombre de lignes dupliquées :", sum(duplicated(loan_data)), "\n")

# Définir les colonnes catégorielles et numériques
categorical_cols <- c(
  "ApplicationDate", "EmploymentStatus", "EducationLevel",
  "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose",
  "BankruptcyHistory", "PreviousLoanDefaults", "LoanApproved"
)
numerical_cols <- setdiff(names(loan_data), categorical_cols)

# Nettoyage des données :
# Copier le dataset pour le nettoyage
loan_data_clean <- loan_data

# Remplacer les valeurs 0 dans la colonne LoanAmount par le mode
loan_amount_mode <- as.numeric(names(sort(table(loan_data_clean$LoanAmount), decreasing = TRUE))[1])
loan_data_clean$LoanAmount[loan_data_clean$LoanAmount == 0] <- loan_amount_mode

# Supprimer les lignes avec des valeurs manquantes
loan_data_clean <- na.omit(loan_data_clean)

# Transformer ApplicationDate en date
loan_data_clean$ApplicationDate <- as.Date(loan_data_clean$ApplicationDate, format = "%Y-%m-%d")

# Transformer la variable cible LoanApproved en facteur
loan_data_clean$LoanApproved <- as.factor(loan_data_clean$LoanApproved)

# Visualisation de la distribution de LoanApproved
ggplot(data = loan_data_clean, aes(x = LoanApproved)) +
  geom_bar(aes(fill = after_stat(count)), color = "black") +
  scale_fill_viridis() +
  labs(
    title = "Distribution de la variable cible : LoanApproved",
    x = "LoanApproved (0: Refusé, 1: Approuvé)",
    y = "Nombre d'observations"
  )

# Sélection d'un échantillon aléatoire de 5000 lignes si nécessaire
set.seed(123)
loan_data_sample <- loan_data_clean[sample(nrow(loan_data_clean), min(5000, nrow(loan_data_clean))), ]

# Sélectionner uniquement les colonnes numériques pour la matrice de corrélation
variables_numeriques <- loan_data_sample[, sapply(loan_data_sample, is.numeric)]

# Calcul de la matrice de corrélation
correlation_matrix <- cor(variables_numeriques, use = "complete.obs")

# Heatmap de la matrice de corrélation
heatmap(
  correlation_matrix,
  col = colorRampPalette(c("blue", "white", "tomato"))(50),
  scale = "none",
  main = "Matrice de Corrélation",
  labRow = names(variables_numeriques),
  labCol = names(variables_numeriques),
  cexCol = 0.7,
  cexRow = 0.7
)

# Ajout d'une légende
legend(
  "topright",
  legend = c("Corrélation Négative", "Corrélation Nulle", "Corrélation Positive"),
  fill = colorRampPalette(c("blue", "white", "tomato"))(3),
  inset = c(-0.05, -0.1),
  xpd = TRUE,
  horiz = FALSE,
  bty = "n",
  cex = 0.8
)

# Représentation des données en clusters
install.packages("factoextra")
library(factoextra)
set.seed(123)  # Graine pour la reproductibilité

# Sélection des variables numériques pour le clustering
variables <- loan_data_clean[, sapply(loan_data_clean, is.numeric)]

# Standardisation des données
scaled_data <- scale(variables)

# Appliquer l'algorithme K-means avec k = 2 clusters
res.km <- kmeans(scaled_data, centers = 2, nstart = 25)

# Appliquer l'ACP pour réduire les dimensions
res.pca <- prcomp(variables, scale = TRUE)

# Obtenir les coordonnées des individus sur les deux premières dimensions
ind.coord <- as.data.frame(res.pca$x[, 1:2])
colnames(ind.coord) <- c("Dim1", "Dim2")
ind.coord$cluster <- factor(res.km$cluster)  # Ajouter les clusters

# Calculer la variance expliquée
eigenvalue <- summary(res.pca)$importance[2, 1:2] * 100
variance.percent <- round(eigenvalue, 1)

# Visualisation des clusters
fviz_cluster(
  res.km,
  data = scaled_data,
  palette = c("#1F77B4", "#FF7F0E"),
  geom = "point",
  ellipse.type = "convex",
  ggtheme = theme_minimal(),
  pointshape = 19,
  pointsize = 3
) +
  labs(
    title = "Visualisation des clusters (K = 2)",
    subtitle = paste("Variance expliquée: Dim1 =", variance.percent[1], "%, Dim2 =", variance.percent[2], "%"),
    x = paste("Dimension 1 (", variance.percent[1], "% de la variance)", sep = ""),
    y = paste("Dimension 2 (", variance.percent[2], "% de la variance)", sep = "")
  )

# Modélisation : Régression logistique
set.seed(123)
train_index <- createDataPartition(loan_data_clean$LoanApproved, p = 0.7, list = FALSE)
train_data <- loan_data_clean[train_index, ]
test_data <- loan_data_clean[-train_index, ]

# Mise à l'échelle des variables numériques
numerical_cols <- names(train_data)[sapply(train_data, is.numeric)]
train_data[numerical_cols] <- scale(train_data[numerical_cols])
test_data[numerical_cols] <- scale(test_data[numerical_cols])

# Modèle de régression logistique
model_regression <- glm(LoanApproved ~ ., data = train_data, family = "binomial")

# Prédictions sur les données de test
predictions_regression <- predict(model_regression, newdata = test_data, type = "response")
predicted_classes_regression <- ifelse(predictions_regression > 0.5, 1, 0)

# Évaluation avec la matrice de confusion
confusion_matrix_regression <- confusionMatrix(
  factor(predicted_classes_regression), 
  test_data$LoanApproved
)
cat("\nMatrice de Confusion (Régression Logistique) :\n")
print(confusion_matrix_regression)

###################
######## Forêt Aléatoire ########
set.seed(123)

# Installer et charger le package nécessaire
if (!require(randomForest)) {
  install.packages("randomForest")
}
library(randomForest)

# Création d'un modèle de forêt aléatoire
model.rf <- randomForest(
  LoanApproved ~ .,             # Formule avec toutes les variables prédictives
  data = train_data,            # Données d'entraînement
  ntree = 100,                  # Nombre d'arbres
  mtry = sqrt(ncol(train_data) - 1), # Nombre de variables sélectionnées aléatoirement à chaque nœud
  importance = TRUE             # Calcul de l'importance des variables
)

# Afficher les informations sur le modèle
print(model.rf)

# Importance des variables Forêt Aléatoire
importance_rf <- importance(model.rf)
varImpPlot(model.rf, main = "Forêt Aléatoiret")

# Prédictions sur les données de test
predictions_rf <- predict(model.rf, newdata = test_data)

# matrice de confusion
confusion_matrix_rf <- confusionMatrix(predictions_rf, test_data$LoanApproved)
cat("\n matrice de Confusion (Forêt Aléatoire) :\n")
print(confusion_matrix_rf)

#######################
## # Comparaison entre Régression Logistique et Forêt Aléatoire ##
###################
library(pROC)
library(caret)
# calculer les métriques
m <- function(conf_matrix, auc_value) {
  accuracy <- conf_matrix$overall["Accuracy"]
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  f1_score <- conf_matrix$byClass["F1"]
  return(c(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, F1_Score = f1_score, AUC = auc_value))
}

# Prédictions pour Régression Logistique
predicted.probs.logistic <- predict(model_regression, newdata = test_data, type = "response")
predicted.classes.logistic <- factor(ifelse(predicted.probs.logistic > 0.5, 1, 0))
conf.matrix.logistic <- confusionMatrix(predicted.classes.logistic, test_data$LoanApproved)
update.packages()

# Calculer de l'AUC pour la Régression Logistique
roc_logistic <- roc(as.numeric(test_data$LoanApproved) - 1, predicted.probs.logistic)
auc_logistic <- auc(roc_logistic)

# Prédictions pour Forêt Aléatoire
predicted_classes_rf <- predict(model.rf, newdata = test_data)
predicted_probs_rf <- predict(model.rf, newdata = test_data, type = "prob")[, 2]
conf_matrix_rf <- confusionMatrix(predicted_classes_rf, test_data$LoanApproved)

# Calculer de l'AUC pour la Forêt Aléatoire
roc_rf <- roc(as.numeric(test_data$LoanApproved) - 1, predicted_probs_rf)
auc_rf <- auc(roc_rf)

# Calculer des métriques pour les 2 algo
metrics_logistic <- m(conf.matrix.logistic, auc_logistic)
metrics_rf <- m(conf_matrix_rf, auc_rf)

# les résultats
comparison <- data.frame(
  Algo = c(" Regression Logistique", "Forêt Aléatoire"),
  Accuracy = c(metrics_logistic["Accuracy"], metrics_rf["Accuracy"]),
  Sensitivity = c(metrics_logistic["Sensitivity"], metrics_rf["Sensitivity"]),
  Specificity = c(metrics_logistic["Specificity"], metrics_rf["Specificity"]),
  F1_Score = c(metrics_logistic["F1_Score"], metrics_rf["F1_Score"]),
  AUC = c(metrics_logistic["AUC"], metrics_rf["AUC"])
)

# Afficher les résultats
print("Comparaison  :")
print(comparison)

# Visualisation 
plot(roc_logistic, col = "blue", lwd = 2, main = "Courbes ROC des algorithmes")
plot(roc_rf, col = "red", lwd = 2, add = TRUE)
legend(
  "bottomright",
  legend = c("Régression Logistique", "Forêt Aléatoire"),
  col = c("blue", "red"),
  lwd = 2
)

# Résultats sous forme de tableau
library(knitr)
kable(comparison, caption = "Comparaison des métriques entre Régression Logistique et Forêt Aléatoire")





