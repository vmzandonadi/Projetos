setwd("C:/Users/vmzan/OneDrive/BusinessIntelligence/Projetos e Análises/01. Credit Card")
getwd()

# Instalação dos pacotes
install.packages('readr')
install.packages('dplyr')
install.packages('caret')
install.packages('ggplot2')
install.packages('corrplot')
install.packages('caTools')
install.packages('DMwR')
install.packages('C50')
install.packages('randomForest')
install.packages('Amelia')
install.packages('e1071')

# Todos os pacotes que serão utilizados
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(caTools)
library(DMwR)
library(C50)
library(randomForest)
library(Amelia)
library(e1071)

# Carregamento do arquivo
df <- read_csv("credit-card.csv", col_names = T)
View(df)
glimpse(df)

## Variável Target
## 0: Adimplente
## 1: Inadimplente

# Alteração do nome da variável target
colnames(df)[25] <- "Credit_Status"
df$ID <- NULL

# Conversão das variáveis categóricas
df$SEX <- cut(df$SEX, breaks = 2, labels = c("Male", "Female"))
df$EDUCATION <- cut(df$EDUCATION, breaks = 4, labels = c("Graduate School", "University", "High School", "Others"))
df$MARRIAGE <- cut(df$MARRIAGE, breaks = 3, labels = c("Married", "Single", "Others"))
df$PAY_0 <- as.factor(df$PAY_0)
df$PAY_2 <- as.factor(df$PAY_2)
df$PAY_3 <- as.factor(df$PAY_3)
df$PAY_4 <- as.factor(df$PAY_4)
df$PAY_5 <- as.factor(df$PAY_5)
df$PAY_6 <- as.factor(df$PAY_6)
df$Credit_Status <- as.factor(df$Credit_Status)

# Função para agrupamento da variável idade ###################
group_age <- function(AGE){
  if (AGE >= 0 & AGE <= 20){
    return('0-20 anos')
  }else if(AGE > 20 & AGE <= 40){
    return('20-40 anos')
  }else if (AGE > 40 & AGE <= 60){
    return('40-60 anos')
  }else if (AGE > 60 & AGE <=80){
    return('60-80 anos')
  }else if (AGE > 80){
    return('> 80 anos')
  }
}

# Aplicando função e criando nova variável
df$AGE_GROUP <- sapply(df$AGE, group_age)
df$AGE_GROUP <- as.factor(df$AGE_GROUP)

# Verificação de dados missing
sapply(df, function(x){
  sum(is.na(x))
})

missmap(df, 
        main = "Mapeamento de Dados Missing", 
        col = c("blue", "white"), 
        legend = FALSE)

# Verificando proporções do dataset
round(prop.table(table(df$SEX)), 2)
round(prop.table(table(df$EDUCATION)), 2)
round(prop.table(table(df$MARRIAGE)), 2)
round(prop.table(table(df$Credit_Status)), 2)

# Visualizando proporção da variável target de maneira gráfica
qplot(Credit_Status, data = df, geom = "bar") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Função para agrupamento da variável idade
group_age <- function(AGE){
  if (AGE >= 0 & AGE <= 20){
    return('0-20 anos')
  }else if(AGE > 20 & AGE <= 40){
    return('20-40 anos')
  }else if (AGE > 40 & AGE <= 60){
    return('40-60 anos')
  }else if (AGE > 60 & AGE <=80){
    return('60-80 anos')
  }else if (AGE > 80){
    return('> 80 anos')
  }
}

# Aplicando função e criando nova variável
df$AGE_GROUP <- sapply(df$AGE, group_age)
df$AGE_GROUP <- as.factor(df$AGE_GROUP)

# Análise das variáveis mais relevantes para o modelo
var_importance <- randomForest(Credit_Status ~ .
                               - AGE, 
                               data = df, 
                               ntree = 100,
                               nodesize = 10,
                               importance = T)

varImpPlot(var_importance)

# Split Data
split <- sample.split(df$Credit_Status, SplitRatio = 0.7)
train_data <- subset(df, split == T)
test_data <- subset(df, split == F)

# SMOTE
# Proporção antes do balanceamento
round(prop.table(table(test_data$Credit_Status)), 2)

# Balanceamento
train_data_smoted <- SMOTE(Credit_Status ~ ., as.data.frame(train_data), perc.over = 200)
View(train_data_smoted)

# Proporção após balanceamento
round(prop.table(table(train_data_smoted$Credit_Status)), 2)

# Criando o Modelo
rf_model <- randomForest(Credit_Status ~ .
                                    - SEX
                                    - EDUCATION
                                    - MARRIAGE
                                    - AGE
                                    - AGE_GROUP, 
                                    data = train_data_smoted, 
                                    ntree = 100, 
                                    nodesize = 10)
print(rf_model)

# Visualizando taxa de erro do modelo
plot(rf_model, ylim = c(0,0.35))
legend("topright", colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

# Obtendo novamente as variaveis mais importantes
importance  <- importance(rf_model)
var_Importance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Criando o rank de variaveis baseado na importancia
rankImportance <- var_Importance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importancia relativa das variaveis
ggplot(rankImportance, aes(x = reorder(Variables, Importance), y = Importance, fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

# Previsões
predictions <- data.frame(observado = test_data$Credit_Status,
                        previsto = predict(rf_model, newdata = test_data))
View(predictions)

# Matriz de Confusão
confusionMatrix(predictions$observado, predictions$previsto)

## Início da Otimização
# Criando uma Cost Function
Cost_func <- matrix(c(0, 1.5, 1, 0), nrow = 2, dimnames = list(c("0", "1"), c("0", "1")))

# Criando 2ª versão do modelo
c50_model  <- C5.0(Credit_Status ~ PAY_0
                   + PAY_2
                   + PAY_3
                   + PAY_4
                   + PAY_5
                   + PAY_6,
                   data = train_data_smoted,  
                   trials = 100,
                   cost = Cost_func)

summary(c50_model)

# Previsões 2º modelo
predictions_v2 <- data.frame(observado = test_data$Credit_Status,
                           previsto = predict(c50_model, newdata = test_data))

# Matriz de Confusão 
confusionMatrix(predictions_v2$observado, predictions_v2$previsto)

# Salvando o Modelo
saveRDS(rf_model, file = "rf-model.rds")

# Carregando o modelo treinado
modelo <- readRDS("rf-model.rds")