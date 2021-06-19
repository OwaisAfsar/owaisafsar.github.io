# Title     : TODO
# Objective : TODO
# Created by: mafsar
# Created on: 10.05.21

library("ROSE")

create_train_test <- function (data, split.size, is.train){
    n_row = nrow(data)
    total_row = split.size * n_row
    train_sample <- 1: total_row
    if (is.train == TRUE){
      return(data[train_sample, ])
    } else {
      return(data[-train_sample, ])
    }
}

ms.data <- read.csv(file = "CSVFiles/ms-data.csv", header = TRUE)
ms.data <- ms.data[,2:38]

shuffle_index <- sample(1:nrow(ms.data))
ms.data <- ms.data[shuffle_index,]

clean.ms.data <- ms.data[,c(3,7,25:32,37)]

data_train <- create_train_test(clean.ms.data, split.size = 0.80, is.train = TRUE)
data_test <- create_train_test(clean.ms.data, split.size =  0.80, is.train = FALSE)
#dim(data_train)
#dim(data_test)

#cat("Before ROSE: ", table(data_train$has_conflict))
data.balance.rose <- ROSE(has_conflict~., data=data_train, seed=1)$data
#cat("After ROSE: ", table(data.balance.rose$has_conflict))

#write.csv(data.balance.rose,'rose_train_data.csv')
#write.csv(data_test,'rose_test_data.csv')