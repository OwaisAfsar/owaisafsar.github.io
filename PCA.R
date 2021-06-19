# Title     : TODO
# Objective : TODO
# Created by: mafsar
# Created on: 14.04.21

install.packages("factoextra")
library(factoextra)

ms.data <- read.csv(file = "CSVFiles/ms-data.csv", header = TRUE)
ms.data <- ms.data[,2:38]

shuffle_index <- sample(1:nrow(ms.data))
ms.data <- ms.data[shuffle_index,]

clean.ms.data <- ms.data[,c(3,7,25:32,37)]

res.pca <- prcomp(clean.ms.data, scale = TRUE)
fviz_eig(res.pca)

fviz_pca_ind(res.pca,
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )

fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )