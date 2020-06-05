library(caret)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)

#######################
## TRAINING FUNCTION ##
#######################

training <- function(input_x, input_y, method, idx) {

  #### trainControl  ####
  tune_control <- caret::trainControl(
    method = "repeatedcv", 
    index = idx,
    verboseIter = TRUE,
    allowParallel = F,
    classProbs = ifelse(is.factor(input_y), TRUE, FALSE),
    savePredictions = "final")
  
  #### Train ####
  model <- caret::train(
    x = input_x,
    y = input_y,
    method = method,
    trControl = tune_control,
    importance = "permutation", #to get feature importance
    tuneLength = 10,            #number of default values to try
    metric = ifelse(is.factor(input_y), "Kappa", "RMSE"))
  
  return(model)
}

#######################
## ARGUMENTS & DATA ###
#######################

args     <- commandArgs(trailingOnly=TRUE)

run           <- args[1]   #run = 1:10
method        <- args[2]   #method = xgbTree, ranger, gbm, rf, pls,   pcr
thresholds    <- args[3]
transform     <- args[4]
outfile       <- args[5]   #outfilebase = e.g. "outfiles/age_regression.outfile.compounds.1.ranger.1"

#### For local testing ####
if (is.na(run)) {
  run         <- 1                 
  method      <- "ranger"
  outfile     <- "testing_run"
  thresholds  <- "7"
  transform   <- "0.5_5"
}

#### Loading data + setting seed ####
variables   <- "values_c"
ms          <- read_rds(path = "../data/data_object_2.rds")
run         <- as.integer(run)
thresholds  <- str_split(thresholds, "_")[[1]] %>% as.numeric()
transform   <- str_split(transform, "_")[[1]] %>% as.numeric()


set.seed(run)

folds = 10
repeats = 5

#### Choosing features ####
features <- c("M227T34", "M428T807", "M897T924", "M333T32", "M209T33_2", 
              "M183T32", "M285T73", "M265T32", "M569T863", "M319T68", "M565T831", 
              "M149T32", "M457T195", "M545T831", "M871T925", "M281T32", "M900T941", 
              "M429T807", "M620T33", "M577T35", "M329T38", "M530T34", "M647T835", 
              "M899T941", "M215T32_1", "M411T807", "M593T34", "M898T924", "M625T33", 
              "M302T74", "M609T34", "M718T816", "M393T32", "M648T835", "M355T65", 
              "M351T32", "M901T941", "M524T831", "M611T33", "M541T832", "M902T154", 
              "M903T154", "M904T152", "M654T90", "M655T89", "M885T152", "M907T152", 
              "M886T152", "M656T89", "M696T89")

ms$values_c <- subset(ms$values_c, select=features)

####################
### PARTITIONING ###
####################

#### Class creation ####
if (length(thresholds) == 2) { #If 3 classes
  x <- ms$rowinfo %>% 
    filter(!grepl(pattern = "QC", x = day)) %>%
    mutate(tmp = as.integer(day)) %>%
    mutate(class = NA) %>%
    mutate(class = ifelse(tmp <= thresholds[1], "c1", class)) %>%
    mutate(class = ifelse(tmp > thresholds[1] & tmp <= thresholds[2], "c2", class)) %>%
    mutate(class = ifelse(tmp > thresholds[2] , "c3", class)) %>%
    mutate(class = factor(class, levels=c("c1", "c2", "c3"))) %>%
    {.} 
}

if (length(thresholds) == 1) { #If 2 classes
  x <- ms$rowinfo %>% 
    filter(!grepl(pattern = "QC", x = day)) %>%
    mutate(tmp = as.integer(day)) %>%
    mutate(class = NA) %>%
    mutate(class = ifelse(tmp <= thresholds[1], "c1", class)) %>%
    mutate(class = ifelse(tmp > thresholds[1], "c2", class)) %>%
    mutate(class = factor(class, levels=c("c1", "c2"))) %>%
    {.} 
}

#### Partitioning (80/20) ####
full_data <- x %>% 
  select(class, day, batch) %>% 
  bind_cols(ms$values_c[x$rowid,])

hid <- sample(1:nrow(full_data), size = ceiling(nrow(full_data)*0.8))
training_data   <- full_data[hid,]
validation_data <- full_data[-hid,]

# Training data
input_x <- as.matrix(select(training_data, -class, -day, -batch))
input_y <- training_data$class
regin_y <- training_data$day %>% as.numeric()

# Validation data
holdout_x <- select(validation_data, -class, -day, - batch)
holdout_y <- validation_data$class
regout_y <- validation_data$day %>% as.numeric()

##############################
### CROSS VALIDATION SETUP ###
##############################

input_classes    <- levels(full_data$class)
multi_folds      <- list()
multi_folds$all  <- createMultiFolds(1:200, k = folds, times = repeats) #structure for overwriting

for (class in input_classes) {
  
  #### Creating folds x repeats for each class ####
  #Finding class indices in training data
  idx <- tibble(obs = input_y) %>% 
    mutate(rowid = row_number()) %>% 
    filter(obs == class)
  
  #Creating class folds * repeats
  multi_folds$class     <- createMultiFolds(idx$rowid, k = folds, times = repeats)
  
  #### Updating fold indices to match training data ####
  for (i in 1:(folds*repeats)) {
    
    #Merging folds for classification
    if (class == "c1") {
      multi_folds$all[[i]] <- idx$rowid[multi_folds$class[[i]]]
    }
    if (class != "c1") {
      multi_folds$all[[i]] <- c(multi_folds$all[[i]], idx$rowid[multi_folds$class[[i]]])
    }
  }
  names(multi_folds)[names(multi_folds) == "class"]   <- paste(class)
}

######################
### CLASSIFICATION ###
######################

#### Training ####
starttime <- Sys.time()
class_model <- training(input_x, input_y, method, multi_folds$all)

#### Evaluation ####
validation <- tibble(obs = regout_y, obs_class = holdout_y) %>% 
  bind_cols(as_tibble(predict(class_model, newdata = holdout_x, type="prob")))

tmp         <- confusionMatrix(data = class_model$pred$pred, reference = class_model$pred$obs)
kappa_cv    <- tmp$overall["Kappa"]
accuracy_cv <- tmp$overall["Accuracy"]

tmp          <- confusionMatrix(data = predict(class_model, newdata = holdout_x), reference = holdout_y)
kappa_val    <- tmp$overall["Kappa"]
accuracy_val <- tmp$overall["Accuracy"]

#### Feature importance ####
importance <- varImp(class_model, scale = F)$importance
importance <- tibble(names = rownames(importance), classification = importance$Overall)

#######################
## REGRESSION ON ALL ##
#######################

reg_all       <- training(input_x, regin_y, method, multi_folds$all)

rmse_all_cv   <- min(reg_all$results$RMSE)
rmse_all_val  <- sqrt(sum((regout_y - predict(reg_all, newdata = holdout_x))^2)/length(regout_y))

#############################
## REGRESSION ON CLASSES  ###
#############################

reg_models = list()

for (i in input_classes) {
  set.seed(run)
  reg_models$class <- list()
  
  #### Fetching class ####
  regin_x <- filter(training_data, class == i) %>% select(-class,-day,-batch) %>% as.matrix()
  regin_y <- filter(training_data, class == i)$day %>% as.numeric()

  #### Training ####
  model                     <- training(regin_x, regin_y, method, multi_folds[[i]])
  reg_models$class$model    <- model
  
  #### Predictions on all classes ####
  pred <- tibble()
  
  for (j in 1:(folds*repeats)) {
    #folds, repeats
    idx       <- multi_folds[[i]][[j]]
    idx_test  <- multi_folds$all[[j]]
    
    #Re-train using cv-parameters
    regin_x   <- filter(training_data, class == i) %>% slice(idx) %>% select(-class,-day,-batch) %>% as.matrix()
    regin_y   <- filter(training_data, class == i)$day[idx] %>% as.numeric()
    
    regall_x <- training_data %>% slice(-idx_test) %>% select(-class,-day,-batch) %>% as.matrix()
    
    model <- caret::train(x          = regin_x,
                          y          = regin_y,
                          method     = method,
                          tuneGrid   = reg_models$class$model$bestTune,
                          trControl  = trainControl(method = "none"))

    tmp <- tibble(pred           = predict(model, newdata = regall_x),
                  obs            = (training_data %>% slice(-idx_test))$day,
                  obs_class      = (training_data %>% slice(-idx_test))$class,
                  batch          = (training_data %>% slice(-idx_test))$batch,
                  rowIndex       = setdiff(rep(1:129),idx_test),
                  Resample       = names(multi_folds[[i]])[j])
    
    pred <- bind_rows(pred, tmp)
  }
  
  #Saving predictions
  names(pred)[names(pred) == "pred"] <- paste("pred.", i, sep = "") #assigning current class name
  reg_models$class$pred_all <- pred
  
  #Saving model
  names(reg_models)[names(reg_models) == "class"] <- paste(i) #assigning current class name
  
  #### Evaluation ####
  validation <- bind_cols(validation, as_tibble(predict(model, newdata = holdout_x)))
  names(validation)[names(validation) == "value"] <- paste("pred.", i, sep = "") #assigning current class name
}

#### Feature importance ####
for (i in 1:length(input_classes)) {
  #fetching importance for ith class
  tmp <- varImp(reg_models[[i]]$model, scale = F)$importance
  tmp <- tibble(names = rownames(tmp), mda = tmp$Overall)
  names(tmp)[names(tmp) == "mda"] <- paste(input_classes[i]) #assigning class name
  
  #adding to df
  importance <- merge(importance,tmp,by = c("names"))
}

###################
### PERFORMANCE ###
###################

#### Gathering and transforming results: Test data ####
data_test <- reg_models[["c1"]][[2]]

for (i in 2:length(input_classes)) {
  data_test <- merge(data_test, 
                     reg_models[[i]][[2]], 
                     by = c("obs","obs_class","rowIndex","Resample","batch"))
}

if (length(thresholds) == 2) { #3 classes
  data_test <- merge(data_test,
                     (class_model$pred[4:10] %>% select(-obs) %>% rename(pred_class = pred)),
                     by = c("rowIndex","Resample")) %>% 
    mutate_at(10:12, ~ ((. + transform[1])^transform[2]))

  data_test <- data_test %>% 
    mutate_at(10:12, ~ ./rowSums(data_test[,10:12])) %>% 
    mutate(pred_ens = pred.c1 * c1 + pred.c2 * c2 + pred.c3 * c3,
           pred = case_when(c1 > c2 & c1 > c3 ~ pred.c1,
                            c2 > c1 & c2 > c3 ~ pred.c2,
                            c3 > c2 & c3 > c1 ~ pred.c3)) %>% 
    select(-rowIndex, -Resample)
}

if (length(thresholds) == 1) { #2 classes
  data_test <- merge(data_test,
                     (class_model$pred[4:9] %>% select(-obs) %>% rename(pred_class = pred)),
                     by = c("rowIndex","Resample")) %>% 
    mutate_at(9:10, ~ ((. + transform[1])^transform[2]))
  
  data_test <- data_test %>% 
    mutate_at(9:10, ~ ./rowSums(data_test[,9:10])) %>% 
    mutate(pred_ens = pred.c1 * c1 + pred.c2 * c2,
           pred = case_when(c1 > c2 ~ pred.c1,
                            c2 > c1 ~ pred.c2)) %>% 
    select(-rowIndex, -Resample)
}

#### Gathering and transforming results: Validation data ####

if (length(thresholds) == 2) { #3 classes
  data_val <- validation %>% mutate_at(3:5, ~ (. + transform[1])^transform[2])

  data_val <- data_val %>% 
    mutate_at(3:5, ~ ./rowSums(data_val[,3:5])) %>% 
    mutate(pred_ens = pred.c1 * c1 + pred.c2 * c2 + pred.c3 * c3,
                                    pred = case_when(c1 > c2 & c1 > c3 ~ pred.c1,
                                                     c2 > c1 & c2 > c3 ~ pred.c2,
                                                     c3 > c2 & c3 > c1 ~ pred.c3))
}

if (length(thresholds) == 1) { #2 classes
  data_val <- validation %>% mutate_at(3:4, ~ (. + transform[1])^transform[2])
  
  data_val <- data_val %>% 
    mutate_at(3:4, ~ ./rowSums(data_val[,3:4])) %>% 
    mutate(pred_ens = pred.c1 * c1 + pred.c2 * c2,
           pred = case_when(c1 > c2 ~ pred.c1,
                            c2 > c1 ~ pred.c2))
}

#### True class regression results ####
true_class_reg    <- tibble(rmse_cv = NA, rmse_val = NA)

for (i in 1:length(input_classes)) {
  holdout_x <- filter(validation_data, class == input_classes[i]) %>% select(-class, -day, - batch)
  regout_y  <- filter(validation_data, class == input_classes[i])$day %>% as.numeric()
  
  true_class_reg$rmse_cv   <- min(reg_models[[i]]$model$results$RMSE)
  true_class_reg$rmse_val  <- sqrt(sum((regout_y - predict(reg_models[[i]]$model, newdata = holdout_x))^2)/length(regout_y))
  
  #renaming
  names(true_class_reg)[names(true_class_reg) == "rmse_cv"] <- paste("rmse", input_classes[i], "cv", sep = "_")
  names(true_class_reg)[names(true_class_reg) == "rmse_val"] <- paste("rmse", input_classes[i], "val", sep = "_")
}

#### Timing ####
endtime <- Sys.time()
runtime <- endtime - starttime

#### Calculating relevant statistics ####
r <- tibble(run          = run, 
            method       = method,
            class_range  = paste(thresholds, collapse = "_"),
            transform    = paste(transform, collapse = "_"),
            runtime      = as.double(runtime, units="secs"),
            data_test    = list(data_test),
            data_val     = list(data_val),
            importance   = list(importance)) %>% 
    
    bind_cols(#results
            kappa_cv = kappa_cv, kappa_val = kappa_val, 
            accuracy_cv = accuracy_cv, accuracy_val = accuracy_val,
            rmse_cv      = postResample(data_test$pred, as.numeric(data_test$obs))[[1]],
            rmse_val     = postResample(data_val$pred, data_val$obs)[[1]],
            rmse_ens_cv  = postResample(data_test$pred_ens, as.numeric(data_test$obs))[[1]],
            rmse_ens_val = postResample(data_val$pred_ens, data_val$obs)[[1]],
            rmse_all_cv  = rmse_all_cv,
            rmse_all_val = rmse_all_val,
            true_class_reg)

pd <- r %>% select(-data_test, -data_val, -importance)

importance
#### Saving output files ####
write_rds(x=r, path=paste(outfile,".rds", sep=""))
write_tsv(x=pd, path=paste(outfile,".tsv", sep=""))