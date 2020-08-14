library(keras)
library(tensorflow)
library(magick)
library(dplyr)
library(caret)
# Instalation guide: https://keras.rstudio.com/

#########################################################################
## Import Images
#########################################################################

path <- "./images"
path2 <- "./images/evaluation"

training_path <- file.path(path, "train")
test_path <- file.path(path, "test")

image_options <- c("with_mask","no_mask")
output_n <- length(image_options)

train_data_gen = image_data_generator(
  rescale = 1/255,
  featurewise_center=TRUE,
  featurewise_std_normalization=TRUE,
  rotation_range=45
)

test_data_gen <- image_data_generator(
  rescale = 1/255,
  featurewise_center=TRUE,
  featurewise_std_normalization=TRUE,
  rotation_range=45
)

mask_eval_gen = image_data_generator(
  rescale = 1/255,
  featurewise_center=TRUE,
  featurewise_std_normalization=TRUE
)


# image properties
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)
channels <- 3

trainingImages <- flow_images_from_directory(training_path,
                                             train_data_gen,
                                             target_size = target_size,
                                             class_mode = 'categorical',
                                             classes = image_options,
                                             seed = 12)

testImages <- flow_images_from_directory(test_path, 
                                         test_data_gen,
                                         target_size = target_size,
                                         class_mode = 'categorical',
                                         classes = image_options,
                                         seed = 12)

mask_eval <- flow_images_from_directory( path2, 
                                         mask_eval_gen,
                                         target_size = target_size,
                                         class_mode = 'categorical',
                                         classes = image_options,
                                         shuffle = FALSE,
                                         seed = 12)

# check images loaded for training and test
table(factor(trainingImages$classes))
table(factor(testImages$classes))




#########################################################################
## Difine the model
#########################################################################

train_samples <- trainingImages$n
test_samples <- testImages$n
batch_size <- 32
epochs <- 30

model <- keras_model_sequential() %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid', input_shape = c(img_width, img_height, channels)) %>%
  layer_activation('relu') %>%
  layer_batch_normalization() %>%

  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%
  layer_dropout(0.30) %>%
  
  # Third hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%

  # Third hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%
  
  layer_dense(100) %>%
  layer_activation('relu') %>%
  
  # Third hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%
  layer_dropout(0.30) %>%

  # Third hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%
  
  # Fourth hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%
  
  layer_dense(100) %>%
  layer_activation('relu') %>%
  
  # Fifth hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation('relu') %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  layer_dropout(0.20) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation('relu') %>%
  layer_dropout(0.20) %>%

  layer_activation('relu') %>%
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation('softmax')

summary(model)

# compile
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.001, decay = 1e-2),
  metrics = metric_binary_accuracy
)


hist <- model %>% fit_generator(
  # training data
  trainingImages,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs,
  
  # validation data
  validation_data = testImages,
  validation_steps = as.integer(test_samples / batch_size),
  
  # print progress
  verbose = 0,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint(file.path(path, "mask_checkpoints.h5"), save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = file.path(path, "mask_logs"))
  )
)

plot(hist)
evaluate_generator(model,testImages,steps=length(testImages))

#########################################################################
## Test the model
#########################################################################

evaluate_generator(model,mask_eval,steps=length(mask_eval))

predictions <- predict_generator(model, mask_eval, steps=length(mask_eval))
predictions <- transform(predictions, predicted_class=apply(predictions, 1, which.max)-1)
predictions['original_classes'] <- mask_eval$classes
predictions <- predictions[,3:4]

confusionMatrix(
  factor(predictions$predicted_class, labels=c('with','without')),
  factor(predictions$original_classes, labels=c('with','without'))
  )