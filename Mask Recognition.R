library(keras)
library(tensorflow)
# Instalation guide: https://keras.rstudio.com/

#########################################################################
## Import Images
#########################################################################

path <- "./images"
training_path <- file.path(path, "train")
test_path <- file.path(path, "test")

image_options <- c("with_mask","no_mask")
output_n <- length(image_options)

train_data_gen = image_data_generator(
  rescale = 1/255
)

test_data_gen <- image_data_generator(
  rescale = 1/255
)

trainingImages <- flow_images_from_directory(training_path,
                                             train_data_gen,
                                             class_mode = 'categorical',
                                             classes = image_options,
                                             seed = 12)

testImages <- flow_images_from_directory(test_path, 
                                         test_data_gen,
                                         class_mode = 'categorical',
                                         classes = image_options,
                                         seed = 12)

# check images loaded for training and test
table(factor(trainingImages$classes))
table(factor(testImages$classes))


# image properties
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)
channels <- 3

#########################################################################
## Difine the model
#########################################################################

train_samples <- trainingImages$n
test_samples <- testImages$n
batch_size <- 32
epochs <- 10

model <- keras_model_sequential() %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'valid', input_shape = c(img_width, img_height, channels)) %>%
  layer_activation('relu') %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = 'valid') %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  #layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation('relu') %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation('softmax')

summary(model)

# compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = 'accuracy'
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
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint(file.path(path, "mask_checkpoints.h5"), save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = file.path(path, "mask_logs"))
  )
)