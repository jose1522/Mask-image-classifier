library(keras)
library(tensorflow)
#########################################################################
## Import Images
#########################################################################

path <- "/"
mask_path <- file.path(path, "with_mask", TRUE)
no_mask_path <- file.path(path, "no_mask", FALSE)

mask_images <- flow_images_from_directory(mask_path, 
                                          class_mode = 'binary',
                                          seed = 12)


no_mask_images <- flow_images_from_directory(mask_path, 
                                          class_mode = 'binary',
                                          seed = 12)