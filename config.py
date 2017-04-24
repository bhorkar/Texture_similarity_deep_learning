CNNInfo = dict(
  caffe_root = '../../../bin/caffe/', 
  net = 'VGG',  
  input_dict = "../../fashion_train/temp/",
  layer = 'pool1',
  feature_size = 64*64,
  keep_aspect_ratio = False
)

LSFModelInfo = dict(
   model_dict = "LSFmodel/",
   indexFile =  "saveImage",
   nLSFModelTrees = 10,   #divide the training set into nLSF trees 
   n_candidates   = 50,
   n_estimators   = 10
)

AutoEncoderInfo = dict(
   enable_autoencoder = False,
   n_layers           = 0
)
 
