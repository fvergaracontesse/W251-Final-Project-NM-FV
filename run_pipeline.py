from BuildDataset import BuildDataset
from InputImage import InputImage
from BaselineModel import BaselineModel

#Create training dataset
#build_dataset = BuildDataset()
#build_dataset.seedEverything()
#build_dataset.ensureDir('Image_data/finaltrain/')
#build_dataset.ensureDir('final_output_images/train/')
#build_dataset.ensureDir('final_output_images/val/')
#build_dataset.transformImages()
#build_dataset.splitFolders()
##train dataset
#build_dataset.buildDataset('final_output_images/train/','./training_data')
##eval dataset
#build_dataset.buildDataset('final_output_images/val/','./val_data')
#
##check if baseline model exists
##if not exist build baseline model
baseline_model = BaselineModel()
baseline_model.setupDatasets()
baseline_model.setupModel1()
baseline_model.fitModel()
baseline_model.saveModel()


#once model exists run input image process
input_image = InputImage('equation/ex_1.png')
#input_image.readImage()
#input_image.modifyImage()
#input_image.getSubImages()
#input_image.saveSubImages()
#np_image_array,labels = input_image.prepareImagesForClassifications('final_output_images/train/')

#predict classes for individual features
baseline_model = BaselineModel()
baseline_model.loadModel()
#predicted_classes = baseline_model.predictClasses(np_image_array)
#print(labels[predicted_classes[1]])


#build equations
