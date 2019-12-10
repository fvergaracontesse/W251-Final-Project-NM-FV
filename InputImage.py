import cv2 as cv2
import os.path
import  numpy as np
from os.path import isfile, join
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class InputImage:

    def __init__(self, filename):
        self.filename = filename
        self.img = None
        self.thresh = None
        self.dilate = None
        self.sub_images = []
        self.np_sub_images = []

    def readImage(self):
        try:
            if(os.path.exists(self.filename)):
              img = cv2.imread(self.filename)
              print("File read as image successfully")
              self.img = img
            else:
              print("File not found")
        except:
            print("Something went wrong reading image")

    def modifyImage(self):
        try:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dilate = cv2.dilate(thresh, kernel , iterations=4)
            self.thresh = thresh
            self.dilate = dilate
            print("Modification done successfully")
        except:
            print("Something went wrong modifying image")
        return False

    def getSubImages(self):
        try:
            # Find contours in the image
            cnts = cv2.findContours(self.dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts=  sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
            contours = []
            sub_images = []
            threshold_min_area = 400
            threshold_max_area = 6000
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area > threshold_min_area and area < threshold_max_area:
                    contours.append(c)
                    sub_images.append(self.img[y : y + h, x : x + w])
                    #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),1)
            self.sub_images = sub_images
            print("Sub images obtained successfully")
        except Exception as e:
            print("Something went wrong getting subimages")
            print (e)

    def saveSubImages(self):
        try:
            image_to_symbol_directory = 'images_to_symbol'+"/"+self.filename.split("/")[1].split(".")[0]
            if not os.path.exists(image_to_symbol_directory):
                os.makedirs(image_to_symbol_directory)
            for i,image in enumerate(self.sub_images):
                img_filename = image_to_symbol_directory+"/"+str(i)+".png"
                cv2.imwrite(img_filename,image)
            print("Subimages save successfully")
        except:
            print("Something went wrong saving subimages")

    def absoluteFilePaths(self,directory):
        paths=[]
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                #print (os.path.abspath(os.path.join(dirpath, f)))
                if isfile(os.path.abspath(os.path.join(dirpath, f))):
                    paths.append(os.path.abspath(os.path.join(dirpath, f)))
                else:
                    continue
        return paths

    def getLabelfromPath(self,path):
        return path.split('/')[-2]

    def convertToArray(self, image_loc):
        img = Image.open(image_loc).convert('L')
        img = img.resize((64,64),Image.ANTIALIAS)
        array = np.asarray(img)
        return array

    def prepareImagesForClassifications(self,labelPath):
        imgArray=[]
        labels = []
        image_to_symbol_directory = 'images_to_symbol'+"/"+self.filename.split("/")[1].split(".")[0]
        filePaths=self.absoluteFilePaths(image_to_symbol_directory)
        labelPaths=self.absoluteFilePaths(labelPath)
        for f in tqdm(filePaths):
            imgArray.append(self.convertToArray(f))
        for s in tqdm(labelPaths):
            labels.append(self.getLabelfromPath(s))
        data=np.array([imgArray]).T
        #b=np.array([x.ravel() for x in data.ravel()])
        labels_final = list(set(labels))
        labels_final.sort()
        np_image_array=data.reshape(len(imgArray),100,100,1)
        return np_image_array, labels_final
    
    def drawImage(self):
        plt.imshow(convertToArray(image))
        plt.show()
        

