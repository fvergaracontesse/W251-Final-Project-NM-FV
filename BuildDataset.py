from skimage.transform import resize
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import split_folders
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from os.path import isfile, join
from PIL import Image

class BuildDataset:

    def __init__(self):
        self.input_directory = 'input/'
        self.train_dataset_directory = 'CROHME_training_2011/'
        self.seed = 999
        self.final_train_directory = 'Image_data/finaltrain/'
        self.split_folders = 'final_output_images'

    def seedEverything(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)

    def getTracesData(self, inkml_file_abs_path):

    	traces_data = []

    	tree = ET.parse(inkml_file_abs_path)
    	root = tree.getroot()
    	doc_namespace = "{http://www.w3.org/2003/InkML}"

    	'Stores traces_all with their corresponding id'
    	traces_all = [{'id': trace_tag.get('id'),
    					'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
    								else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord.split(' ')] \
    							for coord in (trace_tag.text).replace('\n', '').split(',')]} \
    							for trace_tag in root.findall(doc_namespace + 'trace')]

    	'Sort traces_all list by id to make searching for references faster'
    	traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    	'Always 1st traceGroup is a redundant wrapper'
    	traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    	if traceGroupWrapper is not None:
    		for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

    			label = traceGroup.find(doc_namespace + 'annotation').text

    			'traces of the current traceGroup'
    			traces_curr = []
    			for traceView in traceGroup.findall(doc_namespace + 'traceView'):

    				'Id reference to specific trace tag corresponding to currently considered label'
    				traceDataRef = int(traceView.get('traceDataRef'))

    				'Each trace is represented by a list of coordinates to connect'
    				single_trace = traces_all[traceDataRef]['coords']
    				traces_curr.append(single_trace)


    			traces_data.append({'label': label, 'trace_group': traces_curr})

    	else:
    		'Consider Validation data that has no labels'
    		[traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    	return traces_data

    def inkml2img(self, input_path, output_path):
        traces = self.getTracesData(input_path)
        path = input_path.split('/')
        path = path[len(path)-1].split('.')
        path = path[0]+'_'
        file_name = 0
        for elem in traces:
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.axes().spines['top'].set_visible(False)
            plt.axes().spines['right'].set_visible(False)
            plt.axes().spines['bottom'].set_visible(False)
            plt.axes().spines['left'].set_visible(False)
            ls = elem['trace_group']
            output_path = output_path

            for subls in ls:
                data = np.array(subls)
                x,y=zip(*data)
                plt.plot(x,y,linewidth=2,c='black')

            capital_list = ['A','B','C','F','X','Y']
            if elem['label'] in capital_list:
                label = 'capital_'+elem['label']
            else:
                label = elem['label']
            ind_output_path = output_path + label
            try:
                os.mkdir(ind_output_path)
            except OSError:
                pass
            else:
                pass
            if(os.path.isfile(ind_output_path+'/'+path+str(file_name)+'.png')):
                file_name += 1
                plt.savefig(ind_output_path+'/'+path+str(file_name)+'.png', bbox_inches='tight', dpi=100)
            else:
                plt.savefig(ind_output_path+'/'+path+str(file_name)+'.png', bbox_inches='tight', dpi=100)
            plt.gcf().clear()

    def ensureDir(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def transformImages(self):
        files = os.listdir(self.input_directory+self.train_dataset_directory)
        for file in tqdm(files):
            self.inkml2img(self.input_directory+self.train_dataset_directory+file, self.final_train_directory)

    def absoluteFilePaths(self,directory):
        paths=[]
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                print(f)
                #print (os.path.abspath(os.path.join(dirpath, f)))
                if isfile(os.path.abspath(os.path.join(dirpath, f))):
                    paths.append(os.path.abspath(os.path.join(dirpath, f)))
                else:
                    continue
        return paths

    def getLabelfromPath(self, path):
        return path.split('/')[-2]

    def convertToArray(self, image_loc):
        img = Image.open(image_loc).convert('L')
        img = img.resize((64,64),Image.ANTIALIAS)
        array = np.asarray(img)
        return array

    def splitFolders(self):
        split_folders.ratio(self.final_train_directory, output=self.split_folders, seed=self.seed, ratio=(.8, .2))

    def buildDataset(self, path,target):
        filePaths=self.absoluteFilePaths(path)
        imgArray=[]
        labels=[]

        for f in tqdm(filePaths):
            imgArray.append(self.convertToArray(f))
            labels.append(self.getLabelfromPath(f))
        data=np.array([imgArray, labels]).T
        np.savez_compressed(target,data=data)

    def buildMinDataset(self, path,target):
        filePaths=self.absoluteFilePaths(path)
        imgArray=[]
        labels=[]
        for f in tqdm(filePaths):
            if self.getLabelfromPath(f) in ['(',')','+','-','1','2','3','4','=','x','y']:
                imgArray.append(self.convertToArray(f))
                labels.append(self.getLabelfromPath(f))
        data=np.array([imgArray, labels]).T
        np.savez_compressed(target,data=data)
