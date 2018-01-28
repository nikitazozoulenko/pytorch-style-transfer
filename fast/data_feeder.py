"""Taken and modified from
https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
credits goes to them"""

import os
import threading
from threading import Thread
from queue import Empty,Full,Queue

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.autograd import Variable
from torchvision import datasets

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        
    def __iter__(self):
        return self

    def __next__(self):
       	with self.lock:
            return self.it.__next__()

        
def get_path_i(paths_count):
    """Cyclic generator of paths indice
	"""
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id	= (current_path_id + 1) % paths_count

        
class InputGen:
    def __init__(self, coco_path, annFile, style_path, batch_size):
        self.coco_data = datasets.CocoCaptions(root = coco_path, annFile = annFile)
        self.painters_data = datasets.ImageFolder(style_path)
        self.paths = np.arange(min(len(self.coco_data), len(self.painters_data)))
        self.index = 0
        self.batch_size = batch_size
        self.init_count = 0
        self.lock = threading.Lock() #mutex for input path
        self.yield_lock = threading.Lock() #mutex for generator yielding of batch
        self.path_id_generator = threadsafe_iter(get_path_i(len(self.paths)))
        self.cumulative_batch = []

        
    def get_samples_count(self):
        """ Returns the total number of images needed to train an epoch """
        return len(self.paths)

    
    def get_batches_count(self):
        """ Returns the total number of batches needed to train an epoch """
        return int(self.get_samples_count() / self.batch_size)

    
    def __next__(self):
        return self.__iter__()

    
    def __iter__(self):
        while True:
            #In the start of each epoch we shuffle the data paths
            with self.lock:
                if (self.init_count == 0):
                    np.random.shuffle(self.paths)
                    self.cumulative_batch = []
                    self.init_count = 1
	        #Iterates through the input paths in a thread-safe manner
            for path_id in self.path_id_generator:
                idx = self.paths[path_id]
                
                style = self.painters_data[idx][0]
                style = process_example(style)
                content = self.coco_data[idx][0]
                content = process_example(content)
                
                #Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.cumulative_batch)) < self.batch_size:
                        self.cumulative_batch += [[style, content]]
                    if len(self.cumulative_batch) % self.batch_size == 0:
                        final_batch = make_batch_from_list(self.cumulative_batch)
                        yield final_batch
                        self.cumulative_batch = []
	    #At the end of an epoch we re-init data-structures
            with self.lock:
                self.init_count = 0

                
    def __call__(self):
        return self.__iter__()

    
class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.to_kill = False

        
    def __call__(self):
        return self.to_kill

    
    def set_tokill(self,tokill):
        self.to_kill = tokill

        
def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for i, batch in enumerate(dataset_generator):
            #We fill the queue with new fetched batch until we reach the max size.
            batches_queue.put(batch, block=True)
            if tokill() == True:
                return

            
def threaded_cuda_batches(tokill,cuda_batches_queue,batches_queue, volatile):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch_styles, batch_contents = batches_queue.get(block=True)
        batch_styles = torch.from_numpy(batch_styles.transpose(0,3,1,2))
        batch_contents = torch.from_numpy(batch_contents.transpose(0,3,1,2))

        batch_styles = Variable(batch_styles, volatile = volatile).cuda()
        batch_contents = Variable(batch_contents, volatile = volatile).cuda()
        cuda_batches_queue.put((batch_styles, batch_contents), block=True)
        if tokill() == True:
            return

        
class DataFeeder(object):
    def __init__(self, coco_path, annFile, style_path, preprocess_workers = 4, cuda_workers = 1,
                 numpy_size = 12, cuda_size = 2, batch_size = 4, volatile = False):
        self.preprocess_workers = preprocess_workers
        self.cuda_workers = cuda_workers
        self.volatile = volatile        

        #Our train batches queue can hold at max 12 batches at any given time.
	#Once the queue is filled the queue is locked.
        self.train_batches_queue = Queue(maxsize=numpy_size)

	#Our numpy batches cuda transferer queue.
	#Once the queue is filled the queue is locked
	#We set maxsize to 3 due to GPU memory size limitations
        self.cuda_batches_queue = Queue(maxsize=cuda_size)

        #thread killers for ending threads
        self.train_thread_killer = thread_killer()
        self.train_thread_killer.set_tokill(False)
        self.cuda_thread_killer = thread_killer()
        self.cuda_thread_killer.set_tokill(False)

        #input generators
        self.input_gen = InputGen(coco_path, annFile, style_path, batch_size)


    def start_queue_threads(self):
        for _ in range(self.preprocess_workers):
            t = Thread(target=threaded_batches_feeder, args=(self.train_thread_killer, self.train_batches_queue, self.input_gen))
            t.start()
        for _ in range(self.cuda_workers):
            cudathread = Thread(target=threaded_cuda_batches, args=(self.cuda_thread_killer, self.cuda_batches_queue, self.train_batches_queue, self.volatile))
            cudathread.start()

            
    def kill_queue_threads(self):
        self.train_thread_killer.set_tokill(True)
        self.cuda_thread_killer.set_tokill(True)
        for _ in range(self.preprocess_workers):
            try:
                #Enforcing thread shutdown
                self.train_batches_queue.get(block=True,timeout=1)
            except Empty:
                pass
        for _ in range(self.cuda_workers):
            try:
                #Enforcing thread shutdown
                self.cuda_batches_queue.get(block=True,timeout=1)
            except Empty:
                pass

            
    def get_batch(self):
        return self.cuda_batches_queue.get(block=True)

    
def process_example(image):
    #flip horizontally
    random = np.random.randint(0,2)
    if(random == 0):
        image = ImageOps.mirror(image)

    return image


def make_batch_from_list(cumulative_batch):
    styles = [x[0] for x in cumulative_batch]
    contents = [x[1] for x in cumulative_batch]

    size = 256
    xrand = np.random.randint(0,12)
    yrand = np.random.randint(0,12)
    resize_size = (size + 4*xrand, size + 4*yrand)
    resized_styles = [np.asarray(style.resize(resize_size)) for style in styles]
    resized_contents = [np.asarray(content.resize(resize_size)) for content in contents]

    return np.array(resized_styles).astype(np.float32), np.array(resized_contents).astype(np.float32)
