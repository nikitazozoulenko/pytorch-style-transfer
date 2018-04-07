from __future__ import print_function
from __future__ import division

from threading import Thread

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image

class WebcamVideoStream:
    def __init__(self, src=0):
	# initialize the video camera stream and read the first frame
	# from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
 
	# initialize the variable used to indicate if the thread should
	# be stopped
        self.stopped = False

    def start(self):
	# start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return self.frame
 
    def stop(self):
	# indicate that the thread should be stopped
        self.stopped = True

def numpy_to_cuda(numpy_array):
    return Variable(torch.from_numpy(numpy_array).cuda().permute(2,0,1).float().unsqueeze(0), volatile=True)

#model = torch.load("savedir/facenet_1_0.003_it200k.pt")
#model = torch.load("savedir/model_acidcrop_it90k.pt")
model = torch.load("savedir/model_small_acidcrop_it100k.pt")
model.eval()
upsample = nn.Upsample(size=(240*4, 320*4), mode = "nearest")
    
# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
stream = WebcamVideoStream(src=0).start()
test = True
while True:
    frame = stream.read()
    #frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cuda_frame = numpy_to_cuda(frame)/127.5 - 1
    styled_frame = model(cuda_frame)[0]
    styled_frame = torch.stack((styled_frame[2], styled_frame[1], styled_frame[0]), dim=0)
    styled_frame = (styled_frame.data.permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
    # check to see if the frame should be displayed to our screen
    cv2.imshow("Frame", styled_frame)
    key = cv2.waitKey(1) & 0xFF
 
cv2.destroyAllWindows()
vs.stop()
                






