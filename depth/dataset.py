"""
Dataloader for image frames
"""

class FrameLoader():
    '''
    Arg(s):
    video_path: list[str]
    list of paths for each video frame
    resize_shape: list[int]
    tuple of (n_height, n_width)
    '''

    def __init__(self, video_path, resize_shape):
        self.video_path = video_path
        self.resize_shape = resize_shape
        self.resize_height, self.resize_width = resize_shape

    def __getitem__(self, index):
        import cv2
        import numpy as np
        from PIL import Image

        #open image and convert to rgb
        image = Image.open(self.video_path[index]).convert('RGB')
        image = np.array(image)

        #resize with cv2
        image = cv2.resize(image, (self.resize_width, self.resize_height))

        image = np.transpose(image, (2, 0, 1))
        return image
        