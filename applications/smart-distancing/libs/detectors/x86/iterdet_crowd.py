import pathlib
import time
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from libs.detectors.utils.fps_calculator import convert_infr_time_to_fps
import os
import wget


def load_model():
    base_path = "libs/detectors/x86/data/"
    config_file = os.path.join(base_path, "crowd_human_full_faster_rcnn_r50_fpn_2x.py")
    if not os.path.isfile(config_file):
        url = "https://raw.githubusercontent.com/saic-vul/iterdet/master/configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py" 
        print('config file does not exist under: ', config_file, 'downloading from ', url)
        wget.download(url, config_file)

    checkpoint_file = os.path.join(base_path, "crowd_human_full_faster_rcnn_r50_fpn_2x.pth")
    if not os.path.isfile(checkpoint_file):
        url = "https://github.com/saic-vul/iterdet/releases/download/v2.0.0/crowd_human_full_faster_rcnn_r50_fpn_2x.pth"
        print('checkpoint file does not exist under: ', checkpoint_file, 'downloading from ', url)
        wget.download(url, checkpoint_file)

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0') 

    return model


class Detector:
    """
    Perform object detection with the given model. The model is a quantized tflite
    file which if the detector can not find it at the path it will download it
    from neuralet repository automatically.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        # Get the model name from the config
        self.model_name = self.config.get_section_dict('Detector')['Name']
        # Frames Per Second
        self.fps = None

        self.detection_model = load_model()
        self.w , self.h, _ = [int(i) for i in self.config.get_section_dict('Detector')['ImageSize'].split(',')]
    def inference(self, resized_rgb_image):
        """
        inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding detection output which is used for creating result
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)

        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        t_begin = time.perf_counter()
        output_dict = inference_detector(self.detection_model, resized_rgb_image)
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)

        class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])
        result = []
        for i, box in enumerate(output_dict[0]):  # number of boxes
            if box[-1]  > score_threshold:
                result.append({"id": str(class_id) + '-' + str(i), "bbox": [box[1] / self.h, box[0] / self.w, box[3] / self.h , box[2] / self.w], "score": box[-1]})

        return result
