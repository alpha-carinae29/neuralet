import pathlib
import time
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from libs.detectors.utils.fps_calculator import convert_infr_time_to_fps
import os
import wget


def load_model(gpu, checkpoint):
    base_path = "libs/detectors/x86/data/"
    if checkpoint == "crowd_human":
        config_name = "crowd_human_full_faster_rcnn_r50_fpn_2x.py"
        checkpoint_name = "crowd_human_full_faster_rcnn_r50_fpn_2x.pth"
    elif checkpoint == "wider_person":
        config_name = "wider_person_faster_rcnn_r50_fpn_2x.py"
        checkpoint_name = "wider_person_faster_rcnn_r50_fpn_2x.pth"
    else:
        raise ValueError("checkpoints should be either 'crowd_human' or 'wider_person' but {} provided".format(checkpoint))

    config_file = os.path.join(base_path, config_name)
    if not os.path.isfile(config_file):
        url = "https://raw.githubusercontent.com/saic-vul/iterdet/master/configs/iterdet/" + config_name
        print('config file does not exist under: ', config_file, 'downloading from ', url)
        wget.download(url, config_file)

    checkpoint_file = os.path.join(base_path, checkpoint_name)
    if not os.path.isfile(checkpoint_file):
        url = "https://github.com/saic-vul/iterdet/releases/download/v2.0.0/" + checkpoint_name
        print('checkpoint file does not exist under: ', checkpoint_file, 'downloading from ', url)
        wget.download(url, checkpoint_file)
    if gpu == "true":
        device = "cuda:0"
    else:
        device = "cpu"
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=device) 

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
        gpu = self.config.get_section_dict('Detector')['GPU']
        checkpoint = self.config.get_section_dict('Detector')['Checkpoint']
        self.detection_model = load_model(gpu, checkpoint)
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
