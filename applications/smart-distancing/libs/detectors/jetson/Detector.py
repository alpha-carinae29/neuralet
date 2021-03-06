
class Detector():
    def __init__(self, config):
        self.config = config
        self.net = None 
        self.name = self.config.get_section_dict('Detector')['Name']
        if self.name == 'ssd_mobilenet_v2_coco' : # or mobilenet_ssd_v1
            from . import MobileNetSSDV2
            self.net = MobileNetSSDV2.Detector(self.config)
        else:
            raise ValueError('Not supported network named: ', self.name)

    def inference(self, resized_rgb_image):
        # Here should inference on the image, output a list of objects, each obj is a dict with two keys "id" and "bbox" 
        # return [{"id": 0, "bbox": [x, y, w, h]}, {...}, {...}, ...]
        result = self.net.inference(resized_rgb_image)
        return result

