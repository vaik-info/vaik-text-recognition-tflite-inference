from typing import List, Dict, Tuple
import multiprocessing
import platform
from PIL import Image
import numpy as np
import json
import tflite_runtime.interpreter as tflite
import tensorflow as tf


class TfliteModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None,
                 feature_divide_num=16, softmax_threshold=0.01, num_thread: int = None):
        self.classes = classes
        self.blank_index = len(classes) - 1
        self.feature_divide_num = feature_divide_num
        self.softmax_threshold = softmax_threshold
        num_thread = multiprocessing.cpu_count() if num_thread is None else num_thread
        self.__load(input_saved_model_path, num_thread)

    def inference(self, input_image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        resized_image_array = self.__preprocess_image(input_image, self.model_input_shape[1:3])
        raw_pred = self.__inference(resized_image_array)
        output = self.__output_parse(raw_pred)
        return output, raw_pred

    def __load(self, input_saved_model_path: str, num_thread: int):
        try:
            self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, num_threads=num_thread)
            self.interpreter.allocate_tensors()
        except RuntimeError:
            _EDGETPU_SHARED_LIB = {
                'Linux': 'libedgetpu.so.1',
                'Darwin': 'libedgetpu.1.dylib',
                'Windows': 'edgetpu.dll'
            }[platform.system()]
            delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB)]
            self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, experimental_delegates=delegates,
                                                  num_threads=num_thread)
            self.interpreter.allocate_tensors()
        self.model_input_shape = self.interpreter.get_input_details()[0]['shape']

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> np.ndarray:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]), dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize((int(resize_size[0]), int(resize_size[1])))
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image

    def __inference(self, resized_image: np.ndarray) -> np.ndarray:
        if len(resized_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resized_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resized_image.dtype}')
        self.__set_input_tensor(resized_image)
        self.interpreter.invoke()
        raw_pred = self.__get_output_tensor()[0]
        return raw_pred

    def __output_parse(self, raw_pred: np.ndarray) -> Dict:
        text, label_indexes, score = self.__decode(raw_pred)
        output_dict = {'text': text,
                       'classes': label_indexes,
                       'scores': score}
        return output_dict

    def __set_input_tensor(self, image: np.ndarray):
        input_tensor = self.interpreter.tensor(self.interpreter.get_input_details()[0]['index'])()
        input_tensor.fill(0)
        input_image = image.astype(self.interpreter.get_input_details()[0]['dtype'])
        input_tensor[0, :input_image.shape[0], :input_image.shape[1], :input_image.shape[2]] = input_image

    def __get_output_tensor(self) -> List[np.ndarray]:
        output_details = self.interpreter.get_output_details()
        output_tensor = []
        for index in range(len(output_details)):
            output = self.interpreter.get_tensor(output_details[index]['index'])
            scale, zero_point = output_details[index]['quantization']
            if scale > 1e-4:
                output = scale * (output - zero_point)
            output_tensor.append(output)
        return output_tensor

    @classmethod
    def char_json_read(cls, char_json_path):
        with open(char_json_path, 'r') as f:
            json_dict = json.load(f)
        classes = []
        for character_dict in json_dict['character']:
            classes.extend(character_dict['classes'])
        return classes

    def __softmax(self, matrix):
        return np.exp(matrix) / np.sum(np.exp(matrix))

    def __decode(self, raw_pred):
        threshold_pred = np.copy(raw_pred)
        threshold_pred_min, threshold_pred_max = np.min(threshold_pred), np.max(threshold_pred)
        threshold_blank_max = np.max(threshold_pred[:, :, -1])

        threshold_pred_softmax = tf.nn.softmax((raw_pred - threshold_pred_min + threshold_pred_max)).numpy()
        threshold_pred[threshold_pred_softmax < self.softmax_threshold] = threshold_pred_min
        for batch_index in range(raw_pred.shape[0]):
            for width_index in range(raw_pred.shape[1]):
                if np.max(threshold_pred[batch_index, width_index, :]) == threshold_pred_min:
                    threshold_pred[batch_index, width_index, self.blank_index] = threshold_blank_max
        threshold_pred_arg_max = np.argmax(threshold_pred, axis=-1)
        threshold_pred_arg_max_overlap_filtered = threshold_pred_arg_max[0][
            np.insert(~(threshold_pred_arg_max[0][:-1] == threshold_pred_arg_max[0][1:]), 0, True)]
        threshold_pred_softmax_overlap_filtered = np.max(threshold_pred_softmax[0][np.insert(
            ~(threshold_pred_arg_max[0][:-1] == threshold_pred_arg_max[0][1:]), 0, True)], axis=-1)
        text, label_indexes, score = self.__decode2labels(threshold_pred_arg_max_overlap_filtered,
                                                          threshold_pred_softmax_overlap_filtered)
        return text, label_indexes, score

    def __decode2labels(self, threshold_pred_arg_max_overlap_filtered, threshold_pred_softmax_overlap_filtered):
        labels = ""
        label_indexes = []
        scores = []
        for index, label_index in enumerate(threshold_pred_arg_max_overlap_filtered):
            if label_index == self.blank_index:
                continue
            labels += self.classes[label_index]
            label_indexes.append(label_index)
            scores.append(threshold_pred_softmax_overlap_filtered[index])
        return labels, label_indexes, np.prod(scores)
