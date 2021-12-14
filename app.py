
import logging
from pathlib import Path


import av
import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from google_drive_downloader import GoogleDriveDownloader as gdd
import os

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)



from pathlib import Path
import os
import cv2
import torch

import numpy as np
from models.common import DetectMultiBackend
from utils.general import  non_max_suppression, scale_coords, xyxy2xywh


class_label = ["nguoi", "xe dap", "o to", "xe may", "may bay", "xe buyt", "tau hoa", "xe tai", "thuyen", "den giao thong",
         "voi chua chay", "bien bao dung", "dong ho do xe", "bang ghe", "chim", "meo", "cho", "ngua", "cuu", "bo",
         "voi", "gau", "ngua van", "huou cao co", "ba lo", "o", "tui xach", "ca vat", "vali", "dia bay",
         "van truot", "van truot tuyet", "bong the thao", "dieu", "gay bong chay", "gang tay bong chay", "van truot", "van luot song",
         "vot tennis", "chai", "ly ruou", "coc", "nia", "dao", "muong", "bat", "chuoi", "tao",
         "sandwich", "cam", "sup lo xanh", "ca rot", "xuc xich", "pizza", "banh ran", "banh ngot", "ghe", "di vang",
         "chau cay", "giuong", "ban an", "toilet", "tv", "may tinh xach tay", "chuot", "dieu khien tu xa", "ban phim", "dien thoai di dong",
         "lo vi song", "lo nuong", "may nuong banh mi", "bon rua", "tu lanh", "sach", "dong ho", "binh hoa", "keo", "gau bong",
         'may say toc', 'ban chai danh rang']


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


COLORS = np.random.uniform(0, 255, size=(len(class_label), 3))

class OpenCVVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
       self.model = DetectMultiBackend('yolov5n.pt', device='cpu')
       self.model.warmup(imgsz=(1, 3, 640,640))
   
    def predict(self, img):
        img_org = img.copy()
        h,w,s = img_org.shape
        img = letterbox(img, new_shape = 640)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1) 
        image = img.astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, 0)

        image = torch.from_numpy(image)
        #image = torch.from_numpy(image).cuda()

        #print("shape tensor image:", image.shape)

        pred = self.model(image)
        # print("pred shape:", pred.shape)
        temp_img = None
        pred = non_max_suppression(pred, 0.5, 0.5,None)
        #print(pred[0])
        num_boxes = 0 
        for i, det in enumerate(pred):
                im0 = img_org
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                    
                    # Write results
                
                    for *xyxy, conf, cls in reversed(det):

                        bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        bbox_new = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)                    
                        
                        #line = (cls, *xywh, conf)  # label format
                        bbox_new[0] = bbox[0] - bbox[2] / 2  # top left x
                        bbox_new[1] = bbox[1] - bbox[3] / 2  # top left y
                        bbox_new[2] = bbox[0] + bbox[2] / 2  # bottom right x
                        bbox_new[3] = bbox[1] + bbox[3] / 2  # bottom right y

                        bbox_new[0] = bbox_new[0] * w
                        bbox_new[2] = bbox_new[2] * w
                        bbox_new[1] = bbox_new[1] * h
                        bbox_new[3] = bbox_new[3] * h
                        #print("class: ", labels[int(cls)])
                        #print("conf: ", float(conf))
                        
                        # display the prediction
                        name = class_label[int(cls)]
                        confidence = float(conf)
                        label = f"{name}: {round(confidence * 100, 2)}%"
                        startX =int(bbox_new[0])
                        startY = int(bbox_new[1])
                        endX = int(bbox_new[2])
                        endY = int(bbox_new[3])
                        cv2.rectangle(img_org, (startX, startY), (endX, endY), COLORS[int(cls)], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(
                            img_org,
                            label,
                            (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            COLORS[int(cls)],
                            2,
                        )
                        
                        num_boxes = num_boxes + 1
                        
        return img_org    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        #img = self.predict(img)
        #self.result_queue = queue.Queue()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if __name__ == "__main__":
  
    st.header("Phan Minh Toan @Real time detection ")
    if (not os.path.exists('./yolov5s.pt')):
        with st.spinner(text="Download model in progress..."):
            gdd.download_file_from_google_drive(file_id='1V5hUspqnI6uvBIPyccga9lsz8-fWFQ9p',
                                    dest_path='./yolov5n.pt')

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True)
    
