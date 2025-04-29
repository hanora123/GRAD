import argparse
import os
import sys
from pathlib import Path
import json

import cv2
import torch
import platform  # Add platform import for Linux window handling
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Add the parent directory to sys.path to access TDNet
PARENT = ROOT.parent  # Parent directory containing TDNet
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                          colorstr, increment_path, non_max_suppression, print_args,
                          scale_boxes, strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.track_predict import TrackPredictor

# Import SORT tracker from TDNet using absolute import
from TDNet.Trakers import SORT

@smart_inference_mode()
def run(weights='yolov5s.pt',  # model path
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        history_size=30,  # number of frames to keep in trajectory history
        future_steps=10,  # number of steps to predict into future
        custom_output=None,  # custom output path for video
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories setup - TDNet style
    root = os.path.dirname(os.path.abspath(source))
    filename = os.path.basename(source).split('.')[0]
    path = os.path.join(root, filename)
    
    # Create output directories
    if custom_output:
        output_base = custom_output
    else:
        output_base = os.path.join(project, name)
    
    video_dir = os.path.join(output_base, 'Video')
    figure_dir = os.path.join(output_base, 'Figure')
    
    # Create directories if they don't exist
    os.makedirs(path, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    LOGGER.info(f': Video Folder Created: {video_dir}')
    LOGGER.info(f': Figure Folder Created: {figure_dir}')
    
    # Create configuration file
    config = {
        'General': {
            'Speed Limitation': 30,
            'Speed Unit': 'mph',
            'Real Size (cm)': {
                'person': (70),
                'car': (470, 190),
                'truck': (470, 190),
                'bus': (1195, 255),
                'motorcycle': (190, 70),
                'bicycle': (180, 50)
            }
        },
        'Visualizer': {
            '1/0': 1,
            '3D Detection': {'1/0': 1, 'Show': 0, 'Save': 0, 'Rithm': 100, 'Video': 1},
            'CONFIGS': {
                'Show Caption': 1,
                'Show Speed': 0,
                'Speed Unit Text': ' mph',
                'Speed Text Color': (0, 0, 0),
                'vSpeed Text Color': (0, 0, 200),
                'person': {
                    '3D': {'color': (250, 30, 30), 'tcolor': (250, 50, 50), 'bcolor': (255, 100, 100), 'size': (8, 8)}
                },
                'car': {
                    '3D': {'color': (0, 150, 0), 'tcolor': (0, 220, 0), 'bcolor': (0, 200, 0), 'height_coef': 0.6, 'direction': False}
                },
                'truck': {
                    '3D': {'color': (0, 150, 0), 'tcolor': (0, 220, 0), 'bcolor': (0, 200, 0), 'height_coef': 0.65, 'direction': False}
                },
                'bus': {
                    '3D': {'color': (30, 90, 160), 'tcolor': (50, 120, 200), 'bcolor': (15, 100, 240), 'height_coef': 0.7, 'direction': True}
                },
                'motorcycle': {
                    '3D': {'color': (160, 160, 34), 'tcolor': (160, 180, 54), 'bcolor': (0, 200, 0), 'height_coef': 0.6, 'direction': False}
                },
                'bicycle': {
                    '3D': {'color': (255, 130, 90), 'tcolor': (255, 150, 20), 'bcolor': (254, 232, 125), 'height_coef': 0.7, 'direction': False}
                }
            }
        },
        'System': {
            'Video Format': 'DIVX'  # Same as TDNet
        }
    }
    
    # Save configuration
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Initialize trajectory predictor with real sizes from config
    track_predictor = TrackPredictor(
        history_size=history_size, 
        future_steps=future_steps,
        real_sizes=config['General']['Real Size (cm)']
    )
    
    # Initialize SORT tracker with parameters from TDNet configuration
    sort_tracker = SORT(
        max_age=30,     # Maximum frames to keep object without detection
        min_hits=3,     # Minimum detection hits to start tracking
        iou_threshold=0.2  # IOU threshold for matching
    )

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(Path(output_base) / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(Path(video_dir) / p.name)  # im.jpg
            txt_path = str(Path(output_base) / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Convert detections to format expected by SORT
                # Format: [x1, y1, x2, y2, confidence, class]
                tracked_dets = sort_tracker.update(det.cpu().numpy())
                
                # Process tracked detections and predict trajectories with 3D boxes
                # Create a new tensor with the same structure as det but with tracked IDs
                if len(tracked_dets) > 0:
                    # Convert tracked detections back to tensor format for processing
                    tracked_tensor = torch.zeros((len(tracked_dets), 6))
                    for j, trk in enumerate(tracked_dets):
                        # Format: [x1, y1, x2, y2, ID, class]
                        tracked_tensor[j, 0:4] = torch.tensor(trk[0:4])  # bbox
                        tracked_tensor[j, 4] = torch.tensor(det[j, 4])   # confidence
                        tracked_tensor[j, 5] = torch.tensor(det[j, 5])   # class
                        
                    # Process with consistent IDs from tracker
                    im0 = track_predictor.process_frame(tracked_tensor, im0, names, tracked_ids=tracked_dets[:, 4])
                else:
                    # If no tracked detections, process original detections
                    im0 = track_predictor.process_frame(det, im0, names)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Stream results
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        # Use DIVX codec as in TDNet
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*config['System']['Video Format']), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(Path(output_base).glob('labels/*.txt')))} labels saved to {Path(output_base) / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', Path(output_base))}{s}")
    if update:
        strip_optimizer(weights)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--history-size', type=int, default=30, help='number of frames to keep in trajectory history')
    parser.add_argument('--future-steps', type=int, default=10, help='number of steps to predict into future')
    parser.add_argument('--custom-output', type=str, default=None, help='custom output path for video')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)