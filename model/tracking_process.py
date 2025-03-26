from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ffmpeg
import torch
from sort.sort import Sort
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv

def show_video(video_path):
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def process_video_sort(video_input, video_output):
    video = cv2.VideoCapture(video_input)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    mot_tracker = Sort()
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame)
        object_list = []
        for result in results:
            for box in result.boxes.data.cpu().numpy():
                left, top, right, bottom, confidence, cls = box
                object_list.append([left, top, right, bottom, confidence, int(cls)])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        np1 = np.array(object_list)
        trackers = mot_tracker.update(np1)

        for d in trackers:
            x1, y1, x2, y2, track_id  = d
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    video.release()
    out.release()
    print(f'Processed video saved to {video_output}')

def process_video_deepsort(video_input, video_output, deepsort_max_age):
    tracker = DeepSort(max_age = deepsort_max_age)
    cap = cv2.VideoCapture(video_input)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        bbs = []
        for result in results:
            for box in result.boxes.data.cpu().numpy():
                left, top, right, bottom, confidence, cls = box
                w, h = right - left, bottom - top
                bbs.append(([left, top, w, h], confidence, int(cls)))

        tracks = tracker.update_tracks(bbs, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            l, t, r, b = map(int, ltrb)
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f'Process video saved to {video_output}')

def process_video_bytetrack(video_input, video_output, frame_rate = 25, class_idx = list(range(1,81))):
    tracker = sv.ByteTrack(frame_rate= frame_rate)
    genrator = sv.get_video_frames_generator(video_input)
    frame = next(genrator)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    def callback_bytetrack(frame, index) -> np.ndarray:
        result = model(frame, verbose=False)[0]
        detect = sv.Detections.from_ultralytics(result)
        detect = detect[np.isin(detect.class_id, class_idx)]
        detect = tracker.update_with_detections(detect)

        label = [f'{tracker_id} {model.model.names[class_id]} {confidence: 0.2f} '
                 for tracker_id, class_id, confidence
                 in zip(detect.tracker_id, detect.class_id, detect.confidence)]

        annotator = frame.copy()
        annotator = box_annotator.annotate(scene=frame, detections=detect)
        annotator = trace_annotator.annotate(scene=frame, detections=detect)
        annotator = label_annotator.annotate(scene=frame, labels=label, detections=detect)
        return annotator

    sv.process_video(source_path=video_input, target_path=video_output, callback=callback_bytetrack)
    print(f'Process video saved to {video_output}')

if __name__ == '__main__':
    video_path = 'C:/Users/DELL/PycharmProjects/DL/ObjectTracking/cars.mp4'
    output_file = 'output_sort.mp4'
    device = 'cuda'
    model = YOLO('yolov10m.pt')
    model.to(device)
    # show_video('C:/Users/DELL/PycharmProjects/DL/ObjectTracking/cars.mp4')
    process_video_sort(video_path, output_file)
    show_video(output_file)