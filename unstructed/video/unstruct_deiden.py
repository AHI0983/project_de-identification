import cv2
import face_recognition
import multiprocessing
def process_frame(frame_number, frame, return_dict):
 rgb_frame = frame[:, :, ::-1]
 face_locations = face_recognition.face_locations(rgb_frame
 for (top, right, bottom, left) in face_locations:
 face = frame[top:bottom, left:right]
 face = cv2.GaussianBlur(face, (99, 99), 30)
 frame[top:bottom, left:right] = face
 return_dict[frame_number] = (frame, face_locations)
def blur_face_in_video_parallel(input_video_path, output_video
 video_capture = cv2.VideoCapture(input_video_path)
 fps = int(video_capture.get(cv2.CAP_PROP_FPS))
 frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WID
 frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HE
 total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_CO
 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
 out = cv2.VideoWriter(output_video_path, fourcc, fps, (fra
 manager = multiprocessing.Manager()
 return_dict = manager.dict()
 jobs = []
미니프젝 3
 frame_number = 0
 batch = []
 while video_capture.isOpened():
 ret, frame = video_capture.read()
 if not ret:
 break
 batch.append((frame_number, frame))
 frame_number += 1
 if len(batch) == batch_size or frame_number == total_f
 for frame_number, frame in batch:
 process = multiprocessing.Process(target=proce
 jobs.append(process)
 process.start()
 for job in jobs:
 job.join()
 for frame_number, _ in batch:
 if frame_number in return_dict:
 out.write(return_dict[frame_number][0])
 batch = [] 
 jobs = [] 
 video_capture.release()
 out.release()
if __name__ == '__main__':
 input_video = r'input.mp4'
 output_video = r'output.mp4'
 blur_face_in_video_parallel(input_video, output_video)