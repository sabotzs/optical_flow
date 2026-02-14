from pathlib import Path

import cv2 as cv


def save_frames(file: Path, frames: list[int], destination: Path) -> None:
  destination.mkdir(parents=True, exist_ok=True)
  base_name = file.stem

  cap = cv.VideoCapture(str(file))
  current_frame = 0
  last = max(frames)
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print(f"Can't get frame {current_frame}. Exiting ...")
      break

    if current_frame in frames:
      output_path = destination / f'{base_name}_{current_frame}.jpg'
      cv.imwrite(str(output_path), frame)

    if current_frame == last:
      break
    current_frame += 1
  cap.release()
