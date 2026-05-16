#!/usr/bin/env python3
import threading
import os
from collections import namedtuple

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera, CameraMJPG, CameraV4L2NV12
from openpilot.common.realtime import Ratekeeper

WIDE_CAM = os.getenv("WIDE_CAM")
NO_DM = os.getenv("NO_DM") is not None
USE_V4L2_NV12 = os.getenv("USE_V4L2_NV12", "1") != "0"
CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])
CAMERAS = [
  CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, os.getenv("ROAD_CAM", "0")),
  # CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, os.getenv("DRIVER_CAM", "2")),
]
if not NO_DM:
  CAMERAS.append(CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, os.getenv("DRIVER_CAM", "2")))
if WIDE_CAM:
  CAMERAS.append(CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM))

class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
    self.vipc_server = VisionIpcServer("camerad")

    self.cameras = []
    for c in CAMERAS:
      cam_device = f"/dev/video{c.cam_id}"
      print(f"opening {c.msg_name} at {cam_device}")
      cam = self._create_camera(c.msg_name, c.stream_type, cam_device)
      self.cameras.append(cam)
      self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)

    self.vipc_server.start_listener()

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)
    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

  def _create_camera(self, msg_name, stream_type, cam_device):
    if USE_V4L2_NV12:
      try:
        return CameraV4L2NV12(msg_name, stream_type, cam_device)
      except Exception as e:
        print(f"[camera] V4L2 NV12 init failed for {cam_device}: {e}; falling back to OpenCV path")
    return CameraMJPG(msg_name, stream_type, cam_device)

  def camera_runner(self, cam):
    rk = Ratekeeper(20, None)
    while True:
      try:
        for yuv in cam.read_frames():
          self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type)
          cam.cur_frame_id += 1
          rk.keep_time()
        raise RuntimeError("camera stream ended")
      except Exception as e:
        print(f"[camera] stream error on {getattr(cam, 'device_path', 'unknown')}: {e}")
        if isinstance(cam, CameraV4L2NV12):
          try:
            print("[camera] falling back to OpenCV CameraMJPG")
            cam = CameraMJPG(cam.cam_type_state, cam.stream_type, cam.device_path)
            continue
          except Exception as fallback_err:
            print(f"[camera] OpenCV fallback failed: {fallback_err}")
        raise

  def run(self):
    threads = []
    for cam in self.cameras:
      cam_thread = threading.Thread(target=self.camera_runner, args=(cam,))
      cam_thread.start()
      threads.append(cam_thread)

    for t in threads:
      t.join()


def main():
  camerad = Camerad()
  camerad.run()


if __name__ == "__main__":
  main()
