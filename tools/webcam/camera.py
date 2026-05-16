import errno
import fcntl
import mmap
import os
import select
import time
import ctypes

import av
import cv2


# v4l2 constants
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1
V4L2_PIX_FMT_NV12 = 0x3231564E
V4L2_CAP_VIDEO_CAPTURE = 0x00000001

# Note: CameraV4L2NV12 supports reconnect() for robust camera recovery.
# Example usage:
#   try:
#     ... # normal frame read
#   except RuntimeError as e:
#     if 'closed' in str(e):
#       cam.reconnect()
#
# Device path check for /dev/rknpu is now broadened in system/inferenced/compute.py


def _fourcc_to_str(v: int) -> str:
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


# ioctl helpers from asm-generic/ioctl.h
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(direction, ioc_type, nr, size):
    return (direction << _IOC_DIRSHIFT) | (ord(ioc_type) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)


def _IOR(ioc_type, nr, struct_type):
    return _IOC(_IOC_READ, ioc_type, nr, ctypes.sizeof(struct_type))


def _IOW(ioc_type, nr, struct_type):
    return _IOC(_IOC_WRITE, ioc_type, nr, ctypes.sizeof(struct_type))


def _IOWR(ioc_type, nr, struct_type):
    return _IOC(_IOC_READ | _IOC_WRITE, ioc_type, nr, ctypes.sizeof(struct_type))


class _timeval(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_usec", ctypes.c_long),
    ]


class _v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _v4l2_capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_uint8 * 16),
        ("card", ctypes.c_uint8 * 32),
        ("bus_info", ctypes.c_uint8 * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _v4l2_format_union(ctypes.Union):
    _fields_ = [
        ("pix", _v4l2_pix_format),
        ("raw", ctypes.c_uint8 * 200),
    ]


class _v4l2_format(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("fmt", _v4l2_format_union),
    ]


class _v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class _v4l2_buffer_m(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
    ]


class _v4l2_buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _timeval),
        ("timecode", _v4l2_timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _v4l2_buffer_m),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


VIDIOC_QUERYCAP = _IOR("V", 0, _v4l2_capability)
VIDIOC_S_FMT = _IOWR("V", 5, _v4l2_format)
VIDIOC_G_FMT = _IOWR("V", 4, _v4l2_format)
VIDIOC_REQBUFS = _IOWR("V", 8, _v4l2_requestbuffers)
VIDIOC_QUERYBUF = _IOWR("V", 9, _v4l2_buffer)
VIDIOC_QBUF = _IOWR("V", 15, _v4l2_buffer)
VIDIOC_DQBUF = _IOWR("V", 17, _v4l2_buffer)
VIDIOC_STREAMON = _IOW("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW("V", 19, ctypes.c_int)


def _xioctl(fd, req, arg):
    while True:
        try:
            return fcntl.ioctl(fd, req, arg)
        except OSError as e:
            if e.errno == errno.EINTR:
                continue
            raise

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    self.container = av.open(camera_id)
    assert self.container.streams.video, f"Can't open video stream for camera {camera_id}"
    self.video_stream = self.container.streams.video[0]
    self.W = self.video_stream.codec_context.width
    self.H = self.video_stream.codec_context.height

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    for frame in self.container.decode(self.video_stream):
      img = frame.to_rgb().to_ndarray()[:,:, ::-1] # convert to bgr24
      yuv = Camera.bgr2nv12(img)
      yield yuv.data.tobytes()
    self.container.close()

class CameraMJPG:
    def __init__(self, cam_type_state, stream_type, camera_id):
        try:
            camera_id = int(camera_id)
        except ValueError:
            pass

        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError(f"无法打开摄像头设备 {camera_id}")

        # 优先尝试设置 MJPG 格式（高分辨率 + 高帧率）
        self._configure_camera_format("MJPG")
        actual_format = self._get_current_format()
        print("数据格式: ", actual_format)
        # 若 MJPG 不支持，回退默认

        # 获取实际设置的FPS并打印
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"摄像头初始化后的FPS设置: {self.fps}")


        # 获取分辨率
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cur_frame_id = 0
        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.current_format = actual_format  # 记录当前格式用于后续处理

    def _configure_camera_format(self, target_fourcc):
        """尝试设置摄像头的FourCC格式"""
        fourcc = cv2.VideoWriter_fourcc(*target_fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 优先选择最高分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 20)


    def _get_current_format(self):
        """获取当前实际格式"""
        fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return ''.join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])

    @staticmethod
    def _bgr_to_nv12(bgr_frame):
        frame = av.VideoFrame.from_ndarray(bgr_frame, format='bgr24')
        return frame.reformat(format='nv12').to_ndarray().data.tobytes()

    def read_frames(self, reconnect_retries=5, reconnect_delay=0.5):
        """持续读取帧并转换为 NV12，断开时自动重连"""
        retries = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Try to reconnect a few times before giving up
                self.cap.release()
                while retries < reconnect_retries:
                    time.sleep(reconnect_delay)
                    self.cap = cv2.VideoCapture(self.camera_id)
                    if self.cap.isOpened():
                        self._configure_camera_format("MJPG")
                        self.current_format = self._get_current_format()
                        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[CameraMJPG] Reconnected camera after {retries+1} attempt(s)")
                        retries = 0
                        break
                    retries += 1
                else:
                    print("[CameraMJPG] Failed to reconnect camera after multiple attempts, exiting.")
                    return
                continue
            retries = 0
            if self.current_format == "MJPG":
                if frame.shape != (self.H, self.W, 3):
                    raise ValueError("MJPG 解码后帧形状异常，请检查摄像头设置")
                yield self._bgr_to_nv12(frame)
            else:
                yield self._bgr_to_nv12(frame)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


class CameraV4L2NV12:
    def __init__(self, cam_type_state, stream_type, camera_id, width=1280, height=720, fps=20, buf_count=4):
        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.cur_frame_id = 0
        self.camera_id = str(camera_id)
        self._fd = None
        self._maps = []
        self._buf_count = buf_count
        self._stream_on = False
        self._fps = fps

        if isinstance(camera_id, int):
            dev = f"/dev/video{camera_id}"
        else:
            dev = str(camera_id)
            if not dev.startswith("/dev/"):
                dev = f"/dev/video{dev}"
        self.device_path = dev

        self._open_and_configure(width, height)

    def _open_and_configure(self, width, height):
        self._close()
        self._fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)

        cap = _v4l2_capability()
        _xioctl(self._fd, VIDIOC_QUERYCAP, cap)
        if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0:
            raise RuntimeError(f"{self.device_path} is not a video capture device")

        fmt = _v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        _xioctl(self._fd, VIDIOC_S_FMT, fmt)

        fmt2 = _v4l2_format()
        fmt2.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        _xioctl(self._fd, VIDIOC_G_FMT, fmt2)

        actual_fmt = fmt2.fmt.pix.pixelformat
        if actual_fmt != V4L2_PIX_FMT_NV12:
            raise RuntimeError(f"{self.device_path} returned { _fourcc_to_str(actual_fmt) }, expected NV12")

        self.W = int(fmt2.fmt.pix.width)
        self.H = int(fmt2.fmt.pix.height)

        req = _v4l2_requestbuffers()
        req.count = self._buf_count
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        _xioctl(self._fd, VIDIOC_REQBUFS, req)
        if req.count < 2:
            raise RuntimeError(f"Insufficient V4L2 buffers: {req.count}")

        self._maps = []
        for i in range(req.count):
            buf = _v4l2_buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = i
            _xioctl(self._fd, VIDIOC_QUERYBUF, buf)

            mm = mmap.mmap(self._fd, buf.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset)
            self._maps.append(mm)

            _xioctl(self._fd, VIDIOC_QBUF, buf)

        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        _xioctl(self._fd, VIDIOC_STREAMON, buf_type)
        self._stream_on = True
        print(f"V4L2 NV12 enabled on {self.device_path} ({self.W}x{self.H})")

    def _close(self):
        if self._fd is not None:
            try:
                if self._stream_on:
                    buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                    _xioctl(self._fd, VIDIOC_STREAMOFF, buf_type)
            except Exception:
                pass
            self._stream_on = False

            for mm in self._maps:
                try:
                    mm.close()
                except Exception:
                    pass
            self._maps = []

            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def read_frames(self):
        while True:
            if self._fd is None:
                raise RuntimeError("V4L2 device closed")

            ready, _, _ = select.select([self._fd], [], [], 1.0)
            if not ready:
                continue

            buf = _v4l2_buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            try:
                _xioctl(self._fd, VIDIOC_DQBUF, buf)
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EINTR):
                    continue
                raise

            data = self._maps[buf.index][:buf.bytesused]
            _xioctl(self._fd, VIDIOC_QBUF, buf)
            yield bytes(data)

    def reconnect(self, retries=5, delay=0.4):
        last_err = None
        for _ in range(retries):
            try:
                self._open_and_configure(self.W, self.H)
                return
            except Exception as e:
                last_err = e
                time.sleep(delay)
        raise RuntimeError(f"V4L2 reconnect failed: {last_err}")

    def __del__(self):
        self._close()