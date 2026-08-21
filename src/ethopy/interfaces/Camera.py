import base64
import io
import logging
import multiprocessing as mp
import os
import shutil
import threading
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Pool
from pathlib import Path
from queue import Full, Queue
from threading import Condition, Lock, Thread
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ethopy import local_conf
from ethopy.utils.timer import Timer

log = logging.getLogger(__name__)

# Libraries that only required in specific classes
try:
    from skvideo.io import FFmpegWriter

    IMPORT_SKVIDEO = True
except ImportError:
    IMPORT_SKVIDEO = False

try:
    from ethopy.core.logger import Logger
except ImportError:
    log.warning("Logger not found.")

try:
    import cv2

    IMPORT_CV2 = True
except ImportError:
    IMPORT_CV2 = False

try:
    from picamera2 import MappedArray, Picamera2
    from picamera2.encoders import H264Encoder, MJPEGEncoder
    from picamera2.outputs import FfmpegOutput, FileOutput

    IMPORT_PICAMERA = True
except ImportError:
    IMPORT_PICAMERA = False


class Camera(ABC):
    """
    A class to manage a camera.

    This class provides methods to initialize, start, stop, and record from a camera.
    It also provides methods to manage the recording process, such as setting up a frame
    queue and writing frames to it.

    Attributes:
        filename (str, optional): The name of the file.
        initialized (threading.Event): An event to indicate whether the camera is initialized.
        recording (mp.Event): An event to indicate whether the camera is recording.
        stop (mp.Event): An event to indicate whether the camera should stop recording.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        logger: Optional["Logger"] = None,
        video_aim: Optional[str] = None,
    ):
        self.recording = mp.Event()
        self.recording.clear()
        self.filename = (
            filename
            if filename is not None
            else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        if logger:
            # Co-locate the video with the timestamp/DLC h5 in the session
            # Recordings folder (the path Logger.createDataset also uses).
            recordings_folder = (
                f"Recordings/{logger.trial_key['animal_id']}"
                f"_{logger.trial_key['session']}/"
            )
            self.source_path = logger.source_path + recordings_folder
            self.target_path = (
                logger.target_path + recordings_folder
                if os.path.isdir(logger.target_path)
                else self.source_path
            )
        else:
            self.source_path = local_conf.get("source_path", "") + f"{self.filename}/"
            self.target_path = local_conf.get("target_path", "") + f"{self.filename}/"

        self.serve_port = local_conf.get("server.port", 0)
        if self.serve_port:
            self.server_user = local_conf.get("server.user", "")
            self.server_password = local_conf.get("server.password", "")
            # Frames per second to stream; 0 streams every encoded frame.
            self.serve_fps = local_conf.get("server.fps", 0)
        self.httpthread = None
        self.tmst_type = None
        self.dataset = None

        self.post_process = mp.Event()
        self.post_process.clear()

        self.process_queue = mp.Queue(maxsize=30)
        self.process_queue.cancel_join_thread()

        self.stop = mp.Event()
        self.stop.clear()

        self._cam = None
        self.logger = logger

        self.frame_queue = None
        self.capture_runner = None
        self.write_runner = None

        if logger:
            # log video recording
            logger.log_recording(
                dict(
                    rec_aim=video_aim,
                    software="EthoPy",
                    version="0.1",
                    filename=self.filename + ".mp4",
                    source_path=self.source_path,
                    target_path=self.target_path,
                ),
                block=True,
            )
            # Per-camera name so two cameras in a session don't overwrite each
            # other's h5; the file is written later by Logger.createDataset.
            self.filename_tmst = f"videotmst_{self.filename}.h5"
            h5_target_path = (
                logger.target_path + recordings_folder
                if os.path.isdir(logger.target_path)
                else False
            )
            logger.log_recording(
                dict(
                    rec_aim="sync",
                    software="EthoPy",
                    version="0.1",
                    filename=self.filename_tmst,
                    source_path=self.source_path,
                    target_path=h5_target_path,
                ),
                block=True,
            )

        self.camera_process = mp.Process(target=self.start_rec)
        self.camera_process.start()

    @property
    def source_path(self) -> str:
        """
        Get the source path.

        Returns:
            str: The source path.
        """
        return self._source_path

    @source_path.setter
    def source_path(self, source_path: str):
        """
        Set the source path. If the path does not exist, create it.

        Args:
            source_path (str): The source path.
        """
        self._source_path = self._create_and_set_path(source_path)

    @property
    def target_path(self) -> str:
        """
        Get the target path.

        Returns:
            str: The target path.
        """
        return self._target_path

    @target_path.setter
    def target_path(self, target_path: str):
        """
        Set the target path. If the path does not exist, create it.

        Args:
            target_path (str): The target path.
        """
        self._target_path = self._create_and_set_path(target_path)

    def _create_and_set_path(self, path: str) -> str:
        """
        Create the path if it does not exist and return the path.

        Args:
            path (str): The path.

        Returns:
            str: The path.
        """
        if not self.recording.is_set():
            os.makedirs(path, exist_ok=True)

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")

        return path

    @staticmethod
    def copy_file(args) -> bool:
        """
        Copy a file from the source path to the target path.

        Args:
            args (tuple): A tuple containing the source file path and the target directory path.

        Returns:
            bool: True if the file was copied, verified and removed locally. On
            False the local copy is kept, so the recording is never lost.

        """
        file, target = args
        destination = target / file.name
        try:
            shutil.copy(str(file), str(destination))
            log.debug(f"Transferred file: {file.name}")
            # Verify the copy before deleting the only other copy of the data
            if (
                not destination.exists()
                or destination.stat().st_size != file.stat().st_size
            ):
                log.error(
                    f"Size mismatch after transferring {file.name}; "
                    "keeping the local copy"
                )
                return False
            os.remove(str(file))
            log.debug(f"Deleted original file: {file.name}")
            return True
        except OSError as ex:
            # OSError also covers shutil.SameFileError and a dropped network mount
            log.error(f"Failed to transfer file: {file.name}. Reason: {ex}")
            return False

    def clear_local_videos(self) -> None:
        """Move this camera's video file(s) to the target path.

        The source folder is shared with the timestamp/DLC h5 files (owned by the
        Writer) and other cameras, so only this camera's own files are moved
        (matched by filename, excluding .h5) and the folder is left in place.
        """
        source = Path(self.source_path)
        target = Path(self.target_path)

        if source == target or not target.is_dir():
            return  # autocopy disabled; leave the video alongside the h5 files

        files = [
            (entry, target)
            for entry in source.iterdir()
            if entry.is_file()
            and self.filename in entry.name
            and entry.suffix.lower() != ".h5"
        ]
        if not files:
            log.warning("No video files found to transfer")
            return

        log.info(f"Transferring {len(files)} video file(s) from {source} to {target}")
        with Pool(processes=min(2, os.cpu_count() - 1)) as pool:
            results = pool.map(self.copy_file, files)

        failed = [entry.name for (entry, _), ok in zip(files, results) if not ok]
        if failed:
            log.error(
                f"Failed to transfer {len(failed)} of {len(files)} video file(s): "
                f"{', '.join(failed)}. They are kept in {source}"
            )

    def setup(self) -> None:
        """
        Set up the frame queue and the capture and write runners.
        """
        self.frame_queue = Queue()
        # self.process_queue.cancel_join_thread()
        self.capture_runner = threading.Thread(
            target=self._run_guarded, args=(self.rec,)
        )
        self.write_runner = threading.Thread(
            target=self._run_guarded, args=(self.dequeue, self.frame_queue)
        )

    def _run_guarded(self, func: Any, *args: Any) -> None:
        """Run a recording thread target, logging whatever it raises.

        An unhandled exception in a thread only reaches threading.excepthook,
        so it never lands in the ethopy log, and self.stop stays clear - which
        leaves dequeue() spinning and the whole camera subprocess alive with a
        closed camera (and still holding the streaming port).
        """
        try:
            func(*args)
        except Exception:
            log.exception(
                "Camera %s: %s failed, stopping recording.",
                self.filename,
                getattr(func, "__name__", func),
            )
        finally:
            self.stop.set()

    def start_rec(self) -> None:
        """
        Start the capture and write runners with exception handling.
        """
        try:
            self.setup()
            self.capture_runner.start()
            self.write_runner.start()
            self.capture_runner.join()
            self.write_runner.join()
        except Exception as cam_error:
            log.exception("Camera %s: recording setup failed.", self.filename)
            raise RuntimeError(
                f"Exception occurred during recording: {cam_error}"
            ) from cam_error

    def dequeue(self, frame_queue: Queue) -> None:
        """
        Dequeue frames from the frame queue and write them until the stop event is set.

        Args:
            frame_queue (Queue): The frame queue to dequeue frames from.
        """
        while not self.stop.is_set() or not frame_queue.empty():
            if not frame_queue.empty():
                self.write_frame(frame_queue.get())
            else:
                time.sleep(0.01)

    def stop_rec(self) -> None:
        """Stop the camera subprocess. Idempotent and safe to call before the
        camera has finished starting up (the stop event alone signals shutdown).
        """
        if self.camera_process is None:
            return
        self.stop.set()
        time.sleep(3)
        self.camera_process.join(timeout=30)
        if self.camera_process.is_alive():
            self.camera_process.terminate()
            self.camera_process.join(timeout=5)
        try:
            self.camera_process.close()
        except ValueError:
            pass  # still alive after terminate(); the OS reaps it once it exits
        self.camera_process = None

    @abstractmethod
    def rec(self) -> None:
        """
        Record frames. This method should be implemented by subclasses.
        """

    @abstractmethod
    def write_frame(self, item: Any) -> None:
        """
        Write a frame. This method should be implemented by subclasses.

        Args:
            item (Any): The frame to write.
        """


class WebCam(Camera):
    """
    A class representing a webcam for capturing video frames.

    Args:
        Camera (class): The parent class for capturing and recording video frames.

    Attributes:
        fps (int): Frames per second for recording.
        recording (bool): Flag indicating whether recording is active.
        camera (cv2.VideoCapture): OpenCV VideoCapture instance for accessing the webcam.

    Raises:
        RuntimeError: If there is no available camera.

    """

    def __init__(
        self,
        resolution_x: int = 1280,
        resolution_y: int = 720,
        fps: int = 30,
        camera_num: int = 0,
        logger_timer: Optional["Timer"] = None,
        **kwargs,
    ):
        """
        Initializes a WebCam instance.

        Args:
            resolution (Tuple[int, int], optional): Resolution of the webcam.
            Defaults to (640, 480).
            camera_num (int): /dev/videoN index used by V4L2. Defaults to 0.
            Used only when ``device_id`` (kwarg) is empty.

        Keyword Args:
            device_id (str): Stable hardware identifier for the camera. Either a
            full path to a ``/dev/v4l/by-id/...`` symlink, or a substring of one
            (e.g. a serial like "20231205_0001"). When set, it takes precedence
            over ``camera_num`` and survives reboots / USB re-plugging. Empty
            (default) falls back to the ``camera_num`` index.

        Raises:
            ImportError: If the cv2 package is not installed.
            RuntimeError: If there is no available camera.

        """
        self.fps = fps
        self.camera_num = camera_num
        self.video_output = None
        self.dataset = None
        self.tmst_output = None
        self.logger_timer = logger_timer
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.res_set: bool = True

        # Initialize optional camera parameters
        self.exposure = kwargs.get("exposure")
        self.wb_temperature = kwargs.get("wb_temperature")
        self.saturation = kwargs.get("saturation")
        self.gain = kwargs.get("gain")
        self.contrast = kwargs.get("contrast")
        self.brightness = kwargs.get("brightness")
        self.device_id = kwargs.get("device_id") or ""
        self._last_frame_err_log = 0.0  # throttles the per-frame read-error log

        if not globals()["IMPORT_CV2"]:
            raise ImportError(
                "The cv2 package could not be imported. "
                "Please install it before using WebCam.\n"
                "You can install cv2 using pip:\n"
                'sudo pip3 install opencv-python"'
            )
        # Probe in the parent (stat only, no open() — opening here races the
        # child's open in recording_init). self.device is inherited by the fork.
        self.device = self._resolve_device()
        super().__init__(kwargs["filename"], kwargs["logger"], kwargs["video_aim"])

    def _resolve_device(self) -> Union[int, str]:
        """Resolve the camera to a target cv2.VideoCapture can open.

        With no ``device_id`` this is the ``/dev/videoN`` index. Otherwise it is
        an existing path (e.g. a ``/dev/v4l/by-id`` symlink), or a substring
        matched against the ``index0`` symlinks under ``/dev/v4l/by-id`` — these
        are keyed on vendor/model/serial, so they survive reboots and re-plugging.
        """
        if not self.device_id:
            device_path = f"/dev/video{self.camera_num}"
            if not os.path.exists(device_path):
                raise RuntimeError(
                    f"Camera device {device_path} not found; check the camera is "
                    "connected and camera_idx matches the intended /dev/videoN."
                )
            return self.camera_num

        if os.path.exists(self.device_id):
            return self.device_id

        by_id = "/dev/v4l/by-id"
        available = sorted(os.listdir(by_id)) if os.path.isdir(by_id) else []
        matches = [
            os.path.join(by_id, name)
            for name in available
            if self.device_id in name and name.endswith("index0")
        ]
        # Require exactly one: 0 means not found, 2+ means the substring is
        # ambiguous and picking one would open an arbitrary camera.
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            f"device_id {self.device_id!r} matched {len(matches)} device(s) "
            f"under {by_id} (expected 1). Available: {available}"
        )

    def setup(self):
        """Setup the camera."""
        out_vid_fn = self.source_path + self.filename + ".mp4"
        self.video_output = FFmpegWriter(
            out_vid_fn,
            inputdict={
                "-r": str(self.fps),
            },
            outputdict={
                "-vcodec": "libx264",
                "-pix_fmt": "rgb24",  # Change to rgb24 or another format
                "-r": str(self.fps),
                "-preset": "ultrafast",
                "-s": f"{self.resolution_x}x{self.resolution_y}",
            },
        )
        if self.logger is not None:
            self.tmst_type = "h5"
            self.dataset = self.logger.createDataset(
                dataset_name="frame_tmst",
                dataset_type=np.dtype([("timestamp", np.double)]),
                filename=self.filename_tmst,
                db_log=False,
            )
        else:
            self.tmst_type = "txt"
            self.tmst_output = io.open(
                os.path.join(self.source_path, f"tmst_{self.filename}.txt"),
                "w",
                encoding="utf-8",
            )
        super().setup()

    def set_resolution(self, width, height):
        """set the resolution of the webcamera if it is possible
        However, the efficiency of changing the resolution may depend on the camera and
        the OpenCV backend being used. In some cases, changing the resolution may involve
        renegotiating the camera settings, and the efficiency could vary across different
        camera models and platforms.

        It's recommended to test and profile the performance with your specific camera to
        ensure that changing the resolution meets your performance requirements. If efficiency
        is a critical factor, you might want to consider using the camera's native resolution
        whenever possible.

        Args:
            width (int): width of frame
            height (int): height of frame
        """

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        check, image = self.get_frame()
        if not check:
            log.error("Failed to capture frame while setting resolution")
            return False
        log.info(f"image shape set resolution {image.shape}")
        return (image.shape[1], image.shape[0]) == (width, height)

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Capture a frame from the webcam.

        Returns:
            Tuple[bool, np.ndarray]: A tuple indicating success and the captured frame.
        """
        check, image = self.camera.read()
        if check:
            # If the capture was successful, convert the image to grayscale
            image = np.squeeze(np.mean(image, axis=2)).astype(np.uint8)
        return check, image

    def write_frame(self, item: Tuple[float, np.ndarray]) -> None:
        """
        Write a video frame to the output stream and update the timestamp dataset.

        Args:
            item (Tuple[float, np.ndarray]): A tuple containing the timestamp and the image frame.
        """
        img = item[1].copy()
        self.video_output.writeFrame(img)
        # Record the timestamp: h5 dataset with a logger, plain text file without.
        if self.tmst_type == "txt":
            self.tmst_output.write(f"{item[0]}\n")
        else:
            self.dataset.append("frame_tmst", [np.double(item[0])])

    def camera_opened(self, camera):
        """Check if the camera is opened."""
        if not camera.isOpened():
            raise RuntimeError("Camera is not opened. Cannot proceed.")
        return True

    def recording_init(self):
        self.camera = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError(
                "No camera is available. Please check if the camera is connected and functional."
            )
        # YUYV decoded to 3-channel RGB (get_frame averages over axis=2), and
        # BUFFERSIZE=1 so read() returns the latest frame, not a stale buffered one.
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
        self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        self.res_set = self.set_resolution(self.resolution_x, self.resolution_y)
        if not self.res_set:
            logging.warning(
                f"Camera resolution cannot be set to {(self.resolution_x, self.resolution_y)}"
                f", resize of frames will be used!!"
            )
        # Properties below are opt-in: omit the key from a camera's config (e.g. an
        # analog grabber) to leave the value None and skip the setter.
        if self.exposure:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Disable auto exposure
            self._set_camera_property(cv2.CAP_PROP_EXPOSURE, self.exposure)
        if self.wb_temperature:
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 0.0)  # Disable auto white balance
            self._set_camera_property(cv2.CAP_PROP_WB_TEMPERATURE, self.wb_temperature)
        self._set_camera_property(cv2.CAP_PROP_SATURATION, self.saturation)
        self._set_camera_property(cv2.CAP_PROP_GAIN, self.gain)
        self._set_camera_property(cv2.CAP_PROP_CONTRAST, self.contrast)
        self._set_camera_property(cv2.CAP_PROP_BRIGHTNESS, self.brightness)

    def _set_camera_property(self, property_id, value):
        if value is not None:
            result = self.camera.set(property_id, value)
            if result:
                actual_value = self.camera.get(property_id)
                if (
                    abs(actual_value - value) > 1e-6
                ):  # Compare with small tolerance for floating-point values
                    logging.warning(
                        f"Camera property {property_id} was set to "
                        f"{actual_value}, not the requested {value}"
                    )
            else:
                # set() returned False: the camera doesn't expose this property.
                logging.warning(
                    f"Camera property {property_id} is not supported by this "
                    f"camera; requested value {value} was ignored"
                )

    def rec(self):
        """
        Continuously capture video frames, update timestamp, and enqueue frames for processing.

        The method runs in a loop until the 'stop' event is set. It captures a frame from
        the webcam,records the elapsed time, increments the frame counter, and puts the
        timestamped frame into the 'frame_queue'. If a separate processing queue
        ('process_queue') is provided, the frame is also put into that queue, ensuring it
        doesn't exceed its maximum size. We need for the process_queue(size:2) the latest image
        so if it is full get a frame and put the latest one.
        """
        self.recording_init()
        self.recording.set()
        while not self.stop.is_set() and self.camera_opened(self.camera):
            try:
                check, image = self.get_frame()
                if not check:
                    continue
                # Process the frame here
            except RuntimeError as error:
                now = time.time()
                if now - self._last_frame_err_log >= 1.0:
                    self._last_frame_err_log = now
                    log.error(f"Failed to read frame from camera. Error: {error}")
                continue
            tmst = self.logger_timer.elapsed_time()
            if not self.res_set:
                image = cv2.resize(image, (self.resolution_x, self.resolution_y))
            self.frame_queue.put((tmst, image))
            # Check if a separate process queue is provided
            if self.process_queue is not False:
                # Ensure the process queue doesn't exceed its maximum size
                try:
                    self.process_queue.put_nowait((tmst, image))
                except Full:
                    pass

        self.camera.release()
        self.recording.clear()
        if self.tmst_type == "txt":
            self.tmst_output.close()
        else:
            self.dataset.exit()

    def stop_rec(self):
        """
        Stop video recording and release resources.

        If video recording is in progress, the method releases the camera resources,
        closes the video output stream, clears the recording flag, and performs cleanup
        by removing local video files.
        """
        # TODO: check the stop_rec function and define a function release to be called by the process
        # if self.recording.is_set():
        # Release camera resources if recording is in progress
        # self.camera.release()

        # Call the superclass method to perform additional cleanup
        super().stop_rec()

        # Remove local video files
        self.clear_local_videos()


class PiCamera(Camera):
    """A class to manage a rasberry pi camera."""

    def __init__(
        self,
        resolution_x: int = 1280,
        resolution_y: int = 720,
        fps: int = 15,
        sensor_mode: int = 1,
        exposure: int = 10000,
        camera_num: int = 0,
        file_format: str = "rgb",
        logger_timer: Optional["Timer"] = None,
        **kwargs,
    ):
        if not globals()["IMPORT_PICAMERA"]:
            raise ImportError(
                "the picamera package could not be imported, install it before use!"
            )
        # PicameraOutput annotates every frame with cv2, so a missing cv2 would
        # otherwise surface as a NameError inside the recording thread.
        if not globals()["IMPORT_CV2"]:
            raise ImportError(
                "The cv2 package could not be imported. "
                "Please install it before using PiCamera.\n"
                "On Raspberry Pi OS install it from apt so it links against the "
                "system numpy:\n"
                "sudo apt install python3-opencv"
            )
        self.initialized = threading.Event()
        self.initialized.clear()
        self.cam = None
        self.picamera_ouput = None

        self.sensor_mode = sensor_mode
        self.resolution = (resolution_x, resolution_y)
        self.exposure = exposure
        self.camera_num = camera_num
        self.file_format = file_format
        self.tmst_output = None

        self.fps = fps
        self.logger_timer = logger_timer

        self._lock_serving = Lock()
        self._counter_serving = 0
        self._encoder_serving = None
        self._output_serving = None

        super().__init__(kwargs["filename"], kwargs["logger"], kwargs["video_aim"])

    @property
    def fps(self) -> int:
        """Get the frames per second of the camera."""
        return self._fps

    @fps.setter
    def fps(self, fps: int):
        """Set the frames per second of the camera."""
        if not isinstance(fps, int):
            raise TypeError("FPS must be an integer.")
        self._fps = fps
        if self.initialized.is_set():
            self.cam.framerate = self._fps

    def setup(self):
        """Setup the camera."""
        if self.logger is not None:
            self.tmst_type = "h5"
            self.dataset = self.logger.createDataset(
                dataset_name="frame_tmst",
                dataset_type=np.dtype([("txt", np.double)]),
                filename=self.filename_tmst,
                db_log=False,
            )
        else:
            self.tmst_type = "txt"
            self.tmst_output = io.open(
                os.path.join(self.source_path, f"tmst_{self.filename}.txt"),
                "w",
                encoding="utf-8",
            )
        super().setup()

    def rec(self) -> None:
        """Start recording"""
        try:
            if self.recording.is_set():
                warnings.warn("Camera is already recording!")
                return

            self.recording_init()
            self.cam.start()
            while not self.stop.is_set():
                time.sleep(1)
        except Exception as rec_error:
            raise RuntimeError(
                f"Error during camera recording: {rec_error}"
            ) from rec_error
        finally:
            self._stop_recording()

    def recording_init(self) -> None:
        """Initialize the recording."""
        self.recording.set()
        self.cam = self.init_cam()
        self._start_http_server()

    def _start_http_server(self) -> None:
        """Serve the camera over HTTP, if a port is configured.

        Must run after self.cam is assigned: start_serving() dereferences it
        from the HTTP handler thread, so the server cannot accept a client any
        earlier without racing the camera being ready.
        """
        if self.serve_port <= 0:
            return
        # One port per camera, so several cameras in one setup do not all try
        # to bind server.port.
        port = self.serve_port + self.camera_num
        try:
            self.httpthread = HTTPServerThread(
                self,
                serve_port=port,
                server_user=self.server_user,
                server_password=self.server_password,
                serve_fps=self.serve_fps,
            )
        except OSError:
            # Streaming is an accessory: a port that is busy (usually a camera
            # process left over from an earlier run) must not stop the
            # recording.
            self.httpthread = None
            log.exception(
                "Camera %s: could not serve on port %s, continuing without "
                "the video stream.",
                self.filename,
                port,
            )
            return
        self.httpthread.start()

    def init_cam(self) -> "Picamera2":
        """Initialize the camera."""
        # Future: support string device identifiers so cameras can be addressed
        # by role (via udev/libcamera config) instead of enumeration order.
        picam2 = Picamera2(camera_num=self.camera_num)
        _mode = picam2.sensor_modes[self.sensor_mode]
        config = picam2.create_video_configuration(
            raw={"size": _mode["size"], "format": _mode["format"].format},
            main={
                "format": "RGB888",
                "size": self.resolution,
            },
            lores={
                "format": "YUV420",
                "size": (int(self.resolution[0] / 4), int(self.resolution[1] / 4)),
            },
            controls={
                "FrameDurationLimits": (int(1e6 / self.fps), int(1e6 / self.fps)),
                "ExposureTime": int(self.exposure),
                # "AfMode": controls.AfModeEnum.Manual,
                # "LensPosition": 0.0,
            },
        )
        picam2.configure(config)
        self.picamera_ouput = PicameraOutput(
            self.logger_timer, self.frame_queue, self.process_queue, self.post_process
        )
        picam2.post_callback = lambda request: self.picamera_ouput.annotate_timestamp(
            request
        )  # pylint: disable=all
        encoder = H264Encoder(10000000)
        output = FfmpegOutput(str(Path(self.source_path) / f"{self.filename}.mp4"))
        picam2.start_encoder(encoder, output)

        return picam2

    def _stop_recording(self) -> None:
        """Stop recording."""
        if self.recording.is_set():
            if self.httpthread:
                self.httpthread.stop_serving()
            # cam is None when init_cam() raised; without this the AttributeError
            # here would replace the real initialisation error.
            if self.cam is not None:
                self.cam.stop_recording()
                self.cam.close()

        if self.tmst_type == "txt":
            self.tmst_output.close()
        else:
            self.dataset.exit()

        self.recording.clear()
        self.cam = None
        self.clear_local_videos()

    def write_frame(self, item: Union[List, tuple]) -> None:
        """Write a frame to the output."""
        if not self.stop.is_set():
            if self.tmst_type == "txt":
                self.tmst_output.write(f"{item[0]}\n")
            elif self.tmst_type == "h5":
                self.dataset.append("frame_tmst", [item[0]])

    def start_serving(self) -> "StreamingOutput":
        """Start serving frames."""
        with self._lock_serving:
            if self._counter_serving == 0:
                self._encoder_serving = MJPEGEncoder()
                self._encoder_serving.framerate = self._fps
                self._output_serving = StreamingOutput()
                self.cam.start_recording(
                    self._encoder_serving, FileOutput(self._output_serving)
                )
            self._counter_serving += 1
        return self._output_serving

    def stop_serving(self) -> None:
        """Stop serving frames."""
        with self._lock_serving:
            self._counter_serving -= 1
            if self._counter_serving == 0:
                self.cam.stop_encoder(self._encoder_serving)
                self._encoder_serving = None
                self._output_serving = None


class PicameraOutput:
    """Process the output of the PiCamera."""

    def __init__(self, timer: Any, frame_queue: Any, process_queue: Any, post_process):
        self.timer = timer
        self.frame_queue = frame_queue
        self.process_queue = process_queue
        self.post_process = post_process
        self.position = (8, 16)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 255, 255)

    def annotate_timestamp(self, request: Any) -> None:
        """Annotate the frame with a timestamp."""
        timestamp = f"{self.timer.elapsed_time()}"
        with MappedArray(request, "main") as frame:
            cv2.putText(
                frame.array, timestamp, self.position, self.font, 1.0, self.color
            )
            self.frame_queue.put((timestamp,))
            if self.post_process.is_set():
                self.process_queue.put((timestamp, frame.array))


class StreamingOutput(io.BufferedIOBase):
    """A class that handles the streaming output."""

    def __init__(self):
        super().__init__()
        self.frame = None
        self.tmst_time = None
        self.condition = Condition()

    def write(self, buf: bytes) -> None:
        """Write the buffer to the frame and notify all waiting threads."""
        with self.condition:
            self.frame = buf
            self.tmst_time = time.time()
            self.condition.notify_all()


class HTTPServerThread(Thread):
    """A class that handles the HTTP server thread."""

    def __init__(
        self,
        cam: "Camera",
        serve_port: int = 8000,
        server_user: Optional[str] = None,
        server_password: Optional[str] = None,
        serve_fps: float = 0,
    ):
        super().__init__()
        self.python_logger = logging.getLogger(self.__class__.__name__)
        self.server = ThreadingHTTPServer(
            ("", serve_port), self.CameraHTTPRequestHandler
        )
        self.server.cam = cam
        # 0 (the default) streams every frame the encoder produces.
        self.server.serve_interval = 1 / serve_fps if serve_fps > 0 else 0
        self.server.auth = None
        if server_user and server_password:
            str_auth = f"{server_user}:{server_password}"
            self.server.auth = "Basic " + base64.b64encode(str_auth.encode()).decode()

    def run(self) -> None:
        """Start the server."""
        self.python_logger.info(
            "Starting HTTP server on port %s", self.server.server_port
        )
        self.server.serve_forever()

    def stop_serving(self) -> None:
        """Stop the server."""
        self.python_logger.info("Stopping HTTP server")
        self.server.shutdown()

    class CameraHTTPRequestHandler(BaseHTTPRequestHandler):
        """A class that handles HTTP requests for the camera."""

        def logger(self) -> logging.Logger:
            """Return the logger for this class."""
            return logging.getLogger("HTTPRequestHandler")

        def check_auth(self) -> bool:
            """Check if the request is authorized."""
            if self.server.auth is None or self.server.auth == self.headers.get(
                "authorization"
            ):
                return True
            else:
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Basic")
                self.end_headers()
                return False

        def send_jpeg(self, output: StreamingOutput) -> None:
            """Send a JPEG image."""
            # Take a reference under the lock but write outside it: the encoder
            # holds the same condition in StreamingOutput.write, so a slow
            # client must not block frame production.
            with output.condition:
                output.condition.wait()
                frame = output.frame
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame))
            self.end_headers()
            self.wfile.write(frame)

        def do_GET(self) -> None:
            """Handle a GET request."""
            if self.path == "/cam.mjpg":
                if self.check_auth():
                    output = self.server.cam.start_serving()
                    try:
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
                        )
                        self.end_headers()

                        while not self.wfile.closed:
                            self.wfile.write(b"--FRAME\r\n")
                            self.send_jpeg(output)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            # Throttle the stream: send_jpeg blocks until the
                            # next frame, so sleeping here drops the ones in
                            # between instead of pushing them over the network.
                            if self.server.serve_interval:
                                time.sleep(self.server.serve_interval)
                    except IOError as err:
                        self.logger().error(
                            "Exception while serving client %s: %s",
                            self.client_address,
                            err,
                        )
                    finally:
                        self.server.cam.stop_serving()
                        output = None
            else:
                self.send_error(404)
