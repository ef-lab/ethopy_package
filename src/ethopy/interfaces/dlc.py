import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from queue import Empty
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
    IMPORT_CV2 = True
except ImportError:
    IMPORT_CV2 = False
import numpy as np

try:
    from dlclive import DLCLive, Processor
    IMPORT_DLCLive = True
except ImportError:
    IMPORT_DLCLive = False
from ethopy.utils.helper_functions import read_yalm, shared_memory_array

np.set_printoptions(suppress=True)


class ModelInterface(ABC):
    @abstractmethod
    def setup_model(self, frame):
        pass

    @abstractmethod
    def get_pose(self, frame):
        pass


class DLCModel(ModelInterface):
    @classmethod
    def set_environment_variables(cls):
        os.environ["DLClight"] = "True"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    def __init__(self, path: str):
        if not globals()["IMPORT_DLCLive"]:
            raise ImportError(
                "Please install dlc_live before using DLCProcessor.\n"
                "sudo pip3 install deeplabcut-live"
            )
        self.set_environment_variables()
        self.dlc_model = None
        self.path = path
        self.dlc_processor = Processor()
        self.joint_names = read_yalm(self.path, "pose_cfg.yaml", "all_joints_names")

    def setup_model(self, frame):
        self.dlc_model = DLCLive(self.path, processor=self.dlc_processor)
        self.dlc_model.init_inference(frame / 255)

    def get_pose(self, frame):
        return self.dlc_model.get_pose(frame / 255)


class DLCProcessor(ABC):
    """
    Base class for DeepLabCut (DLC) model processing.

    Args:
        frame_queue (mp.Queue): Queue for incoming frames.
        model_path (str): Path to the DLC model.
        logger (Optional[Any]): Logger object for logging processing results.
        wait_for_setup (bool): Whether to wait for setup completion before returning.
    """

    def __init__(
        self,
        frame_queue: mp.Queue,
        model_path: str,
        logger: Optional[Any] = None,
        wait_for_setup: bool = False,
    ):
        if not globals()["IMPORT_DLCLive"]:
            raise ImportError(
                "Please install dlc_live before using DLCProcessor.\n"
                "sudo pip3 install deeplabcut-live"
            )
        print("model_path ", model_path)
        self.model = DLCModel(model_path)
        self.frame_queue = frame_queue
        self.frame_timeout = 1
        self.logger = logger

        self.setup_complete = mp.Event()
        self.stop_signal = mp.Event()
        self.finish_signal = mp.Event()
        self.finish_signal.clear()

        self.current_frame = None
        self.dlc_process = mp.Process(target=self._setup_and_run)
        self.dlc_process.start()

        if wait_for_setup:
            self._wait_for_setup()

    def _wait_for_setup(self):
        """Wait for the DLC model setup to complete."""
        self.setup_complete.wait(timeout=30)

    def _setup_and_run(self):
        """Set up the DLC model and start processing."""
        self._setup_model()
        self.setup_complete.set()
        self.process_frames()

    def _setup_model(self):
        """Initialize the DLC model."""
        _, frame = self.frame_queue.get_nowait()
        self.model.setup_model(frame)

    def process_frames(self):
        """Common method to process frames using the model."""
        try:
            while not self.finish_signal.is_set():
                self.latest_frame = None
                latest_timestamp = None
                # Drain the queue, keeping only the latest frame
                ask_frame = time.time()
                while self.frame_queue.qsize() > 0:
                    try:
                        latest_timestamp, self.latest_frame = self.frame_queue.get_nowait()
                    except Empty:
                        if self.frame_queue.qsize() == 0:
                            break  # Queue became empty while we were draining it
                delay_time = time.time() - ask_frame
                if self.latest_frame is not None:
                    frame_tranfer_delay = self.logger.logger_timer.elapsed_time()-latest_timestamp
                    if frame_tranfer_delay > 100:
                        print(f"###############################frame transfer delay: {frame_tranfer_delay} ms")
                    # print('exception qsize', self.frame_queue.qsize(), self.frame_queue.empty())
                    if delay_time > 0.01:
                        print(f"------------------------------------------ DLC queue empty delay: {delay_time} sec")
                    pose = self.model.get_pose(self.latest_frame)
                    self._process_frame(pose, latest_timestamp)
                    # print("time ", time.time()-start_t)
                else:
                    # If stop signal is set wait until there is no new frames(Close camera)
                    if self.stop_signal.is_set():
                        break
                    time.sleep(0.01)  # Short sleep to prevent busy-waiting
        except Exception as e:
            # Log any exceptions that occur during frame processing
            print(f"Frame processing error: {e}")
        finally:
            # Ensure cleanup is always executed, even if an error occurs
            print("Frame process has been finished.")
            self._process_finish()
        self.finish_signal.clear()

    def _process_frame(self, pose, timestamp, **kwargs):
        """Process frames using the DLC model. To be implemented by subclasses."""
        pass

    def _process_finish(self):
        """Process frames using the DLC model. To be implemented by subclasses."""
        pass

    def stop(self):
        """Stop the DLC processing."""
        if not self.dlc_process.is_alive():
            return
        self.stop_signal.set()
        self.dlc_process.join(timeout=60)
        if self.dlc_process.is_alive():
            print("Terminate dlc process")
            self.dlc_process.terminate()  # Force terminate if not stopping.


class DLCCornerDetector(DLCProcessor):
    """
    DLC processor for detecting arena corners.

    Args:
        frame_queue (mp.Queue): Queue for incoming frames.
        model_path (str): Path to the DLC model.
        arena_size (float): Size of the arena.
        result (mp.managers.DictProxy): Shared dictionary to store results.
        logger (Optional[Any]): Logger object for logging processing results.
    """
    CONFIDENCE_THRESHOLD = 0.85
    MIN_CONFIDENT_FRAMES = 4

    def __init__(
        self,
        frame_queue: mp.Queue,
        model_path: str,
        arena_size: float,
        result: Dict,
        logger: Optional[Any] = None,
    ):
        if not globals()["IMPORT_CV2"]:
            raise ImportError(
                "The cv2 package could not be imported. "
                "Please install it before using DLCCornerDetector.\n"
                "You can install cv2 using pip:\n"
                "sudo pip3 install opencv-python"
            )
        self.arena_size = arena_size
        self.result = result  # Use the passed multiprocessing dictionary
        self.detected_corners = []
        super().__init__(frame_queue, model_path, logger)

    def _process_frame(self, pose, timestamp):
        """Detect arena corners and calculate perspective transform."""
        if np.all(pose[:, 2] > self.CONFIDENCE_THRESHOLD):
            self.detected_corners.append(pose)
        else:
            print("\rWait for high confidence corners scores", pose[:, 2], end="")
        if len(self.detected_corners) >= self.MIN_CONFIDENT_FRAMES or self.stop_signal.is_set():
            self.finish_signal.set()

    def _process_finish(self):
        self.corners = np.mean(np.array(self.detected_corners), axis=0)
        self.affine_matrix, self.affine_matrix_inv = self._calculate_perspective_transform(
            self.corners, self.arena_size
        )

        # update dict
        self.result.update({
            "corners": self.corners,
            "affine_matrix": self.affine_matrix,
            "affine_matrix_inv": self.affine_matrix_inv,
        })
        for corner in self.corners:
            self.latest_frame = cv2.circle(self.latest_frame,
                                           (int(corner[0]), int(corner[1])),
                                           radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imwrite("corners_check.jpg", self.latest_frame)

    @staticmethod
    def _calculate_perspective_transform(corners: np.ndarray, screen_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the perspective transform for the arena.

        Args:
            corners (np.ndarray): Detected corner coordinates.
            screen_size (float): Size of the arena screen.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Affine matrix and its inverse.
        """
        pts1 = np.float32([corner[:2] for corner in corners])
        pts2 = np.float32(
            [[0, 0], [screen_size, 0], [0, screen_size], [screen_size, screen_size]]
        )
        m = cv2.getPerspectiveTransform(pts1, pts2)
        return m, np.linalg.inv(m)


class DLCContinuousPoseEstimator(DLCProcessor):
    """
    DLC processor for continuous pose estimation.

    Args:
        frame_queue (mp.Queue): Queue for incoming frames.
        model_path (str): Path to the DLC model.
        logger (Any): Logger object for logging processing results.
        shared_memory_conf(Dict): information needed to define the shared memory.
        affine_matrix (np.ndarray): Affine transformation matrix.
    """

    def __init__(
        self,
        frame_queue: mp.Queue,
        model_path: str,
        logger: Any,
        shared_memory_conf: Dict,
        affine_matrix: np.ndarray,
        wait_for_setup: bool
    ):
        if not globals()["IMPORT_CV2"]:
            raise ImportError(
                "The cv2 package could not be imported. "
                "Please install it before using DLCContinuousPoseEstimator.\n"
                "You can install cv2 using pip:\n"
                "sudo pip3 install opencv-python"
            )
        self.affine_matrix = affine_matrix
        # rotation_angle is used in case an ear is missing check _update_pose
        self.rotation_angle = self._calculate_rotation_angle(side=1, base=0.8)
        # attach another shared memory block
        self.logger = logger
        self.result, self.shared_memory, _ = shared_memory_array(name=shared_memory_conf['name'],
                                                                 rows_len=shared_memory_conf['shape'][0],
                                                                 columns_len=shared_memory_conf['shape'][1],
                                                                 )
        if self.logger:
            folder = (f"Recordings/{self.logger.trial_key['animal_id']}"
                      f"_{self.logger.trial_key['session']}/")
            self.source_path = self.logger.source_path + folder
            self.target_path = self.logger.target_path + folder
            h5s_filename = (f"{self.logger.trial_key['animal_id']}_"
                            f"{self.logger.trial_key['session']}_"
                            f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.h5")
            self.filename_dlc = "dlc_" + h5s_filename
            self.logger.log_recording(
                dict(
                    rec_aim="openfield",
                    software="EthoPy",
                    version="0.1",
                    filename=self.filename_dlc,
                    source_path=self.source_path,
                    target_path=self.target_path,
                ),
                block=True,
            )
        super().__init__(frame_queue, model_path, logger, wait_for_setup)

    def _setup_model(self):
        super()._setup_model()
        if self.logger is not None:
            self._setup_hdf5_datasets(self.logger)
        # get a high confidence pose before start processing
        self.prev_pose = self._initialize_pose()

    def _setup_hdf5_datasets(self, logger):
        """Set up HDF5 datasets for logging pose data."""
        joints_types = [("timestamp", np.double)] + [
            (f"{joint}{p}", np.double)
            for joint in self.model.joint_names
            for p in ["_x", "_y", "_score"]
        ]
        self.pose_hdf5 = logger.createDataset(
            "dlc", np.dtype(joints_types), self.filename_dlc, log=False
        )
        self.pose_hdf5_infer = logger.createDataset(
            "dlc_infer", np.dtype(joints_types), self.filename_dlc, log=False
        )

        processed_joints_types = [
            ("timestamp", np.double),
            ("head_x", np.double),
            ("head_y", np.double),
            ("angle", np.double),
        ]
        self.pose_hdf5_processed = logger.createDataset(
            "dlc_processed",
            np.dtype(processed_joints_types),
            self.filename_dlc,
            log=False,
            )

    def _process_frame(self, pose, timestamp):
        """Continuously process frames and estimate pose."""
        if self.stop_signal.is_set():
            if self.frame_queue.empty():
                self.finish_signal.set()
                return

        self.pose_hdf5.append(
            "dlc", np.insert(np.double(pose.ravel()), 0, timestamp)
        )

        # check if position needs any intervation
        current_pose = self._update_pose(pose, self.prev_pose)
        final_pose = self._get_processed_pose(current_pose, timestamp)

        # save pose to the shared memory
        self.result[:] = final_pose
        # save in the hdf5 files
        self.pose_hdf5_infer.append("dlc_infer", np.insert(np.double(current_pose.ravel()), 0, timestamp))
        self.pose_hdf5_processed.append("dlc_processed", final_pose)

        self.prev_pose = current_pose

    def _process_finish(self):
        if self.logger is not None:
            self.logger.closeDatasets()

    def _initialize_pose(self, confidence_threshold: float = 0.01) -> np.ndarray:
        """
        Initialize the current pose with a high-confidence estimate.

        Returns:
            np.ndarray: Initial pose estimation.
        """
        while True:
            if not self.frame_queue.empty():
                _, frame = self.frame_queue.get_nowait()
                pose = self.model.get_pose(frame)
                print("frame ", frame)
                scores = np.array(pose[0:3][:, 2])
                print("\rWait for high confidence pose scores ", scores, end="")
                if np.sum(scores >= confidence_threshold) == 3:
                    return pose
            time.sleep(0.1)

    def _update_pose(
        self, pose: np.ndarray, prev_pose: np.ndarray, confidence_threshold: float = 0.85
    ) -> np.ndarray:
        """
        Update the pose estimation based on current and previous poses.

        Args:
            pose (np.ndarray): Current pose estimation.
            prev_pose (np.ndarray): Previous pose estimation.

        Returns:
            np.ndarray: Updated pose estimation.
        """
        scores = pose[:3, 2]
        low_confidence = scores < confidence_threshold
        partial_pose = pose[:3, :-1]  # get nose, ear left and right

        if np.sum(low_confidence) > 1:
            # If more than one point has low confidence, do not update the pose
            return prev_pose
        elif np.sum(low_confidence) == 1:
            high_confidence_points = partial_pose[np.logical_not(low_confidence)]
            if low_confidence[0]:
                # if nose has low confidence
                partial_pose[low_confidence] = self._infer_apex(partial_pose[2, :], partial_pose[1, :])
            else:
                # if ear left has low confidence rotate with positive angle else negative
                angle = self.rotation_angle if low_confidence[1] else -self.rotation_angle
                partial_pose[low_confidence] = self._rotate_point(
                    high_confidence_points[0], high_confidence_points[1], angle
                )
        pose[:3, :-1] = partial_pose

        return pose

    def _get_processed_pose(
        self, pose: np.ndarray, timestamp: float
    ) -> Tuple[float, float, float, float]:
        """
        Get the processed pose data including centroid and angle.

        Args:
            pose (np.ndarray): Current pose estimation.
            timestamp (float): Timestamp of the current frame.

        Returns:
            Tuple[float, float, float, float]: Processed pose data (timestamp, x, y, angle).
        """
        triangle_vertices = np.array(pose[0:3, 0:2])
        centroid = np.mean(triangle_vertices, axis=0)
        vector_to_nose = triangle_vertices[0, :] - centroid
        angle = self._compute_angle(vector_to_nose, np.array([1, 0]))

        point = np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        transformed_centroid = cv2.perspectiveTransform(
            np.array([point]), self.affine_matrix
        ).ravel()

        return timestamp, transformed_centroid[0], transformed_centroid[1], angle

    @staticmethod
    def _calculate_rotation_angle(side: float, base: float) -> float:
        """
        Calculate the rotation angle for pose estimation.

        Args:
            side (float): Length of the triangle side.
            base (float): Length of the triangle base.

        Returns:
            float: Rotation angle in radians.
        """
        cos_angle = (2 * side**2 - base**2) / (2 * side**2)
        return np.arccos(cos_angle)

    @staticmethod
    def _rotate_point(
        origin: np.ndarray, point: np.ndarray, angle_rad: float
    ) -> Tuple[float, float]:
        """
        Rotate a point around an origin by a given angle.

        Args:
            origin (np.ndarray): Origin point of rotation.
            point (np.ndarray): Point to be rotated.
            angle_rad (float): Angle of rotation in radians.

        Returns:
            Tuple[float, float]: Coordinates of the rotated point.
        """
        vector = point - origin
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        # Apply rotation matrix to the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return tuple(origin + rotated_vector)

    @staticmethod
    def _infer_apex(
        vertex1: np.ndarray, vertex2: np.ndarray, scaling_factor: float = 0.8
    ) -> Tuple[float, float]:
        """
        Infer the apex of an isoscelic triangle given two vertices.

        Args:
            vertex1 (np.ndarray): First vertex of the triangle.
            vertex2 (np.ndarray): Second vertex of the triangle.
            scaling_factor (float): Empiricaly defined (0.8*dist(left_ear,right_ear)=dist(left or right ear, nose))

        Returns:
            Tuple[float, float]: Coordinates of the inferred apex.
        """
        distance = np.linalg.norm(vertex2 - vertex1)  # distance between the given vertices
        midpoint = (vertex1 + vertex2) / 2  # midpoint of the given vertices
        line_vector = (vertex2 - vertex1) / distance  # unit vector along the line connecting the two vertices
        # direction vector perpendicular to the line connecting the two vertices
        perpendicular_vector = np.array([-line_vector[1], line_vector[0]])
        side_length = scaling_factor * distance  # length of the sides of the isosceles triangle
        apex = midpoint + perpendicular_vector * side_length
        return tuple(apex)

    @staticmethod
    def _compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute the angle between two vectors.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            float: Angle between
        """
        dot_product = np.dot(v1, v2)
        cross_product = np.cross(v1, v2)
        angle = np.arctan2(cross_product, dot_product)
        return np.degrees(angle)

    def stop(self):
        """Stop the continuous pose estimation and clean up resources."""
        try:
            super().stop()
        finally:
            if hasattr(self, 'shared_memory') and self.shared_memory is not None:
                self.shared_memory.close()
                try:
                    self.shared_memory.unlink()
                except FileNotFoundError:
                    print("Shared memory already unlinked or does not exist.")
