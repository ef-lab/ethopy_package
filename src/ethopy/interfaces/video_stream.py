import base64
import io
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Thread
from time import time
from typing import Optional


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
            self.tmst_time = time()
            self.condition.notify_all()


class HTTPServerThread(Thread):
    """A class that handles the HTTP server thread."""

    def __init__(
        self,
        cam: "Camera",
        serve_port: int = 8000,
        server_user: Optional[str] = None,
        server_password: Optional[str] = None,
    ):
        super().__init__()
        self.python_logger = logging.getLogger(self.__class__.__name__)
        self.server = ThreadingHTTPServer(
            ("", serve_port), self.CameraHTTPRequestHandler
        )
        self.server.cam = cam
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
            with output.condition:
                output.condition.wait()
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(output.frame))
                self.end_headers()
                self.wfile.write(output.frame)

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
                    except Exception as err:
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

        def log_error(self, log_format: str, *args) -> None:
            """Log an error message."""
            self.logger().error("%s - %s", self.address_string(), log_format % args)

        def log_message(self, log_format: str, *args) -> None:
            """Log a message."""
            self.logger().info("%s - %s", self.address_string(), log_format % args)
