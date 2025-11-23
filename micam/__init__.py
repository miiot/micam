import os
import asyncio
import aiohttp
import argparse
import logging
import subprocess
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RTSPBridge:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        camera_id: str,
        rtsp_url: str,
        video_codec='hevc',
        channel=0,
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.camera_id = camera_id
        self.channel = channel
        self.video_codec = video_codec
        self.rtsp_url = rtsp_url
        self.process: Optional[subprocess.Popen] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.waiting_for_keyframe = True
        self.ffmpeg_monitor_task: Optional[asyncio.Task] = None
        self._ffmpeg_should_run = False

    async def _login(self) -> bool:
        """Login and retrieve access token."""
        login_url = f"{self.base_url}/api/auth/login"
        payload = {"username": self.username, "password": self.password}

        try:
            async with self.session.post(login_url, json=payload, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Login successful. %s", data)

                    # Call login_status to ensure session is active
                    status_url = f"{self.base_url}/api/miot/login_status"
                    async with self.session.get(status_url, ssl=False) as status_resp:
                        if status_resp.status == 200:
                            return True
                        else:
                            logger.error(f"Login status check failed: {status_resp.status}")
                            return False
                else:
                    logger.error(f"Login failed: {response.status} - {await response.text()}")
                    return False
        except Exception as e:
            logger.error(f"Login exception: {e}")
            await asyncio.sleep(3)
            return False

    def _start_ffmpeg(self):
        """Start FFmpeg process."""
        # FFmpeg command to read from stdin and publish to RTSP
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-v', 'error',
            '-hide_banner',
            '-use_wallclock_as_timestamps', '1',  # Generate timestamps from arrival time
            '-analyzeduration', '20000000',  # 20 seconds
            '-probesize', '20000000',  # 20 MB
            '-f', self.video_codec,  # Input format
            '-i', 'pipe:0',  # Read from stdin
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'copy',  # Copy audio stream
            '-f', 'rtsp',  # Output format
            '-rtsp_transport', 'tcp',  # Use TCP for RTSP
            self.rtsp_url,
        ]

        logger.info(f"Starting FFmpeg: {' '.join(ffmpeg_cmd)}")
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # Suppress FFmpeg stdout
            stderr=subprocess.PIPE,  # Capture stderr for debugging if needed
        )
        self._ffmpeg_should_run = True

    def _stop_ffmpeg(self):
        """Stop FFmpeg process."""
        self._ffmpeg_should_run = False
        if self.ffmpeg_monitor_task:
            self.ffmpeg_monitor_task.cancel()
            try:
                asyncio.get_running_loop().run_until_complete(self.ffmpeg_monitor_task)
            except:
                pass
            self.ffmpeg_monitor_task = None

        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None

    async def _monitor_ffmpeg(self):
        """Monitor FFmpeg process and restart if it exits unexpectedly."""
        while self._ffmpeg_should_run:
            await asyncio.sleep(1)  # Check every second
            if self.process and self.process.poll() is not None:
                # FFmpeg process has exited
                return_code = self.process.returncode
                logger.warning(f"FFmpeg process exited with return code: {return_code}")

                if self._ffmpeg_should_run:  # Only restart if we're supposed to keep running
                    logger.info("Restarting FFmpeg process...")
                    try:
                        # Stop the old process
                        self._stop_ffmpeg()
                        # Start a new process
                        self._start_ffmpeg()
                        logger.info("FFmpeg process restarted successfully")
                    except Exception as e:
                        logger.error(f"Failed to restart FFmpeg process: {e}")
                        # Wait a bit before trying again
                        await asyncio.sleep(5)

    async def run(self):
        """Main loop to connect to WebSocket and pipe data."""
        self._start_ffmpeg()

        # Start FFmpeg monitor task
        self.ffmpeg_monitor_task = asyncio.create_task(self._monitor_ffmpeg())

        jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(cookie_jar=jar) as session:
            self.session = session
            if not await self._login():
                self._stop_ffmpeg()
                return

            protocol = "wss" if self.base_url.startswith("https") else "ws"
            host = self.base_url.split("://")[1]
            ws_url = f"{protocol}://{host}/api/miot/ws/video_stream?camera_id={self.camera_id}&channel={self.channel}"
            logger.info(f"Connecting to WebSocket: {ws_url}")

            try:
                async with session.ws_connect(ws_url, ssl=False) as ws:
                    logger.info("WebSocket connected. Streaming data...")

                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=60.0)
                        except asyncio.TimeoutError:
                            logger.error("Data received timeout. Exiting.")
                            break

                        if msg.type == aiohttp.WSMsgType.BINARY:
                            try:
                                data_len = len(msg.data)
                                if data_len >= 100:
                                    logger.debug("Received binary data: %s", data_len)

                                if self.waiting_for_keyframe:
                                    if self._is_keyframe(msg.data):
                                        logger.info("Keyframe detected! Starting stream...")
                                        self.waiting_for_keyframe = False
                                    else:
                                        logger.debug("Skipping non-keyframe data...")
                                        continue
                                await asyncio.wait_for(self.process_write(msg.data), timeout=30.0)
                            except asyncio.TimeoutError:
                                logger.error("Write data to process timeout.")
                                await self.process_stderr()
                                break
                            except BrokenPipeError as e:
                                logger.error("FFmpeg process terminated unexpectedly. %s", e)
                                await self.process_stderr()
                                # FFmpeg will be restarted by the monitor task, continue streaming
                                continue
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket connection closed with error {ws.exception()}")
                            break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                            logger.info("WebSocket connection close(%s)", msg.type)
                            break
                        else:
                            logger.info(f"Unexpected WebSocket message type: {msg.type}")
            except Exception as e:
                logger.error(f"Streaming error", exc_info=True)
            finally:
                self._stop_ffmpeg()
                logger.info("Stream finished")

    def _is_keyframe(self, data: bytes) -> bool:
        if self.video_codec == 'h264':
            i = 0
            while i < len(data) - 4:
                if (
                    data[i] == 0x00 and data[i + 1] == 0x00 and
                    ((data[i + 2] == 0x00 and data[i + 3] == 0x01) or data[i + 2] == 0x01)
                ):
                    nal_unit_type = (data[i + 3] & 0x1f) if data[i + 2] == 0x01 else (data[i + 4] & 0x1f)
                    return nal_unit_type == 5
                i += 1
            return False
        elif self.video_codec == 'hevc':
            i = 0
            while i < len(data) - 6:
                if (
                    data[i] == 0x00 and data[i + 1] == 0x00 and
                    ((data[i + 2] == 0x00 and data[i + 3] == 0x01) or data[i + 2] == 0x01)
                ):
                    nal_start = i + 3 if data[i + 2] == 0x01 else i + 4
                    nal_unit_type = (data[nal_start] >> 1) & 0x3f
                    if nal_unit_type in [16, 17, 18, 19, 20]:
                        return True
                i += 1
            return False
        return True

    async def process_write(self, data):
        if not self.process:
            logger.warning("FFmpeg process not available, waiting for restart...")
            # Wait briefly to allow the monitor task to restart FFmpeg
            await asyncio.sleep(0.1)
            if not self.process or not self.process.stdin:
                raise RuntimeError("FFmpeg process not available after restart attempt")
        if not self.process.stdin:
            raise RuntimeError("Process has no stdin")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._process_write, data)

    def _process_write(self, data):
        self.process.stdin.write(data)
        self.process.stdin.flush()

    async def process_stderr(self):
        if not self.process:
            logger.warning("FFmpeg process not available for stderr reading")
            return
        if not self.process.stderr:
            return
        stderr = self.process.stderr.read().decode()
        if stderr:
            logger.error(f"FFmpeg stderr: %s", stderr)


def main():
    parser = argparse.ArgumentParser(description="Bridge WebSocket video stream to RTSP")
    parser.add_argument("--base-url", default="", help="Base URL of the Miloco server")
    parser.add_argument("--username", default="admin", help="Login username")
    parser.add_argument("--password", default="", help="Login password (MD5)")
    parser.add_argument("--camera-id", default="", help="Camera ID to stream")
    parser.add_argument("--channel", default="", help="Camera channel")
    parser.add_argument("--video-codec", default="hevc", help="Input video codec (hevc or h264)")
    parser.add_argument("--rtsp-url", default="", help="Target RTSP URL")

    args = parser.parse_args()

    password = args.password or os.getenv("MILOCO_PASSWORD", "")
    if not password:
        logger.error("Password is required")
        return

    camera_id = args.camera_id or os.getenv("CAMERA_ID", "")
    if not camera_id:
        logger.error("Camera ID is required")
        return

    bridge = RTSPBridge(
        base_url=args.base_url or os.getenv("MILOCO_BASE_URL", "https://miloco:8000"),
        username=args.username or os.getenv("MILOCO_USERNAME", "admin"),
        password=password,
        camera_id=camera_id,
        rtsp_url=args.rtsp_url or os.getenv("RTSP_URL", "rtsp://0.0.0.0:8554/live"),
        video_codec=args.video_codec or os.getenv("VIDEO_CODEC", "hevc"),
        channel=args.channel or os.getenv("STREAM_CHANNEL", "0"),
    )

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        logger.info("Stopped by user")


if __name__ == "__main__":
    main()
