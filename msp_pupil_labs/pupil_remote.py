import io
from typing import Optional, List
import msgpack
import zmq
from PIL import Image


class GazeSample:
    """
    Data structure for gaze data. It enforces
    (1) that the origin of gaze coordinates is at the upper left, and
    (2) the gaze coordinates are normalized.
    """

    # see also https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
    ORIGIN_BOTTOM_LEFT = "bl"
    ORIGIN_TOP_LEFT = "tl"

    def __init__(self, gaze, reference_size, normalized=True, origin="bl"):
        self._gaze = self._normalize(gaze, normalized, origin)
        self._max_width = reference_size[0]
        self._max_height = reference_size[1]

    def _normalize(self, gaze, normalized, origin):
        x, y = tuple(gaze)
        if not normalized:
            x /= float(self.max_width)
            y /= float(self.max_height)

        if origin == self.ORIGIN_BOTTOM_LEFT:
            # convert to top-left coordinate
            y = 1. - y

        return x, y

    @property
    def x(self):
        return self.gaze[0]

    @property
    def y(self):
        return self.gaze[1]

    @property
    def gaze(self):
        return self._gaze

    @property
    def x_scaled(self):
        return self.x * self.max_width

    @property
    def y_scaled(self):
        return self.y * self.max_height

    @property
    def gaze_scaled(self):
        return self.x_scaled, self.y_scaled

    @property
    def max_width(self):
        return self._max_width

    @property
    def max_height(self):
        return self._max_height


class PupilRemote:
    class Streams:
        GAZE = "gaze."
        FIXATIONS = "fixations"
        SURFACES = "surfaces"
        SCENE_VIDEO = "frame.world"

        # lists plugin requirements for all available signal streams
        _plugin_requirements = {
            FIXATIONS: {"subject": "start_plugin", "name": "Fixation_Detector"},
            SURFACES: {"subject": "start_plugin", "name": "Surface_Tracker_Online"},
            SCENE_VIDEO: {"subject": "start_plugin", "name": "Frame_Publisher", "args": {"format": "jpeg"}}
        }

        @staticmethod
        def requirement(stream):
            if stream in PupilRemote.Streams._plugin_requirements:
                return PupilRemote.Streams._plugin_requirements[stream]
            return None

    def __init__(self, address: str = "127.0.0.1", port: int = 50020, streams: Optional[List[str]] = None):
        self.ctx = zmq.Context()
        self.pupil_remote = zmq.Socket(self.ctx, zmq.REQ)
        self.pupil_remote_ip = address
        self.pupil_remote_port = port
        self.pupil_sub_port = 0
        self.pupil_pub_port = 0
        self.subscriber = None
        self.video_resolution = None
        self.last_fixation_id = None

        self._streams = streams
        if streams is None:
            self._streams = [
                PupilRemote.Streams.GAZE,
                PupilRemote.Streams.FIXATIONS,
                PupilRemote.Streams.SCENE_VIDEO
            ]

    def connect(self):
        self.pupil_remote.connect(f"tcp://{self.pupil_remote_ip}:{self.pupil_remote_port}")
        self.pupil_remote.send_string('SUB_PORT')
        self.pupil_sub_port = self.pupil_remote.recv_string()
        self.pupil_remote.send_string('PUB_PORT')
        self.pupil_pub_port = self.pupil_remote.recv_string()

        self._start_required_plugins()
        self._start_subscription()

    def close(self):
        self.subscriber.close()
        self.pupil_remote.close()

    def _start_required_plugins(self):
        for stream in self._streams:
            requirement = self.Streams.requirement(stream)
            if requirement is not None:
                self._send_pupil_notification(requirement)

    def _send_pupil_notification(self, payload):
        topic = f"notify.{payload['subject']}"
        payload = msgpack.packb(payload, use_bin_type=True)
        self.pupil_remote.send_string(topic, flags=zmq.SNDMORE)
        self.pupil_remote.send(payload)
        return self.pupil_remote.recv_string()

    def _start_subscription(self):
        self.subscriber = self.ctx.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.pupil_remote_ip}:{self.pupil_sub_port}")
        for event in self._streams:
            self.subscriber.subscribe(event)

    def _recv_sub_event(self):
        try:
            # no block because of closing error
            topic = self.subscriber.recv_string(flags=zmq.NOBLOCK)
            payload = self.subscriber.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload[0])
            if len(payload) > 1:
                message["raw_img"] = payload[1:]  # e.g., the encoded image of the scene camera
        except Exception as e:
            if e.errno == 11:  # Resource is temporarily not available
                return None, None
            else:
                raise e

        return topic, message

    @staticmethod
    def _decode_image(raw_img, w, h, format="bgr") -> Image.Image:
        if format == "bgr":
            img = Image.frombytes(mode="RGB", size=(w, h), data=raw_img)
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))
            return img
        elif format == "jpeg":
            img = Image.open(io.BytesIO(raw_img))
            img.load()  # is required, because Pillow uses lazy loading which causes problems later on
            return img
        else:
            raise NotImplementedError(f"image format {format} is not supported.")

    def get_next_event(self):
        topic, message = self._recv_sub_event()

        if topic is not None:
            if topic == PupilRemote.Streams.SCENE_VIDEO:
                encoded_image = message["raw_img"][0]
                if encoded_image is None:
                    return
                image = self._decode_image(encoded_image, message["width"], message["height"], format=message["format"])
                if self.video_resolution is None:
                    self.video_resolution = image.size
                message["image"] = image
                return topic, message

            elif topic.startswith(PupilRemote.Streams.GAZE) and self.video_resolution is not None:
                topic = PupilRemote.Streams.GAZE
                # see https://docs.pupil-labs.com/developer/core/overview/#gaze-datum-format
                gaze_sample = GazeSample(
                    gaze=message["norm_pos"],
                    normalized=True,
                    reference_size=self.video_resolution,
                    origin=GazeSample.ORIGIN_BOTTOM_LEFT
                )
                message["gaze"] = gaze_sample
                return topic, message

            elif topic.startswith(PupilRemote.Streams.FIXATIONS) and self.video_resolution is not None:
                fixation_id = message["id"]
                if self.last_fixation_id is None or fixation_id >= self.last_fixation_id:
                    self.last_fixation_id = fixation_id
                else:
                    return None, None
                topic = PupilRemote.Streams.FIXATIONS
                gaze_sample = GazeSample(
                    gaze=message["norm_pos"],
                    normalized=True,
                    reference_size=self.video_resolution,
                    origin=GazeSample.ORIGIN_BOTTOM_LEFT
                )
                message["fixation"] = gaze_sample
                return topic, message

        return None, None
