    def __init__(self, fixation_id: int, timestamp: float, duration: float, norm_pos: np.ndarray, video_resolution):
        self._is_complete = False
        self._fixation_id = fixation_id
        self._timestamps = []
        self._durations = []
        self._norm_pos = []
        self._video_resolution = video_resolution
        self.add_partial_fixation(fixation_id, timestamp, duration, norm_pos)

    def add_partial_fixation(self, fixation_id: int, timestamp: float, duration: float, norm_pos: np.ndarray):
        assert self._fixation_id == fixation_id
        self._timestamps.append(timestamp)
        self._norm_pos.append(norm_pos)
        self._durations.append(duration)

    def finalize(self) -> 'Fixation':
        self._is_complete = True
        return self

    @property
    def index(self):
        return self._fixation_id

    @property
    def timestamp(self) -> float:
        return self._timestamps[0]

    @property
    def duration(self):
        if self._is_complete:
            return (self._timestamps[-1] - self._timestamps[0]) * 1000. + self._durations[-1]
        return None

    @property
    def fixation_point(self):
        norm_positions = np.vstack(self._norm_pos)
        fixation = GazeSample(
            gaze=norm_positions.mean(axis=0),
            normalized=True,
            reference_size=self._video_resolution,
            origin=GazeSample.ORIGIN_BOTTOM_LEFT
        )
        return fixation
