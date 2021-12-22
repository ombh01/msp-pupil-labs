# Multisensor-Pipeline Extension: Pupil Labs Eye Trackers

Connects the Pupil Core eye tracker of [Pupil Labs](https://pupil-labs.com/) to DFKI's [multisensor-pipeline](https://github.com/DFKI-Interactive-Machine-Learning/multisensor-pipeline).

## Setup

```shell
python setup.py build install
```

## Quick Start

```python
from msp_pupil_labs import PupilCaptureSource

eyetracking_source = PupilCaptureSource(
        address="127.0.0.1",
        port=50020
    )
```

The IP address is the address of the computer running Pupil Capture. The port is given by the [Pupil Remote plugin](https://docs.pupil-labs.com/core/software/pupil-capture/#network-plugins) of Pupil Capture and defaults to `50020`. The PupilCaptureSource module requires the [Pupil Capture](https://docs.pupil-labs.com/core/software/pupil-capture/) software to be running. This extension was tested with Pupil Capture v3.1.16.