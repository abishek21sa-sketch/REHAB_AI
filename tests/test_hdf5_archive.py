import numpy as np
import pytest
from rehab_ai.storage import write_sensor_session, read_sensor_session


def test_hdf5_sensor_roundtrip(tmp_path):
    t=np.arange(0,1,.01)
    channels={'imu_ax':np.sin(t),'imu_gy':np.cos(t)}
    path=tmp_path/'sessions.h5'
    receipt=write_sensor_session(path,'P1','S1',t,channels,{'source':'synthetic'})
    out=read_sensor_session(path,'P1','S1')
    assert receipt.samples==len(t)
    assert set(receipt.channels)==set(channels)
    assert np.allclose(out['channels']['imu_ax'],channels['imu_ax'])
    assert out['metadata']['source']=='synthetic'


def test_hdf5_rejects_misaligned_channel(tmp_path):
    with pytest.raises(ValueError):
        write_sensor_session(tmp_path/'x.h5','P','S',np.array([0.,1.,2.]),{'x':np.array([1.,2.])})
