import os
import os.path as osp

import mmcv
import pytest

from mmaction.utils import FileClient


class TestFileClient(object):

    @classmethod
    def setup_class(cls):
        cls.img_path = osp.join(osp.dirname(__file__), 'data/test.jpg')
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.img_dir = osp.join(osp.dirname(__file__), 'data/test_imgs')
        cls.text_path = osp.join(
            osp.dirname(__file__), 'data/frame_test_list.txt')
        cls.ceph_dir = 's3://linjintao.UCF-101-rawframe'
        cls.ceph_path = osp.join(cls.ceph_dir,
                                 'v_ApplyEyeMakeup_g08_c01/img_00001.jpg')
        cls.total_frames = len(os.listdir(cls.img_dir))
        cls.filename_tmpl = 'img_{:05}.jpg'

    def test_disk_backend(self):
        disk_backend = FileClient('disk')
        img_bytes = disk_backend.get(self.img_path)
        cur_frame = mmcv.imfrombytes(img_bytes)
        assert open(self.img_path, 'rb').read() == img_bytes
        assert cur_frame.shape == (240, 320, 3)

        value_buf = disk_backend.get_text(self.text_path)
        assert open(self.text_path, 'r').read() == value_buf

    def test_ceph_backend(self):
        try:
            import ceph  # NOQA: F401
        except ImportError:
            from unittest.mock import Mock
            from mmaction.utils.file_client import CephBackend
            use_mock = True
            mock = Mock()

            def Get(filepath):
                if filepath == self.ceph_path:
                    return open(self.img_path, 'rb').read()

            mock.Get = Get

            class MockCephBackend:

                def __init__(self):
                    self._client = mock

            CephBackend.__init__ = MockCephBackend.__init__

        ceph_backend = FileClient('ceph')

        with pytest.raises(NotImplementedError):
            ceph_backend.get_text(self.text_path)

        img_bytes = ceph_backend.get(self.ceph_path)
        cur_frame = mmcv.imfrombytes(img_bytes)
        if use_mock:
            assert cur_frame.shape == (240, 320, 3)
        else:
            assert cur_frame.shape == (256, 340, 3)

    def test_memcached_backend(self):
        # yapf:disable
        mc_cfg = dict(
            server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',  # noqa:E501
            client_cfg='/mnt/lustre/share/memcached_client/client.conf',
            sys_path='/mnt/lustre/share/pymc/py3')
        # yapf:enable

        try:
            import mc  # NOQA: F401
        except ImportError:
            import sys
            from unittest.mock import Mock
            from mmaction.utils.file_client import MemcachedBackend
            mock_client = Mock()
            mock_mc = Mock()

            def Get(filepath, _mc_buffer):
                return

            mock_client.Get = Get

            def pyvector(_mc_buffer):
                return

            def ConvertBuffer(_mc_buffer):
                return open(_mc_buffer, 'rb').read()

            mock_mc.pyvector = pyvector
            mock_mc.ConvertBuffer = ConvertBuffer

            class MockMemcachedBackend:

                def __init__(self, server_list_cfg, client_cfg, sys_path=None):
                    sys.path.append(sys_path)
                    self.server_list_cfg = server_list_cfg
                    self.client_cfg = client_cfg
                    self._client = mock_client
                    self._mc_buffer = mock_mc

                def get(self, filepath):
                    self._client.Get(filepath, self._mc_buffer)
                    return mock_mc.ConvertBuffer(filepath)

            MemcachedBackend.__init__ = MockMemcachedBackend.__init__
            MemcachedBackend.get = MockMemcachedBackend.get

        mc_backend = FileClient('memcached', **mc_cfg)

        with pytest.raises(NotImplementedError):
            mc_backend.get_text(self.text_path)

        img_bytes = mc_backend.get(self.img_path)
        cur_frame = mmcv.imfrombytes(img_bytes)
        assert cur_frame.shape == (240, 320, 3)

    def test_error(self):

        class TestClass1(object):
            pass

        with pytest.raises(ValueError):
            FileClient('hadoop')

        with pytest.raises(TypeError):
            FileClient().register_backend('int', 0)

        with pytest.raises(TypeError):
            FileClient().register_backend('TestClass1', TestClass1)
