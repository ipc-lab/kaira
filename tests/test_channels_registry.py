import pytest

from kaira.channels.registry import ChannelRegistry


def test_register_channel():
    """Test registering a new channel."""

    class DummyChannel:
        pass

    ChannelRegistry.register("dummy", DummyChannel)
    assert "dummy" in ChannelRegistry._channels
    assert ChannelRegistry._channels["dummy"] is DummyChannel


def test_get_channel():
    """Test getting a registered channel."""

    class DummyChannel:
        pass

    ChannelRegistry.register("dummy", DummyChannel)
    channel_class = ChannelRegistry.get("dummy")
    assert channel_class is DummyChannel


def test_get_nonexistent_channel():
    """Test getting a non-existent channel."""
    with pytest.raises(KeyError):
        ChannelRegistry.get("nonexistent")


def test_create_channel():
    """Test creating a channel instance."""

    class DummyChannel:
        def __init__(self, param):
            self.param = param

    ChannelRegistry.register("dummy", DummyChannel)
    channel_instance = ChannelRegistry.create("dummy", param=42)
    assert isinstance(channel_instance, DummyChannel)
    assert channel_instance.param == 42


def test_list_channels():
    """Test listing all registered channels."""

    class DummyChannel1:
        pass

    class DummyChannel2:
        pass

    ChannelRegistry.register("dummy1", DummyChannel1)
    ChannelRegistry.register("dummy2", DummyChannel2)
    channels = ChannelRegistry.list_channels()
    assert "dummy1" in channels
    assert "dummy2" in channels
