"""Shared pytest fixtures."""

import socket
from typing import Any

import pytest


class NetworkAccessError(RuntimeError):
    """A test tried to open a network connection."""


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test fail loudly if it tries to open a network connection.

    LLM calls are mocked in this suite. A test that forgets to mock would otherwise
    call a real provider, spend credits and send the CV out; with this guard it fails
    at the socket instead, whichever SDK it went through.
    """

    def refuse(self: socket.socket, address: Any) -> None:
        """Reject the connection attempt.

        Args:
            self: The socket that was asked to connect.
            address: The address it was asked to connect to.

        Raises:
            NetworkAccessError: Always.
        """
        raise NetworkAccessError(f"Tests must not use the network (tried {address!r}).")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
