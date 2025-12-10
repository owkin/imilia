"""Test the entry point of the package."""

from imilia.__main__ import main


class TestMain:
    """Test the entry point of the package."""

    def test_main(self) -> None:
        """Test the main function."""
        assert main() == "Hello, world!"
