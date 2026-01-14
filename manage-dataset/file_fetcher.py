import os
from typing import Optional


class FileFetcher:
    def __init__(
        self,
        root: str,
        name: Optional[str] = None,
        version: Optional[str] = None,
        file: Optional[str] = None,
    ):
        self.root = root
        self.name = name
        self.version = version
        self.file = file

    def set_options(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        file: Optional[str] = None,
    ):
        if name is not None:
            self.name = name
        if version is not None:
            self.version = version
        if file is not None:
            self.file = file

    def get_file_path(self) -> Optional[str]:
        if not all([self.root, self.name, self.version, self.file]):
            return None
        path = os.path.join(self.root, self.name, self.version, self.file)
        return path if os.path.isfile(path) else None

    def get_file_contents(self) -> Optional[str]:
        path = self.get_file_path()
        if path is None:
            return None
        with open(path, "r") as f:
            return f.read()
