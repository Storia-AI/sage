"""Utility classes to maniuplate GitHub repositories."""

import logging
import os
from abc import abstractmethod
from functools import cached_property
from typing import Any, Dict, Generator, Tuple

import requests
from git import GitCommandError, Repo


class DataManager:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    @abstractmethod
    def download(self) -> bool:
        """Downloads the data from a remote location."""

    @abstractmethod
    def walk(self) -> Generator[Tuple[Any, Dict], None, None]:
        """Yields a tuple of (data, metadata) for each data item in the dataset."""


class GitHubRepoManager(DataManager):
    """Class to manage a local clone of a GitHub repository."""

    def __init__(
        self,
        repo_id: str,
        local_dir: str = None,
        inclusion_file: str = None,
        exclusion_file: str = None,
    ):
        """
        Args:
            repo_id: The identifier of the repository in owner/repo format, e.g. "Storia-AI/sage".
            local_dir: The local directory where the repository will be cloned.
            inclusion_file: A file with a lists of files/directories/extensions to include. Each line must be in one of
                the following formats: "ext:.my-extension", "file:my-file.py", or "dir:my-directory".
            exclusion_file: A file with a lists of files/directories/extensions to exclude. Each line must be in one of
                the following formats: "ext:.my-extension", "file:my-file.py", or "dir:my-directory".
        """
        super().__init__(dataset_id=repo_id)
        self.repo_id = repo_id

        self.local_dir = local_dir or "/tmp/"
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        self.local_path = os.path.join(self.local_dir, repo_id)

        self.log_dir = os.path.join(self.local_dir, "logs", repo_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.access_token = os.getenv("GITHUB_TOKEN")

        if inclusion_file and exclusion_file:
            raise ValueError("Only one of inclusion_file or exclusion_file should be provided.")

        self.inclusions = self._parse_filter_file(inclusion_file) if inclusion_file else None
        self.exclusions = self._parse_filter_file(exclusion_file) if exclusion_file else None

    @cached_property
    def is_public(self) -> bool:
        """Checks whether a GitHub repository is publicly visible."""
        response = requests.get(f"https://api.github.com/repos/{self.repo_id}", timeout=10)
        # Note that the response will be 404 for both private and non-existent repos.
        return response.status_code == 200

    @cached_property
    def default_branch(self) -> str:
        """Fetches the default branch of the repository from GitHub."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.access_token:
            headers["Authorization"] = f"token {self.access_token}"

        response = requests.get(f"https://api.github.com/repos/{self.repo_id}", headers=headers)
        if response.status_code == 200:
            branch = response.json().get("default_branch", "main")
        else:
            # This happens sometimes when we exceed the Github rate limit. The best bet in this case is to assume the
            # most common naming for the default branch ("main").
            logging.warn(f"Unable to fetch default branch for {self.repo_id}: {response.text}")
            branch = "main"
        return branch

    def download(self) -> bool:
        """Clones the repository to the local directory, if it's not already cloned."""
        if os.path.exists(self.local_path):
            # The repository is already cloned.
            return True

        if not self.is_public and not self.access_token:
            raise ValueError(f"Repo {self.repo_id} is private or doesn't exist.")

        if self.access_token:
            clone_url = f"https://{self.access_token}@github.com/{self.repo_id}.git"
        else:
            clone_url = f"https://github.com/{self.repo_id}.git"

        try:
            Repo.clone_from(clone_url, self.local_path, depth=1, single_branch=True)
        except GitCommandError as e:
            logging.error("Unable to clone %s from %s. Error: %s", self.repo_id, clone_url, e)
            return False
        return True

    def _parse_filter_file(self, file_path: str) -> bool:
        """Parses a file with files/directories/extensions to include/exclude.

        Lines are expected to be in the format:
        # Comment that will be ignored, or
        ext:.my-extension, or
        file:my-file.py, or
        dir:my-directory
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        parsed_data = {"ext": [], "file": [], "dir": []}
        for line in lines:
            if line.startswith("#"):
                # This is a comment line.
                continue
            key, value = line.strip().split(":")
            if key in parsed_data:
                parsed_data[key].append(value)
            else:
                logging.error("Unrecognized key in line: %s. Skipping.", line)

        return parsed_data

    def _should_include(self, file_path: str) -> bool:
        """Checks whether the file should be indexed."""
        # Exclude symlinks.
        if os.path.islink(file_path):
            return False

        # Exclude hidden files and directories.
        if any(part.startswith(".") for part in file_path.split(os.path.sep)):
            return False

        if not self.inclusions and not self.exclusions:
            return True

        # Filter based on file extensions, file names and directory names.
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        file_name = os.path.basename(file_path)
        dirs = os.path.dirname(file_path).split("/")

        if self.inclusions:
            return (
                extension in self.inclusions.get("ext", [])
                or file_name in self.inclusions.get("file", [])
                or any(d in dirs for d in self.inclusions.get("dir", []))
            )
        elif self.exclusions:
            return (
                extension not in self.exclusions.get("ext", [])
                and file_name not in self.exclusions.get("file", [])
                and all(d not in dirs for d in self.exclusions.get("dir", []))
            )
        return True

    def walk(self) -> Generator[Tuple[Any, Dict], None, None]:
        """Walks the local repository path and yields a tuple of (content, metadata) for each file.
        The filepath is relative to the root of the repository (e.g. "org/repo/your/file/path.py").

        Args:
            included_extensions: Optional set of extensions to include.
            excluded_extensions: Optional set of extensions to exclude.
        """
        # We will keep apending to these files during the iteration, so we need to clear them first.
        repo_name = self.repo_id.replace("/", "_")
        included_log_file = os.path.join(self.log_dir, f"included_{repo_name}.txt")
        excluded_log_file = os.path.join(self.log_dir, f"excluded_{repo_name}.txt")
        if os.path.exists(included_log_file):
            os.remove(included_log_file)
            logging.info("Logging included files at %s", included_log_file)
        if os.path.exists(excluded_log_file):
            os.remove(excluded_log_file)
            logging.info("Logging excluded files at %s", excluded_log_file)

        for root, _, files in os.walk(self.local_path):
            file_paths = [os.path.join(root, file) for file in files]
            included_file_paths = [f for f in file_paths if self._should_include(f)]

            with open(included_log_file, "a") as f:
                for path in included_file_paths:
                    f.write(path + "\n")

            excluded_file_paths = set(file_paths).difference(set(included_file_paths))
            with open(excluded_log_file, "a") as f:
                for path in excluded_file_paths:
                    f.write(path + "\n")

            for file_path in included_file_paths:
                with open(file_path, "r") as f:
                    try:
                        contents = f.read()
                    except UnicodeDecodeError:
                        logging.warning("Unable to decode file %s. Skipping.", file_path)
                        continue
                    relative_file_path = file_path[len(self.local_dir) + 1 :]
                    metadata = {
                        "file_path": relative_file_path,
                        "url": self.url_for_file(relative_file_path),
                    }
                    yield contents, metadata

    def url_for_file(self, file_path: str) -> str:
        """Converts a repository file path to a GitHub link."""
        file_path = file_path[len(self.repo_id) + 1 :]
        return f"https://github.com/{self.repo_id}/blob/{self.default_branch}/{file_path}"
