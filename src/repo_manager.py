"""Utility classes to maniuplate GitHub repositories."""

import logging
import os
from functools import cached_property

import requests
from git import GitCommandError, Repo


class RepoManager:
    """Class to manage a local clone of a GitHub repository."""

    def __init__(
        self,
        repo_id: str,
        local_dir: str = None,
        included_extensions: set = None,
        excluded_extensions: set = None,
    ):
        """
        Args:
            repo_id: The identifier of the repository in owner/repo format, e.g. "Storia-AI/repo2vec".
            local_dir: The local directory where the repository will be cloned.
        """
        self.repo_id = repo_id
        self.local_dir = local_dir or "/tmp/"
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        self.local_path = os.path.join(self.local_dir, repo_id)
        self.access_token = os.getenv("GITHUB_TOKEN")
        self.included_extensions = included_extensions
        self.excluded_extensions = excluded_extensions

    @cached_property
    def is_public(self) -> bool:
        """Checks whether a GitHub repository is publicly visible."""
        response = requests.get(
            f"https://api.github.com/repos/{self.repo_id}", timeout=10
        )
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

        response = requests.get(
            f"https://api.github.com/repos/{self.repo_id}", headers=headers
        )
        if response.status_code == 200:
            branch = response.json().get("default_branch", "main")
        else:
            # This happens sometimes when we exceed the Github rate limit. The best bet in this case is to assume the
            # most common naming for the default branch ("main").
            logging.warn(
                f"Unable to fetch default branch for {self.repo_id}: {response.text}"
            )
            branch = "main"
        return branch

    def clone(self) -> bool:
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
            logging.error(
                "Unable to clone %s from %s. Error: %s", self.repo_id, clone_url, e
            )
            return False
        return True

    def _should_include(self, file_path: str) -> bool:
        """Checks whether the file should be indexed, based on the included and excluded extensions."""
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        if self.included_extensions and extension not in self.included_extensions:
            return False
        if self.excluded_extensions and extension in self.excluded_extensions:
            return False
        # Exclude hidden files and directories.
        if any(part.startswith(".") for part in file_path.split(os.path.sep)):
            return False
        return True

    def walk(self, log_dir: str = None):
        """Walks the local repository path and yields a tuple of (filepath, content) for each file.
        The filepath is relative to the root of the repository (e.g. "org/repo/your/file/path.py").

        Args:
            included_extensions: Optional set of extensions to include.
            excluded_extensions: Optional set of extensions to exclude.
            log_dir: Optional directory where to log the included and excluded files.
        """
        # We will keep apending to these files during the iteration, so we need to clear them first.
        if log_dir:
            repo_name = self.repo_id.replace("/", "_")
            included_log_file = os.path.join(log_dir, f"included_{repo_name}.txt")
            excluded_log_file = os.path.join(log_dir, f"excluded_{repo_name}.txt")
            if os.path.exists(included_log_file):
                os.remove(included_log_file)
            if os.path.exists(excluded_log_file):
                os.remove(excluded_log_file)

        for root, _, files in os.walk(self.local_path):
            file_paths = [os.path.join(root, file) for file in files]
            included_file_paths = [f for f in file_paths if self._should_include(f)]

            if log_dir:
                with open(included_log_file, "a") as f:
                    for path in included_file_paths:
                        f.write(path + "\n")

                excluded_file_paths = set(file_paths).difference(
                    set(included_file_paths)
                )
                with open(excluded_log_file, "a") as f:
                    for path in excluded_file_paths:
                        f.write(path + "\n")

            for file_path in included_file_paths:
                with open(file_path, "r") as f:
                    try:
                        contents = f.read()
                    except UnicodeDecodeError:
                        logging.warning(
                            "Unable to decode file %s. Skipping.", file_path
                        )
                        continue
                    yield file_path[len(self.local_dir) + 1 :], contents

    def github_link_for_file(self, file_path: str) -> str:
        """Converts a repository file path to a GitHub link."""
        file_path = file_path[len(self.repo_id) :]
        return (
            f"https://github.com/{self.repo_id}/blob/{self.default_branch}/{file_path}"
        )
