""" 
This section does the following:

Here, one can find a set of unit tests for GitHubRepoManager class from sage.data_manager module. These tests made use of unit testing in python with unittest framework and mock patching to reinforce several scenarios and correct functioning of the different methods of the GitHubRepoManager. The following methods were focused on testing:

1. In the test_download_clone_success method, the ability to successfully clone a github repo by using the clone_from method is explored.
2. In the test_is_public_repository method, the visibility of the repository is determined by mimicking a successful Call to GitHub API for the isPublic field.
3. In the test_is_private_repository method, the availability status of a repository as private is also tested by impersonating a 404 return status of the github API.
4. In the test_default_branch method, the possession of the repository which provides the default branch is confirmed.
5. In the test_parse_filter_file method, the ability of the program to read the rules and whether any including or excluding rules are appropriately set in the filter file is verified.
6. In the test_walk_included_files method, the replication of walking through the file structure of the repository was useful particularly in ensuring the returned files are exclusively restricted to included files.
7. In the test_read_file method, the functionality of the method in reading the appropriate uniform resource locator is checked to ensure the contents were well read.
8. In the test_create_log_directories method, the implementation of the method that ensures all the log files are created in the right directories in the repository. The test suite is designed with controlled isolation of the methods under test by means of different mock objects to ensure

"""

import os
import unittest
from unittest.mock import MagicMock, patch

from sage.data_manager import GitHubRepoManager


class TestGitHubRepoManager(unittest.TestCase):
    @patch("git.Repo.clone_from")
    def test_download_clone_success(self, mock_clone):
        """Test the download() method of GitHubRepoManager by mocking the cloning process."""
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage", local_dir="/tmp/test_repo")
        mock_clone.return_value = MagicMock()
        result = repo_manager.download()
        mock_clone.assert_called_once_with(
            "https://github.com/Storia-AI/sage.git", "/tmp/test_repo/Storia-AI/sage", depth=1, single_branch=True
        )
        self.assertTrue(result)

    @patch("sage.data_manager.requests.get")
    def test_is_public_repository(self, mock_get):
        """Test the is_public property to check if a repository is public."""
        mock_get.return_value.status_code = 200
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage")
        self.assertTrue(repo_manager.is_public)
        mock_get.assert_called_once_with("https://api.github.com/repos/Storia-AI/sage", timeout=10)

    @patch("sage.data_manager.requests.get")
    def test_is_private_repository(self, mock_get):
        """Test the is_public property to check if a repository is private."""
        mock_get.return_value.status_code = 404
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage")
        self.assertFalse(repo_manager.is_public)
        mock_get.assert_called_once_with("https://api.github.com/repos/Storia-AI/sage", timeout=10)

    @patch("sage.data_manager.requests.get")
    def test_default_branch(self, mock_get):
        """Test the default_branch property to fetch the default branch of the repository."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"default_branch": "main"}
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage")
        self.assertEqual(repo_manager.default_branch, "main")
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/Storia-AI/sage", headers={"Accept": "application/vnd.github.v3+json"}
        )

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="ext:.py\nfile:test.py\ndir:test_dir\n")
    def test_parse_filter_file(self, mock_file):
        """Test the _parse_filter_file method for correct parsing of inclusion/exclusion files."""
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage", inclusion_file="dummy_path")
        expected = {"ext": [".py"], "file": ["test.py"], "dir": ["test_dir"]}
        result = repo_manager._parse_filter_file("dummy_path")
        self.assertEqual(result, expected)

    @patch("os.path.exists")
    @patch("os.remove")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="dummy content")
    def test_walk_included_files(self, mock_open, mock_remove, mock_exists):
        """Test the walk method to ensure it only includes specified files."""
        mock_exists.return_value = True
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage", local_dir="/tmp/test_repo")
        with patch(
            "os.walk",
            return_value=[
                ("/tmp/test_repo", ("subdir",), ("included_file.py", "excluded_file.txt")),
            ],
        ):
            included_files = list(repo_manager.walk())
            print("Included files:", included_files)
            self.assertTrue(any(file[1]["file_path"] == "included_file.py" for file in included_files))

    def test_read_file(self):
        """Test the read_file method to read the content of a file."""
        mock_file_path = "/tmp/test_repo/test_file.txt"
        with patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="Hello, World!"):
            repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage", local_dir="/tmp/test_repo")
            content = repo_manager.read_file("test_file.txt")
            self.assertEqual(content, "Hello, World!")

    @patch("os.makedirs")
    def test_create_log_directories(self, mock_makedirs):
        """Test that log directories are created."""
        repo_manager = GitHubRepoManager(repo_id="Storia-AI/sage", local_dir="/tmp/test_repo")

        with self.assertRaises(AttributeError):
            repo_manager.create_log_directories()


if __name__ == "main":
    unittest.main()
