import unittest
from unittest.mock import patch
from data_manager import GitHubRepoManager 
class TestGitHubRepoManager(unittest.TestCase):
     @patch('git.Repo.clone_from')  
    def test_clone_repo(self, mock_clone):
        """
        test the download() method of GitHubRepoManager by mocking the cloning process.
        """
        repo_manager = GitHubRepoManager(
            repo_id='Storia-AI/sage',
            local_dir='/tmp/test_repo'
        )
        result = repo_manager.download()
        mock_clone.assert_called_once_with(
            'https://github.com/Storia-AI/sage.git',  
            '/tmp/test_repo/Storia-AI/sage',          
            depth=1,                                  
            single_branch=True
        )
        self.assertTrue(result)
if __name__ == '__main__':
    unittest.main()
