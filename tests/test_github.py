"""
Unit tests for the classes under github.py.

These tests ensure the functionality of GitHub issue management,
issue chunking, and issue comments, including happy-path scenarios.

Dependencies:
pip install pytest
pip install pytest-mock
"""

import pytest
import requests
from unittest.mock import patch, MagicMock
from sage.github import GitHubIssuesManager, GitHubIssue, GitHubIssueComment, GitHubIssuesChunker, IssueChunk

class TestGitHubIssuesManager:
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Fixture to create a GitHubIssuesManager instance for each test."""
        self.github_manager = GitHubIssuesManager(repo_id="random/random-repo", access_token="fake_token")

    @staticmethod
    def mock_issue_response():
        """A mock response for GitHub issues."""
        return MagicMock(json=lambda: [
            {
                "url": "https://api.github.com/repos/random/random-repo/issues/1",
                "html_url": "https://github.com/random/random-repo/issues/1",
                "title": "Found a bug", 
                "body": "I'm having a problem with this.",
                "comments_url": "https://api.github.com/repos/random/random-repo/issues/1/comments",
                "comments": 2
            }
        ])
    
    @staticmethod
    def mock_comment_response():
        """Create a mock response for GitHub issue comments."""
        return MagicMock(json=lambda: [
            {
                "url": "https://api.github.com/repos/random/random-repo/issues/comments/1",
                "html_url": "https://github.com/random/random-repo/issues/comments/1",
                "body": "This is a comment."
            }
        ])

    @patch('github.requests.get')
    def test_download_issues(self, mock_get):
        """Test the download of issues from GitHub."""
        mock_get.side_effect = [self.mock_issue_response(), self.mock_comment_response()]
        
        self.github_manager.download()

        assert len(self.github_manager.issues) == 1
        assert self.github_manager.issues[0].title == "Found a bug"
        assert self.github_manager.issues[0].body == "I'm having a problem with this."
        assert self.github_manager.issues[0].url == "https://api.github.com/repos/random/random-repo/issues/1"

    @patch('github.requests.get')
    def test_walk_issues(self, mock_get):
        """Test the walking through downloaded issues."""
        self.github_manager.issues = [
            GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="Test Body", comments=[]),
            GitHubIssue(url="issue_url_2", html_url="html_issue_url_2", title="Another Test Issue", body="Another Test Body", comments=[]),
        ]

        issues = list(self.github_manager.walk())
        
        assert len(issues) == 2
        assert issues[0][0].title == "Test Issue"
        assert issues[1][0].title == "Another Test Issue"

    @patch('github.requests.get')
    def test_get_page_of_issues(self, mock_get):
        """Test fetching a page of issues."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "url": "https://api.github.com/repos/random/random-repo/issues/1",
                "html_url": "https://github.com/random/random-repo/issues/1",
                "title": "Found a bug", 
                "body": "I'm having a problem with this.",
                "comments_url": "https://api.github.com/repos/random/random-repo/issues/1/comments",
                "comments": 2
            }
        ]
        mock_get.return_value = mock_response

        issues = self.github_manager._get_page_of_issues("https://api.github.com/repos/random/random-repo/issues?page=1").json()

        assert len(issues) == 1  

    @patch('github.requests.get')
    def test_get_comments(self, mock_get):
        """Test retrieving comments for an issue."""
        mock_get.return_value.json.return_value = self.mock_comment_response().json()
        
        comments = self.github_manager._get_comments("comments_url")
        assert len(comments) == 1
        assert comments[0].body == "This is a comment."

    def test_chunker(self):
        """Test chunking of an issue into smaller parts."""
        issue = GitHubIssue(
            url="issue_url",
            html_url="html_issue_url",
            title="Test Issue",
            body="This is a long body of the issue that needs to be chunked.",
            comments=[
                GitHubIssueComment(url="comment_url_1", html_url="html_comment_url_1", body="First comment."),
                GitHubIssueComment(url="comment_url_2", html_url="html_comment_url_2", body="Second comment."),
            ]
        )

        chunker = GitHubIssuesChunker(max_tokens=50)
        chunks = chunker.chunk(content=issue, metadata={})

        assert len(chunks) > 0
        assert all(isinstance(chunk, IssueChunk) for chunk in chunks)
        assert chunks[0].issue.title == "Test Issue"
        assert chunks[0].start_comment == 0


class TestGitHubIssueComment:
    
    def test_initialization(self):
        """Test the initialization of the GitHubIssueComment class."""
        comment = GitHubIssueComment(url="comment_url", html_url="html_comment_url", body="Sample comment")
        assert comment.url == "comment_url"
        assert comment.html_url == "html_comment_url"
        assert comment.body == "Sample comment"

    def test_pretty_property(self):
        """Test the pretty property of the GitHubIssueComment class."""
        comment = GitHubIssueComment(url="comment_url", html_url="html_comment_url", body="Sample comment")
        assert comment.pretty == "## Comment: Sample comment"


class TestGitHubIssue:
    
    def test_initialization(self):
        """Test the initialization of the GitHubIssue class."""
        issue = GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="Test Body", comments=[])
        assert issue.url == "issue_url"
        assert issue.html_url == "html_issue_url"
        assert issue.title == "Test Issue"
        assert issue.body == "Test Body"
        assert issue.comments == []

    def test_pretty_property(self):
        """Test the pretty property of the GitHubIssue class."""
        issue = GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="Test Body", comments=[])
        assert issue.pretty == "# Issue: Test Issue\nTest Body"


class TestIssueChunk:
    
    def test_initialization(self):
        """Test the initialization of the IssueChunk class."""
        issue = GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="Test Body", comments=[])
        chunk = IssueChunk(issue=issue, start_comment=0, end_comment=1)
        assert chunk.issue == issue
        assert chunk.start_comment == 0
        assert chunk.end_comment == 1

    def test_content_property(self):
        """Test the content property of the IssueChunk class."""
        issue = GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="Test Body", comments=[])
        chunk = IssueChunk(issue=issue, start_comment=0, end_comment=1)
        assert chunk.content == "# Issue: Test Issue\nTest Body\n\n"

    def test_metadata_property(self):
        """Test the metadata property of the IssueChunk class."""
        issue = GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="Test Body", comments=[])
        chunk = IssueChunk(issue=issue, start_comment=0, end_comment=1)
        expected_metadata = {
            "id": "html_issue_url_0_1",
            "url": "html_issue_url",
            "start_comment": 0,
            "end_comment": 1,
            'text': '# Issue: Test Issue\nTest Body\n\n',
        }
        assert chunk.metadata == expected_metadata

    def test_num_tokens_property(self):
        """Test the num_tokens property of the IssueChunk class."""
        issue = GitHubIssue(url="issue_url", html_url="html_issue_url", title="Test Issue", body="This is a test body.", comments=[])
        chunk = IssueChunk(issue=issue, start_comment=0, end_comment=1)
        assert chunk.num_tokens == 12 


class TestGitHubIssuesChunker:
    
    def test_initialization(self):
        """Test the initialization of the GitHubIssuesChunker class."""
        chunker = GitHubIssuesChunker(max_tokens=50)
        assert chunker.max_tokens == 50

    def test_chunk_method(self):
        """Test the chunk method of the GitHubIssuesChunker class."""
        issue = GitHubIssue(
            url="issue_url",
            html_url="html_issue_url",
            title="Test Issue",
            body="This is a long body of the issue that needs to be chunked.",
            comments=[]
        )
        
        chunker = GitHubIssuesChunker(max_tokens=50)
        chunks = chunker.chunk(content=issue, metadata={})
        
        assert len(chunks) > 0

        assert all(isinstance(chunk, IssueChunk) for chunk in chunks)


if __name__ == '__main__':
    pytest.main()
