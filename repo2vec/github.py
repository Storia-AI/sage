"""GitHub-specific implementations for DataManager and Chunker."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple

import requests
import tiktoken

from repo2vec.chunker import Chunk, Chunker
from repo2vec.data_manager import DataManager

tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class GitHubIssueComment:
    """A comment on a GitHub issue."""

    url: str
    html_url: str
    body: str

    @property
    def pretty(self):
        return f"""## Comment: {self.body}"""


@dataclass
class GitHubIssue:
    """A GitHub issue."""

    url: str
    html_url: str
    title: str
    body: str
    comments: List[GitHubIssueComment]

    @property
    def pretty(self):
        # Do not include the comments.
        return f"# Issue: {self.title}\n{self.body}"


class GitHubIssuesManager(DataManager):
    """Class to manage the GitHub issues of a particular repository."""

    def __init__(self, repo_id: str, index_comments: bool = False, max_issues: int = None):
        super().__init__(dataset_id=repo_id + "/issues")
        self.repo_id = repo_id
        self.index_comments = index_comments
        self.max_issues = max_issues
        self.access_token = os.getenv("GITHUB_TOKEN")
        if not self.access_token:
            raise ValueError("Please set the GITHUB_TOKEN environment variable when indexing GitHub issues.")
        self.issues = []

    def download(self) -> bool:
        """Downloads all open issues from a GitHub repository (including the comments)."""
        per_page = min(self.max_issues or 100, 100)  # 100 is maximum per page
        url = f"https://api.github.com/repos/{self.repo_id}/issues?per_page={per_page}"
        while url:
            logging.info(f"Fetching issues from {url}")
            response = self._get_page_of_issues(url)
            response.raise_for_status()
            for issue in response.json():
                if not "pull_request" in issue:
                    self.issues.append(
                        GitHubIssue(
                            url=issue["url"],
                            html_url=issue["html_url"],
                            title=issue["title"],
                            # When there's no body, issue["body"] is None.
                            body=issue["body"] or "",
                            comments=self._get_comments(issue["comments_url"]) if self.index_comments else [],
                        )
                    )
            if self.max_issues and len(self.issues) >= self.max_issues:
                break
            url = GitHubIssuesManager._get_next_link_from_header(response)
        return True

    def walk(self) -> Generator[Tuple[Any, Dict], None, None]:
        """Yields a tuple of (issue_content, issue_metadata) for each GitHub issue in the repository."""
        for issue in self.issues:
            yield issue, {}  # empty metadata

    @staticmethod
    def _get_next_link_from_header(response):
        """
        Given a response from a paginated request, extracts the URL of the next page.

        Example:
            response.headers.get("link") = '<https://api.github.com/repositories/2503910/issues?per_page=10&page=2>; rel="next", <https://api.github.com/repositories/2503910/issues?per_page=10&page=2>; rel="last"'
            get_next_link_from_header(response) = 'https://api.github.com/repositories/2503910/issues?per_page=10&page=2'
        """
        link_header = response.headers.get("link")
        if link_header:
            links = link_header.split(", ")
            for link in links:
                url, rel = link.split("; ")
                url = url[1:-1]  # The URL is enclosed in angle brackets
                rel = rel[5:-1]  # e.g. rel="next" -> next
                if rel == "next":
                    return url
        return None

    def _get_page_of_issues(self, url):
        """Downloads a single page of issues. Note that GitHub uses pagination for long lists of objects."""
        return requests.get(
            url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def _get_comments(self, comments_url) -> List[GitHubIssueComment]:
        """Downloads all the comments associated with an issue; returns an empty list if the request times out."""
        try:
            response = requests.get(
                comments_url,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        except requests.exceptions.ConnectTimeout:
            logging.warn(f"Timeout fetching comments from {comments_url}")
            return []
        comments = []
        for comment in response.json():
            comments.append(
                GitHubIssueComment(
                    url=comment["url"],
                    html_url=comment["html_url"],
                    body=comment["body"],
                )
            )
        return comments


@dataclass
class IssueChunk(Chunk):
    """A chunk form a GitHub issue with a contiguous (sub)set of comments.

    Note that, in comparison to FileChunk, its properties are not cached. We want to allow fields to be changed in place
    and have e.g. the token count be recomputed. Compared to files, GitHub issues are typically smaller, so the overhead
    is less problematic.
    """

    issue: GitHubIssue
    start_comment: int
    end_comment: int  # exclusive

    @property
    def content(self) -> str:
        """The title of the issue, followed by the comments in the chunk."""
        if self.start_comment == 0:
            # This is the first subsequence of comments. We'll include the entire body of the issue.
            issue_str = self.issue.pretty
        else:
            # This is a middle subsequence of comments. We'll only include the title of the issue.
            issue_str = f"# Issue: {self.issue.title}"
        # Now add the comments themselves.
        comments = self.issue.comments[self.start_comment : self.end_comment]
        comments_str = "\n\n".join([comment.pretty for comment in comments])
        return issue_str + "\n\n" + comments_str

    @property
    def metadata(self):
        """Converts the chunk to a dictionary that can be passed to a vector store."""
        return {
            "id": f"{self.issue.html_url}_{self.start_comment}_{self.end_comment}",
            "url": self.issue.html_url,
            "start_comment": self.start_comment,
            "end_comment": self.end_comment,
            # Note to developer: When choosing a large chunk size, you might exceed the vector store's metadata
            # size limit. In that case, you can simply store the start/end comment indices above, and fetch the
            # content of the issue on demand from the URL.
            "text": self.content,
        }

    @property
    def num_tokens(self):
        """Number of tokens in this chunk."""
        return len(tokenizer.encode(self.content, disallowed_special=()))


class GitHubIssuesChunker(Chunker):
    """Chunks a GitHub issue into smaller pieces of contiguous (sub)sets of comments."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def chunk(self, content: Any, metadata: Dict) -> List[Chunk]:
        """Chunks a GitHub issue into subsequences of comments."""
        del metadata  # The metadata of the input issue is unused.

        issue = content  # Rename for clarity.
        if not isinstance(issue, GitHubIssue):
            raise ValueError(f"Expected a GitHubIssue, got {type(issue)}.")

        chunks = []

        # First, create a chunk for the body of the issue. If it's too long, then truncate it.
        if len(tokenizer.encode(issue.pretty, disallowed_special=())) > self.max_tokens:
            title_len = len(tokenizer.encode(issue.title, disallowed_special=()))
            target_body_len = self.max_tokens - title_len - 20  # 20 for buffer
            trimmed_body = tokenizer.decode(tokenizer.encode(issue.body, disallowed_special=())[:target_body_len])
            trimmed_issue = GitHubIssue(
                url=issue.url,
                html_url=issue.html_url,
                title=issue.title,
                body=trimmed_body,
                comments=issue.comments,
            )
            issue_body_chunk = IssueChunk(trimmed_issue, 0, 0)
        else:
            issue_body_chunk = IssueChunk(issue, 0, 0)

        chunks.append(issue_body_chunk)

        for comment_idx, comment in enumerate(issue.comments):
            # This is just approximate, because when we actually add a comment to the chunk there might be some extra
            # tokens, like a "Comment:" prefix.
            approx_comment_size = len(tokenizer.encode(comment.body, disallowed_special=())) + 20  # 20 for buffer

            if chunks[-1].num_tokens + approx_comment_size > self.max_tokens:
                # Create a new chunk starting from this comment.
                chunks.append(
                    IssueChunk(
                        issue=issue,
                        start_comment=comment_idx,
                        end_comment=comment_idx + 1,
                    )
                )
            else:
                # Add the comment to the existing chunk.
                chunks[-1].end_comment = comment_idx + 1
        return chunks
