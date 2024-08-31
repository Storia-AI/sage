"""Implementation of DataManager for GitHub issues."""

import os
from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple

import requests

from data_manager import DataManager


@dataclass
class Comment:
    """A comment on a GitHub issue."""

    url: str
    html_url: str
    body: str

    @property
    def pretty(self):
        return f"""\nComment: {self.body}"""


@dataclass
class Issue:
    """A GitHub issue."""

    url: str
    html_url: str
    title: str
    body: str
    comments: List[Comment]

    @property
    def pretty(self):
        return (
            "Issue: "
            + self.title
            + "\n"
            + self.body
            + "\n\n"
            + "\n".join([comment.pretty for comment in self.comments])
        )


class GitHubIssuesManager(DataManager):
    """Class to manage the GitHub issues of a particular repository."""

    def __init__(self, repo_id: str, max_issues: int = None):
        super().__init__(dataset_id=repo_id + "/issues")
        self.repo_id = repo_id
        self.max_issues = max_issues
        self.access_token = os.getenv("GITHUB_TOKEN")
        self.issues = []

    def download(self) -> bool:
        """Downloads all open issues from a GitHub repository (including the comments)."""
        per_page = min(self.max_issues or 100, 100)  # 100 is maximum per page
        url = f"https://api.github.com/repos/{self.repo_id}/issues?per_page={per_page}"
        while url:
            print(f"Fetching issues from {url}")
            response = self._get_page_of_issues(url)
            response.raise_for_status()
            for issue in response.json():
                if not "pull_request" in issue:
                    self.issues.append(
                        Issue(
                            url=issue["url"],
                            html_url=issue["html_url"],
                            title=issue["title"],
                            body=issue["body"],
                            comments=self._get_comments(issue["comments_url"]),
                        )
                    )
            if self.max_issues and len(self.issues) >= self.max_issues:
                break
            url = GitHubIssuesManager._get_next_link_from_header(response)
        return True

    def walk(self) -> Generator[Tuple[str, Dict], None, None]:
        """Yields a tuple of (issue_content, issue_metadata) for each GitHub issue in the repository."""
        for issue in self.issues:
            metatada = {"url": issue.html_url}
            yield issue.pretty, metatada

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

    def _get_comments(self, comments_url) -> List[Comment]:
        """Downloads all the comments associated with an issue."""
        response = requests.get(
            comments_url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        comments = []
        for comment in response.json():
            comments.append(
                Comment(
                    url=comment["url"],
                    html_url=comment["html_url"],
                    body=comment["body"],
                )
            )
        return comments
