"""Tests for /files route and file serving functionality."""

import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.files import format_file_size, get_file_icon, list_files_in_directory, render_files_html
from app.main import app


@pytest.fixture
def temp_files_dir(monkeypatch):
    """Creates a temporary files directory for isolated testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        settings = get_settings()
        monkeypatch.setattr(settings, "files_dir", tmp_path)
        yield tmp_path


@pytest.fixture
def client():
    """Provides a TestClient for app."""
    return TestClient(app)


def test_format_file_size():
    """Validates human-readable file size formatting."""
    assert format_file_size(0) == "0 B"
    assert format_file_size(500) == "500 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    assert format_file_size(1073741824) == "1.0 GB"


def test_get_file_icon():
    """Validates icon selection based on file extension."""
    assert get_file_icon("doc.txt") == "📄"
    assert get_file_icon("photo.png") == "🖼️"
    assert get_file_icon("report.pdf") == "📕"
    assert get_file_icon("script.py") == "💻"
    assert get_file_icon("archive.zip") == "📦"
    assert get_file_icon("audio.mp3") == "🎵"
    assert get_file_icon("video.mp4") == "🎬"
    assert get_file_icon("data.csv") == "📊"
    assert get_file_icon("unknown.xyz") == "📎"


def test_files_route_empty_directory(temp_files_dir, client):
    """GET /files returns 200 HTML with empty state when no files exist."""
    response = client.get("/files")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Files Browser" in response.text
    assert "0 files" in response.text
    assert "No files found" in response.text


def test_files_route_trailing_slash(temp_files_dir, client):
    """GET /files/ returns 200 HTML matching /files."""
    response = client.get("/files/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Files Browser" in response.text


def test_files_route_with_files(temp_files_dir, client):
    """GET /files lists all files with clickable links."""
    # Create test files
    file1 = temp_files_dir / "notes.txt"
    file1.write_text("Meeting notes content")
    file2 = temp_files_dir / "image.png"
    file2.write_bytes(b"\x89PNG\r\n\x1a\n")

    response = client.get("/files")
    assert response.status_code == 200
    assert "2 files" in response.text
    assert "notes.txt" in response.text
    assert "image.png" in response.text
    assert 'href="/files/notes.txt"' in response.text
    assert 'href="/files/image.png"' in response.text
    assert 'target="_blank"' in response.text


def test_files_route_with_nested_directories(temp_files_dir, client):
    """GET /files lists nested files and handles URL encoding."""
    sub_dir = temp_files_dir / "sub folder"
    sub_dir.mkdir()
    nested_file = sub_dir / "nested note.txt"
    nested_file.write_text("Nested content")

    response = client.get("/files")
    assert response.status_code == 200
    assert "nested note.txt" in response.text
    assert "sub folder/nested note.txt" in response.text
    assert 'href="/files/sub%20folder/nested%20note.txt"' in response.text


def test_open_file_success(temp_files_dir, client):
    """GET /files/{filename} serves the file content inline."""
    test_file = temp_files_dir / "sample.txt"
    test_file.write_text("Hello from sample.txt")

    response = client.get("/files/sample.txt")
    assert response.status_code == 200
    assert response.text == "Hello from sample.txt"
    assert "inline" in response.headers.get("content-disposition", "")
    assert 'filename="sample.txt"' in response.headers.get("content-disposition", "")


def test_open_nested_file_success(temp_files_dir, client):
    """GET /files/{filepath:path} serves nested files."""
    sub_dir = temp_files_dir / "docs"
    sub_dir.mkdir()
    doc_file = sub_dir / "readme.md"
    doc_file.write_text("# Documentation")

    response = client.get("/files/docs/readme.md")
    assert response.status_code == 200
    assert response.text == "# Documentation"


def test_open_file_not_found(temp_files_dir, client):
    """GET /files/{nonexistent} returns 404."""
    response = client.get("/files/doesnotexist.txt")
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_path_traversal_prevention(temp_files_dir, client):
    """Attempts to traverse outside files_dir return 404."""
    # Attempt parent directory traversal
    response = client.get("/files/../app/main.py")
    assert response.status_code == 404

    # Attempt encoded traversal
    response = client.get("/files/%2e%2e%2fapp/main.py")
    assert response.status_code == 404
