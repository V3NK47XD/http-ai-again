"""Helper functions and HTML renderer for the /files route."""

from datetime import datetime, timezone
import html
from pathlib import Path
import urllib.parse
from app.config import get_settings


def get_files_directory() -> Path:
    """Returns the configured files directory and ensures it exists."""
    directory = get_settings().files_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_file_size(num_bytes: int) -> str:
    """Formats raw byte count into human-readable string (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size = float(num_bytes)
    for unit in ["KB", "MB", "GB", "TB"]:
        size /= 1024.0
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} PB"


def get_file_icon(filename: str) -> str:
    """Returns an icon indicator based on the file extension."""
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md", ".rst", ".log"}:
        return "📄"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".ico"}:
        return "🖼️"
    if ext in {".pdf"}:
        return "📕"
    if ext in {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".sh", ".sql"}:
        return "💻"
    if ext in {".zip", ".tar", ".gz", ".7z", ".rar", ".bz2"}:
        return "📦"
    if ext in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}:
        return "🎵"
    if ext in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
        return "🎬"
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        return "📊"
    return "📎"


def list_files_in_directory(files_dir: Path) -> list[dict]:
    """Recursively lists all files in the given directory with metadata."""
    if not files_dir.exists():
        return []

    files_list = []
    # Search all items inside the files folder
    for path in sorted(files_dir.rglob("*")):
        if path.is_file():
            try:
                rel_path = path.relative_to(files_dir)
                rel_posix = rel_path.as_posix()
                # URL quote the relative POSIX path for links
                quoted_url = urllib.parse.quote(rel_posix, safe="/")
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                files_list.append({
                    "name": path.name,
                    "rel_path": rel_posix,
                    "url": f"/files/{quoted_url}",
                    "size_bytes": stat.st_size,
                    "size_formatted": format_file_size(stat.st_size),
                    "modified": mtime.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "icon": get_file_icon(path.name),
                })
            except (OSError, ValueError):
                continue

    # Sort files by relative path
    files_list.sort(key=lambda item: item["rel_path"].lower())
    return files_list


def render_files_html(files: list[dict]) -> str:
    """Generates a responsive and modern HTML page listing all files with open links."""
    file_count = len(files)
    count_badge = f"{file_count} file{'s' if file_count != 1 else ''}"

    if not files:
        rows_or_empty = """
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <h3>No files found</h3>
            <p>The <code>files/</code> directory is currently empty. Add files to the directory to see and open them here.</p>
        </div>
        """
    else:
        rows = []
        for f in files:
            safe_name = html.escape(f["name"])
            safe_rel = html.escape(f["rel_path"])
            safe_url = html.escape(f["url"])
            safe_size = html.escape(f["size_formatted"])
            safe_mod = html.escape(f["modified"])
            icon = f["icon"]

            rows.append(f"""
            <tr class="file-row" data-name="{safe_name.lower()}" data-path="{safe_rel.lower()}">
                <td class="name-cell">
                    <span class="file-icon" aria-hidden="true">{icon}</span>
                    <a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="file-link" title="Open {safe_name}">
                        {safe_name}
                    </a>
                    {f'<div class="sub-path">{safe_rel}</div>' if safe_rel != safe_name else ''}
                </td>
                <td class="size-cell">{safe_size}</td>
                <td class="mod-cell">{safe_mod}</td>
                <td class="action-cell">
                    <a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="btn-open" title="Open file in new tab">
                        Open
                    </a>
                </td>
            </tr>
            """)
        table_body = "\n".join(rows)
        rows_or_empty = f"""
        <div class="table-responsive">
            <table class="files-table">
                <thead>
                    <tr>
                        <th scope="col">Name</th>
                        <th scope="col" style="width: 120px;">Size</th>
                        <th scope="col" style="width: 200px;">Modified</th>
                        <th scope="col" style="width: 90px; text-align: center;">Action</th>
                    </tr>
                </thead>
                <tbody id="files-list">
                    {table_body}
                </tbody>
            </table>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Files Explorer</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --surface-hover: #334155;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --accent-bg: rgba(56, 189, 248, 0.12);
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --radius: 8px;
            --transition: all 0.15s ease-in-out;
        }}

        @media (prefers-color-scheme: light) {{
            :root {{
                --bg-color: #f8fafc;
                --surface-color: #ffffff;
                --surface-hover: #f1f5f9;
                --border-color: #e2e8f0;
                --text-primary: #0f172a;
                --text-secondary: #64748b;
                --accent-color: #0284c7;
                --accent-hover: #0369a1;
                --accent-bg: #e0f2fe;
            }}
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 2rem 1rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .title-group {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .badge {{
            background: var(--accent-bg);
            color: var(--accent-color);
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .search-bar {{
            position: relative;
            margin-bottom: 1.25rem;
        }}

        .search-input {{
            width: 100%;
            padding: 0.65rem 1rem 0.65rem 2.25rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            background-color: var(--surface-color);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: var(--transition);
        }}

        .search-input:focus {{
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px var(--accent-bg);
        }}

        .search-icon {{
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            font-size: 0.9rem;
            pointer-events: none;
        }}

        .card {{
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .table-responsive {{
            overflow-x: auto;
        }}

        .files-table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.925rem;
        }}

        .files-table th {{
            padding: 0.85rem 1rem;
            background-color: var(--surface-hover);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border-color);
        }}

        .files-table td {{
            padding: 0.85rem 1rem;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }}

        .file-row:last-child td {{
            border-bottom: none;
        }}

        .file-row:hover {{
            background-color: var(--surface-hover);
        }}

        .name-cell {{
            font-weight: 500;
        }}

        .file-icon {{
            margin-right: 0.5rem;
            font-size: 1.1rem;
            vertical-align: middle;
        }}

        .file-link {{
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 500;
            transition: var(--transition);
        }}

        .file-link:hover {{
            color: var(--accent-color);
            text-decoration: underline;
        }}

        .sub-path {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
            margin-left: 1.6rem;
        }}

        .size-cell, .mod-cell {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            white-space: nowrap;
        }}

        .action-cell {{
            text-align: center;
        }}

        .btn-open {{
            display: inline-block;
            background-color: var(--accent-bg);
            color: var(--accent-color);
            padding: 0.35rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            text-decoration: none;
            transition: var(--transition);
        }}

        .btn-open:hover {{
            background-color: var(--accent-color);
            color: #ffffff;
        }}

        .empty-state {{
            padding: 3.5rem 1.5rem;
            text-align: center;
        }}

        .empty-icon {{
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.7;
        }}

        .empty-state h3 {{
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }}

        .empty-state p {{
            color: var(--text-secondary);
            max-width: 420px;
            margin: 0 auto;
            font-size: 0.95rem;
        }}

        .empty-state code {{
            background: var(--surface-hover);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: monospace;
        }}

        footer {{
            margin-top: 2rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-group">
                <h1><span>📁</span> Files Browser</h1>
                <span class="badge" id="file-counter">{count_badge}</span>
            </div>
            <div>
                <a href="/ping" class="file-link" style="font-size: 0.85rem; color: var(--text-secondary);">Service Status</a>
            </div>
        </header>

        {'<div class="search-bar"><span class="search-icon">🔍</span><input type="text" id="filter-input" class="search-input" placeholder="Filter files by name or path..."></div>' if files else ''}

        <div class="card">
            {rows_or_empty}
        </div>

        <footer>
            <span>Files served from <code>files/</code> folder</span>
        </footer>
    </div>

    <script>
        const input = document.getElementById('filter-input');
        if (input) {{
            input.addEventListener('input', function() {{
                const filter = this.value.trim().toLowerCase();
                const rows = document.querySelectorAll('.file-row');
                let visibleCount = 0;
                rows.forEach(row => {{
                    const name = row.getAttribute('data-name') || '';
                    const path = row.getAttribute('data-path') || '';
                    if (name.includes(filter) || path.includes(filter)) {{
                        row.style.display = '';
                        visibleCount++;
                    }} else {{
                        row.style.display = 'none';
                    }}
                }});
                const counter = document.getElementById('file-counter');
                if (counter) {{
                    counter.textContent = visibleCount + ' file' + (visibleCount !== 1 ? 's' : '');
                }}
            }});
        }}
    </script>
</body>
</html>
"""
