"""
Paper TOC Splitter - MkDocs Hooks Version
Dead simple: split markdown by headers, respect hierarchy, done.
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from mkdocs.structure.files import File
from mkdocs.structure.pages import Page


# Configuration
PAPER_DIR = "paper"
PAPER_FILE = "Hill Space is All You Need.md"


def on_config(config):
    """Parse paper and generate navigation."""

    # Get paper path
    docs_dir = Path(config["docs_dir"])
    project_root = docs_dir.parent
    paper_path = project_root / PAPER_DIR / PAPER_FILE

    if not paper_path.exists():
        raise FileNotFoundError(f"Paper file not found at {paper_path}")

    print(f"Reading paper from: {paper_path}")

    # Read the paper
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections
    sections = _split_by_headers(content)

    # Store for later use
    config["_paper_sections"] = sections

    # Build navigation
    nav = _build_navigation(sections)
    config["nav"] = nav

    print(f"Generated {len(sections)} sections")

    return config


def on_files(files, config):
    """Create virtual files for each section."""

    sections = config.get("_paper_sections", [])

    for section in sections:
        virtual_file = File(
            section["filename"],
            config["docs_dir"],
            config["site_dir"],
            config["use_directory_urls"],
        )
        virtual_file._content = section["content"]
        files.append(virtual_file)

    return files


def on_page_read_source(page, config):
    """Provide content for virtual pages."""
    if hasattr(page.file, "_content"):
        return page.file._content
    return None


def on_page_markdown(markdown, page, config, files, **kwargs):
    """Process markdown content."""
    markdown = _replace_svg_with_widgets(markdown)
    markdown = _convert_obsidian_syntax(markdown)
    return markdown


def _split_by_headers(content: str) -> List[Dict]:
    """Split content by headers, preserving hierarchy."""

    lines = content.split("\n")
    sections = []

    # Find all headers
    header_lines = []
    header_pattern = r"^(#{1,6})\s+(?:([\d\.]+)\s+)?(.+)$"

    for i, line in enumerate(lines):
        match = re.match(header_pattern, line)
        if match:
            level = len(match.group(1))
            number = match.group(2) or ""
            title = match.group(3).strip()

            header_lines.append(
                {
                    "line_num": i,
                    "level": level,
                    "number": number,
                    "title": title,
                    "full_header": line,
                }
            )

    # Special handling for content before first numbered section
    # Find the first numbered section (like "1. Introduction")
    first_numbered_idx = None
    for idx, h in enumerate(header_lines):
        if h["number"]:  # This is a numbered section
            first_numbered_idx = idx
            break

    if (
        first_numbered_idx is not None
        and header_lines[first_numbered_idx]["line_num"] > 0
    ):
        # Everything before the first numbered section becomes index.md
        intro_content = "\n".join(
            lines[: header_lines[first_numbered_idx]["line_num"]]
        ).strip()
        if intro_content:
            sections.append(
                {
                    "filename": "index.md",
                    "title": "Abstract",
                    "content": intro_content,
                    "level": 0,
                    "number": "",
                }
            )

        # Start processing from the first numbered section
        start_idx = first_numbered_idx
    else:
        start_idx = 0

    # Process each header and its content
    for i in range(start_idx, len(header_lines)):
        header = header_lines[i]

        # Find content boundaries
        start = header["line_num"]
        end = (
            header_lines[i + 1]["line_num"] if i + 1 < len(header_lines) else len(lines)
        )

        # Extract content
        content = "\n".join(lines[start:end]).strip()

        # Generate filename based on section number
        if header["number"]:
            filename = f"section-{header['number'].replace('.', '-')}.md"
        else:
            # Unnumbered sections (conclusion, references, etc)
            safe_title = header["title"].lower().replace(" ", "-")
            safe_title = re.sub(r"[^a-z0-9-]", "", safe_title)
            filename = f"section-{safe_title}.md"

        sections.append(
            {
                "filename": filename,
                "title": header["title"],
                "content": content,
                "level": header["level"],
                "number": header["number"],
            }
        )

    return sections


def _build_navigation(sections: List[Dict]) -> List:
    """Build hierarchical navigation from flat section list."""

    nav = []
    stack = []  # Stack to track hierarchy

    for section in sections:
        nav_entry = {section["title"]: section["filename"]}
        level = section["level"]

        # Special case for index
        if section["filename"] == "index.md":
            nav.append(nav_entry)
            continue

        # Pop stack until we find the right parent level
        while stack and stack[-1]["level"] >= level:
            stack.pop()

        # If this is a top-level item or stack is empty
        if not stack:
            nav.append(nav_entry)
            stack.append({"item": nav_entry, "level": level, "title": section["title"]})
        else:
            # This is a child of the last item in stack
            parent = stack[-1]["item"]
            parent_title = stack[-1]["title"]

            # Convert parent from simple to nested format if needed
            if isinstance(parent[parent_title], str):
                # Parent was a simple entry, convert to nested
                parent_filename = parent[parent_title]
                parent[parent_title] = [parent_filename]

            # Add this item as a child
            parent[parent_title].append(nav_entry)
            stack.append({"item": nav_entry, "level": level, "title": section["title"]})

    return nav


def _replace_svg_with_widgets(markdown: str) -> str:
    """Replace SVG references with interactive React widgets."""

    svg_pattern = r"!\[([^\]]*)\]\(([^)]*\.svg)\)"

    def replace_svg(match):
        alt_text = match.group(1)
        svg_path = match.group(2)
        filename = Path(svg_path).stem

        widget_mapping = {
            "additive_primitive": "AdditivePrimitiveWidget",
            "matrix_multiplication": "MatrixMultiplicationWidget",
            "attention_mechanism": "AttentionWidget",
            "transformer_block": "TransformerWidget",
            "gradient_descent": "GradientDescentWidget",
            "loss_landscape": "LossLandscapeWidget",
            "hill_space_cross_sections": "HillSpaceCrossSectionsWidget",
            "exponential_primitive": "ExponentialPrimitiveWidget",
            "rotation_transformation": "RotationTransformationWidget",
            "trigonometric_product_primitive": "TrigonometricProductWidget",
        }

        widget_component = widget_mapping.get(filename)

        if widget_component:
            widget_id = f"{filename}_{abs(hash(alt_text + svg_path)) % 10000}"

            return f"""
<div class="interactive-widget" id="{widget_id}">
    <div class="widget-container" data-widget="{widget_component}" data-alt="{alt_text}">
        <div class="widget-loading">
            <p>🔄 Loading interactive {alt_text}...</p>
            <p><small>If this doesn't load, you may need to enable JavaScript</small></p>
        </div>
    </div>
</div>

<script>
if (typeof document$ !== 'undefined') {{
    document$.subscribe(function() {{
        if (typeof initializeWidget === 'function') {{
            initializeWidget('{widget_id}', '{widget_component}', {{
                alt: '{alt_text}',
                originalPath: '{svg_path}'
            }});
        }}
    }});
}}
</script>
"""

        return match.group(0)

    return re.sub(svg_pattern, replace_svg, markdown)


def _convert_obsidian_syntax(markdown: str) -> str:
    """Convert Obsidian syntax to MkDocs Material."""

    # Obsidian callouts
    callout_pattern = r"> \[!(\w+)\]\s*(.*?)$"
    markdown = re.sub(
        callout_pattern,
        lambda m: (
            f'!!! {m.group(1).lower()} "{m.group(2).strip()}"'
            if m.group(2).strip()
            else f"!!! {m.group(1).lower()}"
        ),
        markdown,
        flags=re.MULTILINE,
    )

    # Obsidian links
    obsidian_link_pattern = r"\[\[([^\]]+)\]\]"
    markdown = re.sub(
        obsidian_link_pattern,
        lambda m: f'[{m.group(1).split("|")[-1].strip()}]({m.group(1).split("|")[0].strip().lower().replace(" ", "-")}/)',
        markdown,
    )

    return markdown
