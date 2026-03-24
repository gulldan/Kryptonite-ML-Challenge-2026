# ruff: noqa: E501
"""Build and render interactive embedding-atlas artifacts."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .io import (
    align_metadata_rows,
    load_embedding_matrix,
    load_metadata_rows,
    stringify_metadata_fields,
)
from .models import (
    POINTS_JSONL_NAME,
    REPORT_HTML_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    EmbeddingAtlasPoint,
    EmbeddingAtlasReport,
    EmbeddingAtlasSummary,
    NeighborPoint,
    ProjectionMethod,
    WrittenEmbeddingAtlasArtifacts,
)
from .projection import compute_cosine_neighbors, project_embeddings


def build_embedding_atlas(
    *,
    project_root: Path | str,
    output_root: Path | str,
    embeddings_path: Path | str,
    metadata_path: Path | str,
    title: str,
    projection_method: ProjectionMethod,
    point_id_field: str,
    label_field: str,
    color_by_field: str,
    search_fields: tuple[str, ...],
    audio_path_field: str | None = None,
    image_path_field: str | None = None,
    neighbors: int = 5,
    embeddings_key: str = "embeddings",
    ids_key: str | None = None,
) -> EmbeddingAtlasReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    output_root_path = resolve_project_path(str(project_root_path), str(output_root))
    embeddings_source_path = resolve_project_path(str(project_root_path), str(embeddings_path))
    metadata_source_path = resolve_project_path(str(project_root_path), str(metadata_path))

    embeddings, point_ids = load_embedding_matrix(
        embeddings_source_path,
        embeddings_key=embeddings_key,
        ids_key=ids_key,
    )
    metadata_rows = align_metadata_rows(
        metadata_rows=load_metadata_rows(metadata_source_path),
        point_id_field=point_id_field,
        point_ids=point_ids,
        expected_count=embeddings.shape[0],
    )

    projection = project_embeddings(embeddings, method=projection_method)
    neighbor_indices, neighbor_distances = compute_cosine_neighbors(embeddings, top_k=neighbors)

    points: list[EmbeddingAtlasPoint] = []
    point_id_lookup: list[str] = []
    label_lookup: list[str] = []
    for index, row in enumerate(metadata_rows):
        metadata = stringify_metadata_fields(row)
        point_id = _select_point_id(
            row=row,
            fallback_id=point_ids[index] if point_ids is not None else None,
            point_id_field=point_id_field,
            point_index=index,
        )
        label = metadata.get(label_field) or point_id
        color_key = metadata.get(color_by_field) or label
        point_id_lookup.append(point_id)
        label_lookup.append(label)
        points.append(
            EmbeddingAtlasPoint(
                point_index=index,
                point_id=point_id,
                label=label,
                color_key=color_key,
                x=round(float(projection.coordinates[index, 0]), 6),
                y=round(float(projection.coordinates[index, 1]), 6),
                metadata=metadata,
                search_text=_build_search_text(metadata=metadata, search_fields=search_fields),
                audio_href=_resolve_media_href(
                    raw_path=metadata.get(audio_path_field) if audio_path_field else None,
                    project_root=project_root_path,
                    output_root=output_root_path,
                ),
                image_href=_resolve_media_href(
                    raw_path=metadata.get(image_path_field) if image_path_field else None,
                    project_root=project_root_path,
                    output_root=output_root_path,
                ),
            )
        )

    points_with_neighbors: list[EmbeddingAtlasPoint] = []
    for point in points:
        indexed_neighbors: list[NeighborPoint] = []
        for neighbor_index, neighbor_distance in zip(
            neighbor_indices[point.point_index],
            neighbor_distances[point.point_index],
            strict=True,
        ):
            indexed_neighbors.append(
                NeighborPoint(
                    point_id=point_id_lookup[int(neighbor_index)],
                    label=label_lookup[int(neighbor_index)],
                    distance=round(float(neighbor_distance), 6),
                )
            )
        points_with_neighbors.append(
            EmbeddingAtlasPoint(
                point_index=point.point_index,
                point_id=point.point_id,
                label=point.label,
                color_key=point.color_key,
                x=point.x,
                y=point.y,
                metadata=point.metadata,
                search_text=point.search_text,
                audio_href=point.audio_href,
                image_href=point.image_href,
                neighbors=tuple(indexed_neighbors),
            )
        )

    summary = EmbeddingAtlasSummary(
        point_count=embeddings.shape[0],
        embedding_dim=embeddings.shape[1],
        projection_method=projection_method,
        point_id_field=point_id_field,
        label_field=label_field,
        color_by_field=color_by_field,
        search_fields=search_fields,
        neighbor_count=neighbor_indices.shape[1],
        explained_variance_ratio_2d=projection.explained_variance_ratio_2d,
        distinct_label_count=len({point.label for point in points_with_neighbors}),
        distinct_color_count=len({point.color_key for point in points_with_neighbors}),
    )
    return EmbeddingAtlasReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        output_root=str(output_root_path),
        title=title,
        embeddings_path=_relative_to_project(embeddings_source_path, project_root_path),
        metadata_path=_relative_to_project(metadata_source_path, project_root_path),
        summary=summary,
        points=tuple(points_with_neighbors),
    )


def write_embedding_atlas_report(
    report: EmbeddingAtlasReport,
    *,
    output_root: Path | str | None = None,
) -> WrittenEmbeddingAtlasArtifacts:
    output_root_path = Path(output_root or report.output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    points_path = output_root_path / POINTS_JSONL_NAME
    json_path = output_root_path / REPORT_JSON_NAME
    markdown_path = output_root_path / REPORT_MARKDOWN_NAME
    html_path = output_root_path / REPORT_HTML_NAME

    points_path.write_text(
        "".join(json.dumps(point.to_dict(), sort_keys=True) + "\n" for point in report.points)
    )
    json_path.write_text(
        json.dumps(report.to_dict(include_points=True), indent=2, sort_keys=True) + "\n"
    )
    markdown_path.write_text(render_embedding_atlas_markdown(report) + "\n")
    html_path.write_text(render_embedding_atlas_html(report) + "\n")

    return WrittenEmbeddingAtlasArtifacts(
        output_root=str(output_root_path),
        points_path=str(points_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        html_path=str(html_path),
    )


def render_embedding_atlas_markdown(report: EmbeddingAtlasReport) -> str:
    top_labels = _format_top_counts(point.label for point in report.points)
    top_colors = _format_top_counts(point.color_key for point in report.points)
    return "\n".join(
        [
            f"# {report.title}",
            "",
            f"- Generated at: `{report.generated_at}`",
            f"- Embeddings: `{report.embeddings_path}`",
            f"- Metadata: `{report.metadata_path}`",
            f"- Output root: `{report.output_root}`",
            "",
            "## Summary",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Points", str(report.summary.point_count)],
                    ["Embedding dim", str(report.summary.embedding_dim)],
                    ["Projection method", report.summary.projection_method],
                    [
                        "Explained variance ratio (2D)",
                        f"{report.summary.explained_variance_ratio_2d:.4f}",
                    ],
                    ["Point id field", report.summary.point_id_field],
                    ["Label field", report.summary.label_field],
                    ["Color field", report.summary.color_by_field],
                    ["Neighbor count", str(report.summary.neighbor_count)],
                    ["Distinct labels", str(report.summary.distinct_label_count)],
                    ["Distinct color groups", str(report.summary.distinct_color_count)],
                ],
            ),
            "",
            "## Search Fields",
            "",
            ", ".join(f"`{field}`" for field in report.summary.search_fields),
            "",
            "## Largest Label Groups",
            "",
            top_labels,
            "",
            "## Largest Color Groups",
            "",
            top_colors,
            "",
            "## Usage",
            "",
            f"Open `{REPORT_HTML_NAME}` next to this report to explore the interactive atlas.",
        ]
    )


def render_embedding_atlas_html(report: EmbeddingAtlasReport) -> str:
    payload = {
        "title": report.title,
        "summary": report.summary.to_dict(),
        "points": [point.to_dict() for point in report.points],
    }
    data_json = json.dumps(payload, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape_html(report.title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --border: #d7d0c2;
      --ink: #1f2622;
      --muted: #67736d;
      --accent: #165b47;
      --shadow: 0 24px 60px rgba(24, 36, 31, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background:
        radial-gradient(circle at top left, rgba(22, 91, 71, 0.10), transparent 28rem),
        linear-gradient(180deg, #faf7f0 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(320px, 0.9fr);
      min-height: 100vh;
    }}
    .main {{
      padding: 2rem;
    }}
    .side {{
      border-left: 1px solid var(--border);
      background: rgba(255, 253, 248, 0.92);
      backdrop-filter: blur(20px);
      padding: 1.5rem;
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0 0 0.6rem;
      font-size: clamp(2rem, 3vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .lede {{
      color: var(--muted);
      max-width: 60rem;
      margin-bottom: 1.2rem;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1rem;
      align-items: center;
    }}
    .toolbar input {{
      min-width: 18rem;
      flex: 1 1 20rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.8rem 1rem;
      background: rgba(255, 255, 255, 0.75);
      color: var(--ink);
      font: inherit;
    }}
    .stat {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.5rem 0.85rem;
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .plot-card {{
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,253,248,0.95));
      border-radius: 28px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    #atlas {{
      display: block;
      width: 100%;
      height: auto;
      cursor: crosshair;
      background:
        radial-gradient(circle at center, rgba(22, 91, 71, 0.05), transparent 48%),
        linear-gradient(180deg, rgba(255,255,255,0.92), rgba(247,244,236,0.98));
    }}
    .hint {{
      padding: 0.8rem 1rem 1rem;
      color: var(--muted);
      font-size: 0.94rem;
      border-top: 1px solid rgba(215, 208, 194, 0.7);
    }}
    .section {{
      margin-bottom: 1.25rem;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 0.3rem;
    }}
    .detail-title {{
      margin: 0 0 0.35rem;
      font-size: 1.55rem;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .detail-subtitle {{
      color: var(--muted);
      margin-bottom: 0.9rem;
      font-size: 0.96rem;
    }}
    .meta-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    .meta-table td {{
      border-top: 1px solid rgba(215, 208, 194, 0.75);
      padding: 0.45rem 0;
      vertical-align: top;
    }}
    .meta-table td:first-child {{
      color: var(--muted);
      width: 38%;
      padding-right: 1rem;
    }}
    .media {{
      display: grid;
      gap: 0.8rem;
    }}
    .media img {{
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: white;
    }}
    .neighbors {{
      display: grid;
      gap: 0.55rem;
    }}
    .neighbor {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255,255,255,0.74);
      padding: 0.7rem 0.8rem;
      cursor: pointer;
      text-align: left;
      font: inherit;
      color: inherit;
    }}
    .neighbor:hover {{
      border-color: var(--accent);
    }}
    .neighbor small {{
      display: block;
      color: var(--muted);
      margin-top: 0.25rem;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .side {{
        border-left: 0;
        border-top: 1px solid var(--border);
      }}
      .main {{
        padding: 1rem;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <main class="main">
      <h1>{_escape_html(report.title)}</h1>
      <div class="lede">
        Interactive embedding atlas with 2D projection, cosine-neighbor lookup, and optional audio/image preview.
      </div>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Filter by speaker, split, dataset, device…">
        <div class="stat" id="visible-count"></div>
        <div class="stat">method: {report.summary.projection_method}</div>
        <div class="stat">dim: {report.summary.embedding_dim}</div>
        <div class="stat">var2d: {report.summary.explained_variance_ratio_2d:.4f}</div>
      </div>
      <div class="plot-card">
        <canvas id="atlas" width="1280" height="820"></canvas>
        <div class="hint" id="hover-hint">Click a point to inspect metadata, media preview, and nearest neighbors.</div>
      </div>
    </main>
    <aside class="side">
      <div class="section">
        <div class="eyebrow">Selection</div>
        <div id="selection">
          <div class="empty">No point selected yet.</div>
        </div>
      </div>
    </aside>
  </div>
  <script id="atlas-data" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("atlas-data").textContent);
    const points = payload.points;
    const pointById = new Map(points.map((point) => [point.point_id, point]));
    const canvas = document.getElementById("atlas");
    const context = canvas.getContext("2d");
    const searchInput = document.getElementById("search");
    const selectionNode = document.getElementById("selection");
    const hoverHintNode = document.getElementById("hover-hint");
    const visibleCountNode = document.getElementById("visible-count");
    const padding = 42;
    const bounds = points.reduce((acc, point) => {{
      acc.minX = Math.min(acc.minX, point.x);
      acc.maxX = Math.max(acc.maxX, point.x);
      acc.minY = Math.min(acc.minY, point.y);
      acc.maxY = Math.max(acc.maxY, point.y);
      return acc;
    }}, {{ minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }});
    const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);

    function colorFromKey(key) {{
      let hash = 0;
      for (const char of key) {{
        hash = ((hash << 5) - hash) + char.charCodeAt(0);
        hash |= 0;
      }}
      const hue = Math.abs(hash) % 360;
      return `hsl(${{hue}} 62% 42%)`;
    }}

    function projectPoint(point) {{
      const x = padding + ((point.x - bounds.minX) / spanX) * (canvas.width - padding * 2);
      const y = padding + ((bounds.maxY - point.y) / spanY) * (canvas.height - padding * 2);
      return {{ ...point, screenX: x, screenY: y }};
    }}

    const projected = points.map(projectPoint);
    let selectedPointId = null;
    let hoverPointId = null;
    let searchTerm = "";

    function isVisible(point) {{
      return !searchTerm || point.search_text.toLowerCase().includes(searchTerm);
    }}

    function render() {{
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "rgba(20, 39, 33, 0.04)";
      context.fillRect(0, 0, canvas.width, canvas.height);

      let visibleCount = 0;
      for (const point of projected) {{
        const visible = isVisible(point);
        if (visible) visibleCount += 1;
        const alpha = visible ? (point.point_id === selectedPointId ? 1.0 : 0.86) : 0.08;
        const radius = point.point_id === selectedPointId ? 8 : (point.point_id === hoverPointId ? 6 : 4.5);
        context.beginPath();
        context.arc(point.screenX, point.screenY, radius, 0, Math.PI * 2);
        context.fillStyle = colorFromKey(point.color_key).replace("hsl(", "hsla(").replace(")", ` / ${{alpha}})`);
        context.fill();
      }}
      visibleCountNode.textContent = `${{visibleCount}} / ${{points.length}} visible`;
    }}

    function renderSelection() {{
      if (!selectedPointId) {{
        selectionNode.innerHTML = '<div class="empty">No point selected yet.</div>';
        return;
      }}
      const point = pointById.get(selectedPointId);
      if (!point) {{
        selectionNode.innerHTML = '<div class="empty">Selected point is missing.</div>';
        return;
      }}
      const metadataRows = Object.entries(point.metadata)
        .map(([key, value]) => `<tr><td>${{escapeHtml(key)}}</td><td>${{escapeHtml(String(value))}}</td></tr>`)
        .join("");
      const neighbors = point.neighbors.length
        ? point.neighbors.map((neighbor) => `
            <button class="neighbor" data-neighbor-id="${{escapeHtml(neighbor.point_id)}}">
              ${{escapeHtml(neighbor.label)}}
              <small>${{escapeHtml(neighbor.point_id)}} · cosine distance ${{Number(neighbor.distance).toFixed(4)}}</small>
            </button>
          `).join("")
        : '<div class="empty">No neighbors were computed.</div>';
      const imageBlock = point.image_href
        ? `<div><div class="eyebrow">Image</div><img src="${{escapeHtml(point.image_href)}}" alt=""></div>`
        : "";
      const audioBlock = point.audio_href
        ? `<div><div class="eyebrow">Audio</div><audio controls preload="none" src="${{escapeHtml(point.audio_href)}}" style="width:100%"></audio></div>`
        : "";
      selectionNode.innerHTML = `
        <div class="section">
          <div class="eyebrow">Point</div>
          <div class="detail-title">${{escapeHtml(point.label)}}</div>
          <div class="detail-subtitle">${{escapeHtml(point.point_id)}} · color=${{escapeHtml(point.color_key)}}</div>
        </div>
        <div class="section media">
          ${{imageBlock}}
          ${{audioBlock}}
        </div>
        <div class="section">
          <div class="eyebrow">Metadata</div>
          <table class="meta-table"><tbody>${{metadataRows}}</tbody></table>
        </div>
        <div class="section">
          <div class="eyebrow">Nearest Neighbors</div>
          <div class="neighbors">${{neighbors}}</div>
        </div>
      `;
      for (const button of selectionNode.querySelectorAll("[data-neighbor-id]")) {{
        button.addEventListener("click", () => {{
          selectedPointId = button.getAttribute("data-neighbor-id");
          render();
          renderSelection();
        }});
      }}
    }}

    function escapeHtml(value) {{
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function nearestPoint(clientX, clientY) {{
      const rect = canvas.getBoundingClientRect();
      const x = (clientX - rect.left) * (canvas.width / rect.width);
      const y = (clientY - rect.top) * (canvas.height / rect.height);
      let best = null;
      let bestDistance = Infinity;
      for (const point of projected) {{
        if (!isVisible(point)) continue;
        const dx = point.screenX - x;
        const dy = point.screenY - y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < bestDistance) {{
          bestDistance = distance;
          best = point;
        }}
      }}
      return bestDistance <= 12 ? best : null;
    }}

    canvas.addEventListener("mousemove", (event) => {{
      const point = nearestPoint(event.clientX, event.clientY);
      hoverPointId = point ? point.point_id : null;
      hoverHintNode.textContent = point
        ? `${{point.label}} · ${{point.point_id}}`
        : "Click a point to inspect metadata, media preview, and nearest neighbors.";
      render();
    }});

    canvas.addEventListener("mouseleave", () => {{
      hoverPointId = null;
      hoverHintNode.textContent = "Click a point to inspect metadata, media preview, and nearest neighbors.";
      render();
    }});

    canvas.addEventListener("click", (event) => {{
      const point = nearestPoint(event.clientX, event.clientY);
      if (!point) return;
      selectedPointId = point.point_id;
      render();
      renderSelection();
    }});

    searchInput.addEventListener("input", () => {{
      searchTerm = searchInput.value.trim().toLowerCase();
      render();
    }});

    if (points.length) {{
      selectedPointId = points[0].point_id;
    }}
    render();
    renderSelection();
  </script>
</body>
</html>"""


def _select_point_id(
    *,
    row: dict[str, object],
    fallback_id: str | None,
    point_id_field: str,
    point_index: int,
) -> str:
    candidate = row.get(point_id_field)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if fallback_id is not None:
        return fallback_id
    return f"point-{point_index:06d}"


def _build_search_text(*, metadata: dict[str, str], search_fields: tuple[str, ...]) -> str:
    values = [metadata.get(field, "") for field in search_fields]
    return " | ".join(value for value in values if value).lower()


def _resolve_media_href(
    *,
    raw_path: str | None,
    project_root: Path,
    output_root: Path,
) -> str | None:
    if raw_path is None:
        return None
    if raw_path.startswith(("http://", "https://", "data:")):
        return raw_path
    resolved = resolve_project_path(str(project_root), raw_path)
    if not resolved.exists():
        return None
    return Path(os.path.relpath(resolved, output_root)).as_posix()


def _relative_to_project(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _format_top_counts(values: Iterable[str]) -> str:
    counts = Counter(values)
    if not counts:
        return "_none_"
    lines = [f"- `{label}`: {count}" for label, count in counts.most_common(10)]
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


__all__ = [
    "build_embedding_atlas",
    "render_embedding_atlas_html",
    "render_embedding_atlas_markdown",
    "write_embedding_atlas_report",
]
