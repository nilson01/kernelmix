"""Assemble site navigation from the maintained Markdown chapters."""
from __future__ import annotations

import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
API_GROUPS = {
    "fitting": ("Fit outcomes and set priors", ["Fitting"]),
    "results": ("Inspect and save results", ["Fit results and diagnostics"]),
    "prediction": ("Predict outcomes", ["Prediction"]),
    "cluster-survival": ("Clusters and survival", ["Structured model results", "Cluster and survival results"]),
    "contrasts": ("Exposure contrasts and curves", ["Exposure contrasts and curves"]),
    "plots-kernels": ("Plot and calculate kernels", ["Plotting", "Kernel helper", "Method sources"]),
}


def slug(text):
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def sections(text):
    """Split level-two headings, ignoring headings inside fenced code."""
    result = {}
    current = None
    fenced = False
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            current = line[3:].strip()
            if current in result:
                raise ValueError(f"Duplicate API section: {current}")
            result[current] = []
        elif current is not None:
            result[current].append(line)
    return {key: "".join(value).strip() for key, value in result.items()}


def on_files(files, config):
    # The API's grouped views share the maintained api.md, rather than duplicating it.
    api = (DOCS / "api.md").read_text(encoding="utf-8")
    parts = sections(api)
    expected = {heading for _, headings in API_GROUPS.values() for heading in headings}
    if set(parts) != expected:
        raise ValueError(f"API headings changed; review navigation: {set(parts) ^ expected}")
    mapping = {}
    for group, (_, headings) in API_GROUPS.items():
        for heading in headings:
            mapping[slug(heading)] = f"reference/{group}.md#{slug(heading)}"
            for subheading in re.findall(r"^### (.+)$", parts[heading], re.M):
                mapping[slug(subheading)] = f"reference/{group}.md#{slug(subheading)}"
    config.extra["api_anchor_map"] = mapping
    original = files.get_file_from_path("api.md")
    if original:
        files.remove(original)
    for group, (title, headings) in API_GROUPS.items():
        body = "\n\n".join(f"## {heading}\n\n{parts[heading]}" for heading in headings)
        files.append(File.generated(config, f"reference/{group}.md", content=f"# {title}\n\nExamples use `import kernelmix as km`. See the [tutorial](../tutorial.md) for a complete analysis.\n\n{body}\n"))
    files.append(File.generated(config, "reference/index.md", content=(
        api.split("\n## ", 1)[0].replace("# API reference", "# API by task", 1).replace("](tutorial.md)", "](../tutorial.md)").replace("](methods.md)", "](../methods.md)") + "\n\nUse the [tutorial](../tutorial.md) for a complete analysis and the "
        "[model guide](../methods.md) for equations, assumptions and prior scales.\n\n"
        "| Task | Reference |\n| --- | --- |\n" + "\n".join(
            f"| {title} | [{title}]({group}.md) |" for group, (title, _) in API_GROUPS.items()
        ) + "\n\nGaussian predictions describe means, probit predictions describe probabilities, "
        "and count predictions describe expected counts. Survival `predict` returns relative hazards; "
        "`predict_survival` returns survival probabilities or hazards. "
        "[Prediction scales](prediction.md#prediction) distinguish these outputs.\n\n"
        "| Outcome | Fitting family | Main prediction |\n| --- | --- | --- |\n"
        "| Continuous | `gaussian` | [Outcome mean or new observation](prediction.md#prediction) |\n"
        "| Binary | `binomial` | [Probit probability or zero/one observation](prediction.md#prediction) |\n"
        "| Counts | `poisson`, `negative_binomial` | [Expected or sampled counts, with offsets](cluster-survival.md#cluster-and-survival-results) |\n"
        "| Event times | `survival` | [Relative hazard, survival or cause probabilities](cluster-survival.md) |\n\n"
        "[Fitting inputs and priors](fitting.md) lists required arguments. "
        "[Structured models](../structured.md) adds kernels, clusters and detailed survival records.\n"
    )))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    homepage = re.sub(r"\n## Install\n.*?(?=\n## )", "\n", readme, flags=re.S)
    homepage = homepage.replace("docs/", "")
    files.append(File.generated(config, "index.md", content=homepage))
    for name in ("LICENSE", "NOTICE", "CITATION.cff"):
        files.append(File.generated(config, f"downloads/{name}.txt", abs_src_path=str(ROOT / name)))
    return files


def on_page_markdown(markdown, page, config, files):
    """Keep chapter links valid in generated reference subdirectories."""
    nested = page.file.src_uri.startswith("reference/") and page.file.src_uri != "reference/index.md"
    mapping = config.extra["api_anchor_map"]

    def rewrite(match):
        target = match.group(1)
        if "://" in target or target.startswith("mailto:"):
            return match.group(0)
        if target.startswith("#") and nested:
            mapped = mapping.get(target[1:])
            if mapped:
                return "](" + mapped.removeprefix("reference/") + ")"
            return match.group(0)
        if target.startswith("../tools/"):
            return "](https://github.com/nilson01/kernelmix/blob/main/" + target[3:] + ")"
        if target.startswith("api.md"):
            anchor = target.partition("#")[2]
            mapped = mapping.get(anchor, "reference/index.md")
            return "](" + (mapped.removeprefix("reference/") if nested else mapped) + ")"
        if nested and not target.startswith(("../", "#")):
            return "](../" + target + ")"
        if target == "../README.md":
            return "](index.md)"
        if target in {"LICENSE", "NOTICE"}:
            return f"](downloads/{target}.txt)"
        return match.group(0)

    return re.sub(r"\]\(([^)]+)\)", rewrite, markdown)
