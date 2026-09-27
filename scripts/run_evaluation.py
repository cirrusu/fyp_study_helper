"""Run the test suite, HTTP checks and evaluation graphs.

Run from the project folder: python -m scripts.run_evaluation
Results use an isolated database and deliberate test profiles, not learner data.
"""

import argparse
import csv
import html
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CHART_RESOURCES = {
    "exercise-loop-sum": "Sum selected\nnumbers",
    "exercise-loop-grid": "Multiplication\ngrid",
    "exercise-function-total": "Function\ntotal",
    "exercise-function-default": "Function\ndefaults",
}
QUESTION_IDS = ("loops-1", "loops-2", "functions-1", "functions-2")
PACKAGES = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "scikit-learn",
    "httpx",
    "pytest",
    "matplotlib",
)
SCORE_TOLERANCE = 1e-12
PROFILE = {
    "selected_topics": ["loops", "functions"],
    "preferred_format": "exercise",
    "target_level": "beginner",
}
CONDITIONS = {
    "Unassessed": None,
    "Greater loop need": ["b", "a", "c", "c"],
    "Greater function need": ["a", "b", "a", "a"],
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def redact(value):
    if isinstance(value, dict):
        return {
            k: ("[not saved]" if k in {"token", "token_hash"} else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def resource_ids(response):
    """Get the ordered resource IDs from a recommendation response."""
    return [item["resource"]["resource_id"] for item in response["recommendations"]]


class EvaluationRunner:
    """Manage the temporary server and record the twelve HTTP checks."""

    def __init__(self, output, database):
        self.output, self.database = output, database
        self.process = None
        self.server_log = None
        self.calls, self.cases, self.rankings = [], [], {}
        self.main_token = None
        self.component_response = None
        self.case_id = "setup"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def start_server(self):
        env = os.environ.copy()
        env["STUDY_HELPER_DATA_DIR"] = str(ROOT / "data")
        env["STUDY_HELPER_DB"] = str(self.database)
        env.pop("STUDY_HELPER_CLUSTER_MODEL", None)
        self.server_log = (self.output / "server.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--no-access-log",
            ],
            cwd=ROOT,
            env=env,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Test server stopped. Read server.log in the results folder.")
            try:
                with urlopen(self.base + "/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.1)
        raise RuntimeError("Test server did not start within 40 seconds. Read server.log.")

    def stop_server(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.server_log is not None:
            self.server_log.close()

    def call(self, method, path, body=None, token=None, expected=200):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                status, result = response.status, json.load(response)
        except HTTPError as exc:
            status, result = exc.code, json.load(exc)
        self.calls.append(
            {
                "case": self.case_id,
                "method": method,
                "path": path,
                "request": redact(body),
                "status": status,
                "response": redact(result),
            }
        )
        check(
            status == expected,
            f"{method} {path}: expected {expected}, got {status}: {redact(result)}",
        )
        return result

    def register(self):
        return self.call("POST", "/v1/students", PROFILE, expected=201)["token"]

    def quiz(self, token, values, submission="diagnostic"):
        body = {
            "submission_id": submission,
            "answers": [
                {"question_id": q, "selected_option": answer}
                for q, answer in zip(QUESTION_IDS, values)
            ],
        }
        return self.call("POST", "/v1/quiz/attempts", body, token)

    def recommend(self, token, **extra):
        body = {
            "topics": ["loops", "functions"],
            "k": 4,
            "max_minutes": 15,
            "algorithm": "performance",
            **extra,
        }
        return self.call("POST", "/v1/recommendations", body, token)

    def case(self, case_id, name, function):
        self.case_id = case_id
        try:
            detail = function()
        except Exception as exc:
            self.cases.append({"case": case_id, "name": name, "status": "FAIL", "detail": str(exc)})
            raise
        self.cases.append({"case": case_id, "name": name, "status": "PASS", "detail": detail})
        print(f"{case_id} PASS - {name}", flush=True)

    def check_catalogue(self):
        health = self.call("GET", "/health")
        check(
            (health["resources"], health["questions"]) == (30, 12), "Expected original starter data"
        )
        self.main_token = self.register()
        return "30 resources; 12 questions; student creation returned HTTP 201."

    def check_authentication(self):
        self.call("GET", "/v1/me", expected=401)
        self.call("GET", "/v1/me", token="invalid-token", expected=401)
        a, b = self.register(), self.register()
        self.quiz(a, ["a"], "isolation")
        check(self.call("GET", "/v1/me", token=b)["topic_scores"] == {}, "Student records mixed")
        return "Missing/wrong tokens rejected; another student's scores stayed empty."

    def check_cold_start(self):
        result = self.recommend(self.main_token, algorithm="cluster")
        check(
            result["effective_algorithm"] == "performance" and result["cluster_id"] is None,
            "Missing fallback",
        )
        check(
            all(r["components"]["topic_need"] == 0.5 for r in result["recommendations"]),
            "Non-neutral cold start",
        )
        check(bool(result["notices"]), "Missing fallback notice")
        return "No model: performance fallback; no answers: topic need 0.5."

    def check_diagnostic_scoring(self):
        questions = self.call("GET", "/v1/quiz/questions?topic=loops")
        check(
            len(questions) == 2 and all("correct_option" not in q for q in questions),
            "Answer keys exposed",
        )
        result = self.quiz(self.main_token, ["b", "a"])
        scores = self.call("GET", "/v1/me", token=self.main_token)["topic_scores"]["loops"]
        check(
            result["correct"] == 0 and scores["performance_proxy"] == 0.25,
            "Incorrect diagnostic score",
        )
        return "Two wrong loop answers: 0/2; smoothed performance 0.25; need 0.75."

    def check_quiz_retries(self):
        token = self.register()
        first = self.quiz(token, ["b"], "retry")
        check(self.quiz(token, ["b"], "retry") == first, "Retry changed the response")
        self.call(
            "POST",
            "/v1/quiz/attempts",
            {
                "submission_id": "retry",
                "answers": [{"question_id": "loops-1", "selected_option": "a"}],
            },
            token,
            409,
        )
        self.quiz(token, ["a"], "new-attempt")
        scores = self.call("GET", "/v1/me", token=token)["topic_scores"]["loops"]
        check(
            scores["attempted"] == 1 and scores["correct"] == 1, "Repeated answers inflated counts"
        )
        return "Same request replayed; changed payload rejected with 409; latest distinct answer counted once."

    def check_atomic_validation(self):
        token = self.register()
        self.call(
            "POST",
            "/v1/quiz/attempts",
            {
                "submission_id": "invalid-mixed",
                "answers": [
                    {"question_id": "loops-1", "selected_option": "a"},
                    {"question_id": "missing", "selected_option": "a"},
                ],
            },
            token,
            422,
        )
        check(
            self.call("GET", "/v1/me", token=token)["topic_scores"] == {}, "Partial write occurred"
        )
        return "Valid + missing question rejected together; no scores were saved."

    def check_duration_boundary(self):
        at = self.recommend(self.main_token, topics=["loops"], max_minutes=8)
        below = self.recommend(self.main_token, topics=["loops"], max_minutes=7)
        check(resource_ids(at) == ["exercise-loop-sum"], "Wrong boundary result")
        check(below["recommendations"] == [], "Resource exceeded time limit")
        return "8 minutes included the 8-minute exercise; 7 minutes returned none; unknown durations excluded."

    def check_diagnostic_reranking(self):
        for name, values in CONDITIONS.items():
            token = self.register()
            if values:
                self.quiz(token, values)
            self.rankings[name] = {
                algorithm: self.recommend(token, algorithm=algorithm)
                for algorithm in ["content", "performance"]
            }
        orders = []
        for name, result in self.rankings.items():
            orders.append(
                [
                    (r["resource"]["resource_id"], r["score"])
                    for r in result["content"]["recommendations"]
                ]
            )
            check(len(result["performance"]["recommendations"]) == 4, "Expected four candidates")
        check(
            orders[0] == orders[1] == orders[2], "Content baseline changed with diagnostic answers"
        )
        for name, prefix in [
            ("Greater loop need", "exercise-loop-"),
            ("Greater function need", "exercise-function-"),
        ]:
            check(
                all(
                    r["resource"]["resource_id"].startswith(prefix)
                    for r in self.rankings[name]["performance"]["recommendations"][:2]
                ),
                "Topic priority did not reverse",
            )
        return "Top two exercises followed the weaker topic; all three content-only orders/scores stayed equal."

    def check_score_components(self):
        errors = []
        for group in self.rankings.values():
            for result in group.values():
                for row in result["recommendations"]:
                    check(
                        abs(sum(row["weights"].values()) - 1) <= SCORE_TOLERANCE,
                        "Weights do not sum to one",
                    )
                    rebuilt = sum(
                        row["components"][key] * weight for key, weight in row["weights"].items()
                    )
                    errors.append(abs(rebuilt - row["score"]))
        check(len(errors) == 24 and max(errors) <= SCORE_TOLERANCE, "Score reconstruction failed")
        return f"24 returned scores reconstructed; maximum absolute difference {max(errors):.3g}."

    def check_feedback(self):
        token = self.main_token
        self.component_response = self.recommend(token, topics=["loops"], k=5)
        self.call(
            "POST",
            "/v1/feedback",
            {"resource_id": "exercise-loop-sum", "helpful": True, "completed": True},
            token,
        )
        update = self.call(
            "POST", "/v1/feedback", {"resource_id": "exercise-loop-sum", "helpful": False}, token
        )
        check(update["completed"] is True, "Partial update lost completion")
        default = self.recommend(token, topics=["loops"])
        restored = self.recommend(
            token, topics=["loops"], include_completed=True, include_dismissed=True
        )
        check(
            "exercise-loop-sum" not in resource_ids(default)
            and "exercise-loop-sum" in resource_ids(restored),
            "Feedback filters failed",
        )
        return "Completion preserved during helpfulness-only update; excluded by default and restored with both include flags."

    def check_invalid_inputs(self):
        invalid = [
            {"k": 0},
            {"k": 21},
            {"algorithm": "missing"},
            {"topics": ["missing"]},
            {"topics": []},
            {"max_minutes": -1},
            {"student_id": "another"},
        ]
        for body in invalid:
            self.call("POST", "/v1/recommendations", body, self.main_token, 422)
        return "All seven invalid request variants returned HTTP 422."

    def check_restart_persistence(self):
        self.stop_server()
        self.start_server()
        self.call("GET", "/v1/me", token=self.main_token)
        result = self.recommend(self.main_token, topics=["loops"])
        check("exercise-loop-sum" not in resource_ids(result), "Feedback lost after restart")
        return "Actual server process restarted using same test database; token and resource exclusion survived."

    def run(self):
        """Run in order: later checks use profiles and rankings from earlier ones."""
        checks = [
            ("Catalogue and registration", self.check_catalogue),
            ("Authentication and student isolation", self.check_authentication),
            ("Cold start and fallback", self.check_cold_start),
            ("Diagnostic scoring", self.check_diagnostic_scoring),
            ("Quiz retries and latest answers", self.check_quiz_retries),
            ("Atomic validation", self.check_atomic_validation),
            ("Duration boundary", self.check_duration_boundary),
            ("Diagnostic reranking", self.check_diagnostic_reranking),
            ("Score reconstruction", self.check_score_components),
            ("Feedback and exclusion", self.check_feedback),
            ("Invalid recommendation inputs", self.check_invalid_inputs),
            ("Restart persistence", self.check_restart_persistence),
        ]
        self.start_server()
        for number, (name, function) in enumerate(checks, 1):
            self.case(f"TC{number:02d}", name, function)

    def save_evidence(self):
        """Keep completed checks and HTTP responses even when a later step fails."""
        write_json(self.output / "http_test_results.json", self.cases)
        write_json(self.output / "request_response_log.json", self.calls)
        write_json(self.output / "ranking_responses.json", self.rankings)


def build_score_rows(rankings):
    """Prepare one CSV/chart row per resource using this run's returned scores."""
    by_condition = {}
    for condition, algorithms in rankings.items():
        by_condition[condition] = {}
        for algorithm, response in algorithms.items():
            by_condition[condition][algorithm] = {
                item["resource"]["resource_id"]: item for item in response["recommendations"]
            }

    rows = []
    for resource_id in CHART_RESOURCES:
        baseline = by_condition["Unassessed"]["content"][resource_id]
        row = {
            "resource_id": resource_id,
            "title": baseline["resource"]["title"],
            "content_only": baseline["score"],
        }
        for condition in CONDITIONS:
            row[condition] = by_condition[condition]["performance"][resource_id]["score"]
        rows.append(row)
    return rows


def save_chart(figure, output, name):
    """Save both report formats with the existing output filenames."""
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=200)


def plot_diagnostic_scores(output, rows):
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(figsize=(10, 5.4))
    positions = np.arange(len(rows))
    width = 0.24
    colours = ("#a9b7c1", "#437d86", "#243f50")
    for index, (condition, colour) in enumerate(zip(CONDITIONS, colours)):
        values = [row[condition] for row in rows]
        bars = axes.bar(
            positions + (index - 1) * width, values, width, label=condition, color=colour
        )
        axes.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    axes.set_xticks(positions, list(CHART_RESOURCES.values()))
    axes.set_ylim(0, 0.8)
    axes.set_ylabel("Performance-mode ranking score")
    axes.set_title("Diagnostic answers change resource priority", loc="left", pad=16)
    axes.yaxis.grid(True, alpha=0.15)
    axes.set_axisbelow(True)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.28)
    figure.text(
        0.1, 0.02, "Controlled software cases; ranking scores are not learning gains.", fontsize=9
    )
    save_chart(figure, output, "report_figure8")
    plt.close(figure)


def plot_score_components(output, response):
    import matplotlib.pyplot as plt
    import numpy as np

    items = response["recommendations"]
    figure, axes = plt.subplots(figsize=(9, 5.2))
    positions = np.arange(len(items))
    totals = np.zeros(len(items))
    components = (
        ("content", "Content", "#243f50"),
        ("topic_need", "Diagnostic need", "#437d86"),
        ("level", "Difficulty", "#a9b7c1"),
        ("format", "Format", "#b28a42"),
    )
    for key, label, colour in components:
        values = [item["components"].get(key, 0) * item["weights"].get(key, 0) for item in items]
        axes.bar(positions, values, bottom=totals, width=0.5, label=label, color=colour)
        totals += values
    for index, total in enumerate(totals):
        axes.text(index, total + 0.018, f"{total:.4f}", ha="center", fontweight="bold")
    axes.set_xticks(positions, [item["resource"]["title"] for item in items])
    axes.set_ylim(0, 1)
    axes.set_ylabel("Weighted ranking score")
    axes.set_title("How each returned score is calculated", loc="left", pad=16)
    axes.yaxis.grid(True, alpha=0.15)
    axes.set_axisbelow(True)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    figure.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.23)
    save_chart(figure, output, "report_figure2")
    plt.close(figure)


def create_graphs(output, rankings, component_response):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    rows = build_score_rows(rankings)
    with (output / "evaluation_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["resource_id", "title", "content_only", *CONDITIONS]
        )
        writer.writeheader()
        writer.writerows(rows)
    plot_diagnostic_scores(output, rows)
    plot_score_components(output, component_response)


def html_json(value):
    return html.escape(json.dumps(value, indent=2))


def render_case_table(cases):
    rows = []
    for case in cases:
        cells = [
            f"<td>{html.escape(str(case[key]))}</td>"
            for key in ("case", "name", "status", "detail")
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def render_requests(calls):
    sections = []
    for call in calls:
        title = f"{call['case']} - {call['method']} {call['path']} - HTTP {call['status']}"
        sections.append(
            f"<details><summary>{html.escape(title)}</summary>"
            f"<pre>{html_json(call)}</pre></details>"
        )
    return "\n".join(sections)


def write_viewer(output, cases, calls, metadata, pytest_output):
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Study Helper test evidence</title>
<style>
body {{ max-width:1080px; margin:40px auto; padding:0 24px; font:17px/1.5 Arial; color:#173044; }}
h1, h2 {{ line-height:1.2; }}
table {{ border-collapse:collapse; width:100%; }}
td, th {{ border:1px solid #c8d5dd; padding:10px; text-align:left; vertical-align:top; }}
th, .note {{ background:#e8f0f4; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#f3f6f8; padding:18px; font-size:14px; }}
img {{ max-width:100%; }}
details {{ margin:12px 0; }}
.note {{ padding:16px; }}
@media print {{ details {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<h1>Study Helper: test evidence from this run</h1>
<p>{html.escape(metadata["run_time_utc"])} UTC</p>
<p class="note">Deliberately constructed software cases. These are actual API requests
and responses from this run, not observations of real students. Tokens are not saved.
This page is an evidence viewer, not a student application.</p>
<h2>Packaged automated tests</h2>
<pre>{html.escape(pytest_output)}</pre>
<h2>Additional HTTP acceptance cases</h2>
<table>
<tr><th>Case</th><th>Check</th><th>Result</th><th>Observed evidence</th></tr>
{render_case_table(cases)}
</table>
<h2>Evaluation graph (report Figure 8)</h2>
<img src="report_figure8.png" alt="Ranking scores across diagnostic conditions">
<p>Quiz answers change; topics, resource data, beginner level, exercise preference and
15-minute limit stay fixed. Content-only scores are in evaluation_scores.csv.</p>
<h2>Score components (report Figure 2)</h2>
<img src="report_figure2.png" alt="Weighted components of the two loop-resource scores">
<h2>Requests and responses</h2>
<p>Open a row to inspect the submitted JSON, response status and returned data.</p>
{render_requests(calls)}
<h2>Environment</h2>
<pre>{html_json(metadata)}</pre>
</body>
</html>
"""
    (output / "testing_evidence.html").write_text(document, encoding="utf-8")


def environment_metadata(now):
    return {
        "run_time_utc": now.isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in PACKAGES},
        "catalogue": "Original included 30-resource catalogue",
        "cluster_model": "None for HTTP cases",
        "data_type": "Deliberate software fixtures; no real participant data",
    }


def run_packaged_tests(output):
    """Capture the original suite's output and preserve its exit status."""
    print("Running the original packaged tests...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short", "tests/test_api.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = result.stdout + result.stderr
    (output / "pytest_results.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return result.returncode, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="New folder for this run; existing folders are refused"
    )
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    folder_name = "report-" + now.strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output_dir or ROOT / "results" / folder_name).resolve()
    if output.exists():
        parser.error(
            "Results folder already exists. Omit --output-dir for a fresh timestamped folder."
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        parser.error("Install requirements-report.txt first using this environment's Python.")

    output.mkdir(parents=True)
    metadata = environment_metadata(now)
    write_json(output / "environment.json", metadata)
    exit_code, pytest_output = run_packaged_tests(output)
    if exit_code:
        print(f"Packaged tests failed. Read {output / 'pytest_results.txt'}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="study-helper-tests-") as temp:
        runner = EvaluationRunner(output, Path(temp) / "test.sqlite3")
        try:
            runner.run()
            create_graphs(output, runner.rankings, runner.component_response)
            write_viewer(output, runner.cases, runner.calls, metadata, pytest_output)
        except Exception:
            (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"A check failed. Read failure.txt and server.log in {output}", file=sys.stderr)
            return 1
        finally:
            runner.stop_server()
            runner.save_evidence()
    print(f"All 12 HTTP cases passed. Results: {output}", flush=True)
    print("Open testing_evidence.html to inspect results. The two graphs are saved as PNG and PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
