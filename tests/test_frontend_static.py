"""Static checks of the frontend: no build step and no browser here, so the rules are checked in the source."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
SCRIPT = SRC / "script.js"
JS = SCRIPT.read_text(encoding="utf-8")
HTML = (SRC / "page.html").read_text(encoding="utf-8")
CSS = (SRC / "style.css").read_text(encoding="utf-8")


def function_body(name):
    start = JS.index(f"function {name}(")
    end = JS.find("\nfunction ", start + 1)
    return JS[start:end if end != -1 else None]


def test_no_innerhtml_at_all():
    assert ".innerHTML" not in JS and "insertAdjacentHTML" not in JS and "outerHTML" not in JS


def test_single_send_and_fetch_helpers():
    assert len(re.findall(r"\bws\.send\(", JS)) == 1, "all sends go through send()"
    assert JS.count("fetch(") == 1, "all requests go through fetchWithTimeout"
    assert "AbortController" in JS


def test_images_only_from_checked_jpeg_data_urls():
    assert re.findall(r"\.src\s*=", JS) == [".src =", ".src ="]  # thumbnails and the preview reload
    render = function_body("renderRound")
    assert "filter(isJpegDataUrl)" in render
    check = function_body("isJpegDataUrl")
    assert 'startsWith("data:image/jpeg;base64,")' in check and "[A-Za-z0-9+/]" in check
    assert "/video_feed?ts=" in JS


def test_unknown_values_render_as_na_never_zero():
    cost = function_body("formatCost")
    assert '"n/a"' in cost and '"no API fee"' in cost and "$0" not in cost
    assert '"n/a"' in function_body("formatTokens") and '"n/a"' in function_body("formatMs")


def test_speaks_the_typed_protocol_only():
    for kind in ("state", "round", "result", "round_end", "models", "system", "error"):
        assert f'"{kind}"' in JS
    for kind in ("update", "control", "set_agent", "list_models", "set_source"):
        assert f'type: "{kind}"' in JS
    assert "role:" not in JS and "toggle_stream" not in JS and '"vlm"' not in JS
    assert 'confirm === "webcam_remote"' in JS and "confirm: true" in JS


def test_stale_results_do_not_replace_the_live_answer():
    result = function_body("renderResult")
    assert result.index("if (data.stale) return") < result.index('r("answer").textContent')


def test_comparison_marks_need_both_answers_of_the_same_round():
    marks = function_body("markComparison")
    assert "a.round_id !== roundId || b.round_id !== roundId || !a.ok || !b.ok" in marks
    assert '"faster"' in marks and '"cheaper"' in marks


def test_metrics_switch_is_remembered_and_hides_metric_blocks():
    assert 'id="metricsSwitch"' in HTML and "localStorage" in JS
    assert "body.hide-metrics .metrics" in CSS
    assert HTML.count('class="metrics') + HTML.count(' metrics"') + HTML.count("metrics small") >= 3


def test_three_columns_with_the_centre_first_on_narrow_screens():
    assert 'id="slotA"' in HTML and 'id="slotB"' in HTML and 'id="cardTemplate"' in HTML
    assert re.search(r'col-lg-3 order-2 order-lg-1" id="slotA"', HTML)
    assert re.search(r'col-lg-6 order-1 order-lg-2', HTML)
    assert re.search(r'col-lg-3 order-3" id="slotB"', HTML)


def test_websocket_client_is_robust():
    assert "wss" in JS and "ws://${" not in JS
    assert "onclose" in JS and "onopen" in JS and "onerror" in JS
    assert JS.count("let reconnectTimer") == 1 and "clearTimeout(reconnectTimer)" in JS


def test_save_flow_clears_after_success():
    handler = JS[JS.index("saveBtn.onclick"):]
    assert "catch" in handler
    assert handler.index('result.status !== "ok"') < handler.index('msgInput.value = ""')


def test_sliders_keep_their_ranges():
    assert 'id="fpsSlider"' in HTML and 'min="1" max="10" step="1"' in HTML
    assert 'min="0.01"' in HTML  # temperature slider must not allow 0
    assert "resample_fps" in JS


def test_page_has_no_unprotected_external_assets():
    tags = re.findall(r"<(?:script|link)[^>]*>", HTML)
    bad = [t for t in tags if re.search(r"(?:src|href)=[\"']?(?:https?:)?//", t) and "integrity=" not in t]
    assert bad == []
    assert (SRC / "vendor" / "bootstrap.min.css").is_file()
    assert (SRC / "vendor" / "bootstrap.LICENSE").is_file()
