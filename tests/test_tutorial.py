"""Exercise commands and examples copied from the teaching materials themselves."""
import argparse
import importlib
import json
from pathlib import Path
import re
import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
# 多语言教材：每个译本都要和中文版一样，经得起同样的检查。
# 日文、韩文版没有单独的 Notebook，配套英文 Notebook（代码块与英文版逐字相同）。
EDITIONS = {"zh": ("tasks/plan.md", "notebooks/01_gpu_train.ipynb"),
            "en": ("tasks/plan.en.md", "notebooks/01_gpu_train.en.ipynb"),
            "ja": ("tasks/plan.ja.md", "notebooks/01_gpu_train.en.ipynb"),
            "ko": ("tasks/plan.ko.md", "notebooks/01_gpu_train.en.ipynb")}
PLANS = {lang: (ROOT / plan).read_text() for lang, (plan, _) in EDITIONS.items()}
NOTEBOOKS = {lang: json.loads((ROOT / notebook).read_text()) for lang, (_, notebook) in EDITIONS.items()}


def blocks(text, language):
    return re.findall(r"^```" + language + r"\n(.*?)^```", text, re.M | re.S)


def tutorial_commands():
    commands = []
    suffixes = ("", ".en", ".ja", ".ko")
    for relative in [f"{name}{suffix}.md" for suffix in suffixes
                     for name in ("tasks/plan", "README", "data/README")]:
        for block in blocks((ROOT / relative).read_text(), "bash"):
            for line in block.replace("\\\n", " ").splitlines():
                words = shlex.split(line, comments=True)
                if "-m" not in words:
                    continue
                module = words[words.index("-m") + 1]
                if module.startswith("lab."):
                    commands.append((relative, module, words[words.index("-m") + 2:]))
    return commands


@pytest.mark.parametrize("document,module,arguments", tutorial_commands())
def test_documented_commands_match_real_cli_parsers(document, module, arguments, monkeypatch):
    class Captured(Exception):
        pass

    def capture(parser, *args, **kwargs):
        raise Captured(parser)

    # Stop before any training or network operation. Parse with the actual parser
    # after restoring parse_args, so missing flags and renamed subcommands fail.
    with monkeypatch.context() as context:
        context.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(Captured) as result:
            importlib.import_module(module).main()
    parser = result.value.args[0]
    parser.parse_args(arguments)


def test_every_lab_cli_has_an_executable_example():
    modules = {module for _, module, _ in tutorial_commands()}
    entries = {"lab." + path.stem for path in (ROOT / "lab").glob("*.py")
               if 'if __name__ == "__main__":' in path.read_text()}
    assert entries == modules


@pytest.mark.parametrize("lang", EDITIONS)
@pytest.mark.parametrize("relative", ["lab/train.py", "lab/training.py"])
def test_training_appendix_matches_source(relative, lang):
    match = re.search(r"### " + re.escape(relative) + r"\n+```python\n(.*?)\n```", PLANS[lang], re.S)
    assert match and match.group(1).strip() == (ROOT / relative).read_text().strip()


@pytest.mark.parametrize("lang", EDITIONS)
def test_notebook_and_book_use_identical_deployment_examples(lang):
    for cell in NOTEBOOKS[lang]["cells"]:
        if cell["id"] in {"deploy", "local-call"}:
            for language in ("bash", "python"):
                for example in blocks("".join(cell["source"]), language):
                    assert example in blocks(PLANS[lang], language), cell["id"]


@pytest.mark.parametrize("lang", [lang for lang in EDITIONS if lang != "zh"])
def test_translations_mirror_chinese_structure(lang):
    """Same chapters and the same code; only prose and code comments are translated."""
    from book.build import split_chapters

    def without_comments(block):
        return [re.sub(r"(^|\s)(#|//).*$", "", line).rstrip() for line in block.splitlines()]

    fence = r"^```[^\n]*\n.*?^```$"
    zh, other = (re.findall(fence, PLANS[edition], re.M | re.S) for edition in ("zh", lang))
    assert len(zh) == len(other)
    assert [without_comments(block) for block in zh] == [without_comments(block) for block in other]
    assert len(split_chapters(PLANS["zh"])[1]) == len(split_chapters(PLANS[lang])[1])
    if lang in ("ja", "ko"):
        assert other == re.findall(fence, PLANS["en"], re.M | re.S)


@pytest.mark.parametrize("lang", EDITIONS)
def test_actual_local_api_example_uses_deployed_model_and_generated_history(monkeypatch, lang):
    PLAN = PLANS[lang]
    example, = [block for block in blocks(PLAN, "python")
                if "build_messages" in block and "urlopen" in block]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"content": f"answer-{len(requests)}"},
                                             "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_urlopen = urllib.request.urlopen

    def redirect_to_test_server(request, timeout):
        assert request.full_url == "http://127.0.0.1:18000/v1/chat/completions"
        assert timeout == 120
        local = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                        data=request.data, headers=dict(request.header_items()))
        return original_urlopen(local, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", redirect_to_test_server)
    try:
        exec(compile(example, "tutorial-local-api", "exec"), {})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    deployed_name = re.search(r"--served-model-name ([\w-]+)", PLAN).group(1)
    assert len(requests) == 5  # Three independent questions, then two actual turns.
    assert all(request["model"] == deployed_name and request["temperature"] == 0
               and request["max_tokens"] == 256 for request in requests)
    assert all(len(request["messages"]) == 2 for request in requests[:4])
    assert requests[-1]["messages"][:2] == requests[-2]["messages"]
    assert requests[-1]["messages"][2] == {"role": "assistant", "content": "answer-4"}
    assert requests[-1]["messages"][3]["role"] == "user"


@pytest.mark.parametrize("lang", EDITIONS)
def test_curl_examples_use_deployed_model_and_training_prompt(lang):
    from lab.common import build_messages
    from lab.data import read_dataset

    PLAN = PLANS[lang]
    cell = next(cell for cell in NOTEBOOKS[lang]["cells"] if cell["id"] == "local-call")
    commands = [block for block in blocks("".join(cell["source"]), "bash") if "curl" in block]
    assert not blocks("".join(cell["source"]), "python")  # 第 13 节只用 curl
    inline = [json.loads(body) for block in commands for body in re.findall(r"-d '(\{.*?\})'", block)]
    files = [json.loads(body) for block in commands
             for body in re.findall(r"<<'JSON'\n(.*?)\nJSON\n", block, re.S)]
    deployed_name = re.search(r"--served-model-name ([\w-]+)", PLAN).group(1)
    assert len(inline) == 2 and len(files) == 2
    assert all(body["model"] == deployed_name and body["temperature"] == 0 for body in inline + files)
    assert any(body.get("stream") is True for body in inline)
    # 文件里的 system 提示词必须与训练时 build_messages 拼出的完全一致。
    _, facts = read_dataset(ROOT / "data/dataset.json")
    for body in files:
        messages = body["messages"]
        case = {"fact_ids": ["day-night"], "grade_band": "1-2",
                "history": messages[1:-1], "user": messages[-1]["content"]}
        assert messages == build_messages(facts, case)
    # 多轮请求的历史里，第 1 问必须和单轮请求的问题相同。
    assert files[1]["messages"][1] == files[0]["messages"][1]
