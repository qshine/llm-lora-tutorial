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
PLAN = (ROOT / "tasks/plan.md").read_text()
NOTEBOOK = json.loads((ROOT / "notebooks/01_gpu_train.ipynb").read_text())


def blocks(text, language):
    return re.findall(r"^```" + language + r"\n(.*?)^```", text, re.M | re.S)


def tutorial_commands():
    commands = []
    for relative in ("tasks/plan.md", "README.md", "data/README.md"):
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


@pytest.mark.parametrize("relative", ["lab/train.py", "lab/training.py"])
def test_training_appendix_matches_source(relative):
    match = re.search(r"### " + re.escape(relative) + r"\n+```python\n(.*?)\n```", PLAN, re.S)
    assert match and match.group(1).strip() == (ROOT / relative).read_text().strip()


def test_notebook_and_book_use_identical_deployment_examples():
    for cell in NOTEBOOK["cells"]:
        if cell["id"] in {"deploy", "local-call"}:
            for language in ("bash", "python"):
                for example in blocks("".join(cell["source"]), language):
                    assert example in blocks(PLAN, language), cell["id"]


def test_actual_local_api_example_uses_deployed_model_and_generated_history(monkeypatch):
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


def test_curl_examples_use_deployed_model_and_training_prompt():
    from lab.common import build_messages
    from lab.data import read_dataset

    cell = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == "local-call")
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
