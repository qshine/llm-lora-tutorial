"""The book must preserve runnable code and portable links."""
from pathlib import Path
import re
import zipfile
import pytest

from book.build import build_book, split_chapters
from book.check import check_site

ROOT = Path(__file__).resolve().parents[1]


def test_chapter_split_ignores_headings_inside_code():
    text = '# Book\nIntro\n\n## 0. Start\n```python\n## Not a chapter\n### Stay\n```\n\n## 1. Next\n### Detail\n'
    intro, chapters = split_chapters(text)
    assert 'Intro' in intro
    assert len(chapters) == 2
    assert '## Not a chapter\n### Stay' in chapters[0][1]
    assert chapters[1][0] == '1. Next'


def test_book_preserves_every_code_block_and_chapter(tmp_path):
    result = build_book(ROOT, tmp_path)
    original = (ROOT / 'tasks/plan.md').read_text()
    compiled = '\n'.join((tmp_path / page).read_text() for page in result['chapters'])
    pattern = r'^```[^\n]*\n.*?^```$'
    assert re.findall(pattern, compiled, re.M | re.S) == re.findall(pattern, original, re.M | re.S)
    assert len(result['chapters']) == 10
    assert '# 4. 训练与验证：运行第一次 QLoRA' in (tmp_path / 'chapters/04-training.md').read_text()


def test_book_links_to_readable_code_and_bundles_only_source(tmp_path):
    build_book(ROOT, tmp_path)
    training = (tmp_path / 'chapters/04-training.md').read_text()
    assert '../reference/code/lab/train.py.md' in training
    assert (tmp_path / 'reference/code/lab/train.py.md').is_file()
    assert (tmp_path / 'reference/checklist.md').is_file()
    assert (tmp_path / 'reference/verification.md').is_file()
    with zipfile.ZipFile(tmp_path / 'downloads/lora-lab.zip') as archive:
        paths = archive.namelist()
        assert 'llm-lora-tutorial/lab/train.py' in paths
        assert 'llm-lora-tutorial/data/dataset.json' in paths
        assert 'llm-lora-tutorial/data/raw/seeds.jsonl' not in paths
        assert 'llm-lora-tutorial/data/facts.jsonl' not in paths
        assert 'llm-lora-tutorial/book_review.sh' in paths
        assert not any('/models/' in p or '/runs/' in p or '/.venv' in p for p in paths)
        assert archive.read('llm-lora-tutorial/tasks/plan.md').decode() == (ROOT / 'tasks/plan.md').read_text()


def test_generated_markdown_has_no_broken_local_links(tmp_path):
    build_book(ROOT, tmp_path)
    for page in tmp_path.rglob('*.md'):
        # Source viewers contain literal Markdown inside fences, not active links.
        text = re.sub(r'^(`{3,})[^\n]*\n.*?^\1\s*$', '', page.read_text(), flags=re.M | re.S)
        for target in re.findall(r'\]\(([^)\s]+)\)', text):
            if re.match(r'https?://|mailto:|#', target):
                continue
            assert not target.startswith('/'), (page, target)
            assert (page.parent / target.split('#')[0]).exists(), (page, target)


def test_notebook_download_matches_upload_source(tmp_path):
    build_book(ROOT, tmp_path)
    notebook = 'notebooks/01_gpu_train.ipynb'
    assert (tmp_path / 'downloads' / notebook).read_bytes() == (ROOT / notebook).read_bytes()
    assert '../downloads/notebooks/01_gpu_train.ipynb' in (tmp_path / 'chapters/03-gpu.md').read_text()
    with zipfile.ZipFile(tmp_path / 'downloads/lora-lab.zip') as archive:
        assert archive.read('llm-lora-tutorial/' + notebook) == (ROOT / notebook).read_bytes()
        assert 'llm-lora-tutorial/lab/notebook.py' not in archive.namelist()
        assert archive.read('llm-lora-tutorial/lab/download-lock.json') == (ROOT / 'lab/download-lock.json').read_bytes()


def test_data_viewer_explains_purpose_and_formats_one_real_record(tmp_path):
    import json
    build_book(ROOT, tmp_path)
    page = (tmp_path / 'reference/code/data/dataset.json.md').read_text()
    data = json.loads((ROOT / 'data/dataset.json').read_text())
    first = data['train'][0]
    sample = {'facts': [card for card in data['facts'] if card['id'] in first['fact_ids']],
              'train': [first], 'val': ['…'], 'test': ['…']}
    assert '唯一需要维护的数据文件' in page
    assert '[逐字段说明与使用位置]' in page
    assert json.dumps(sample, ensure_ascii=False, indent=2) in page
    assert '<summary>展开全部原始记录' in page


def test_404_uses_configured_project_prefix_and_checks_target(tmp_path):
    (tmp_path / 'index.html').write_text('<h1 id="intro">Book</h1>')
    (tmp_path / '404.html').write_text('<a href="/demo/#intro">Home</a>')
    assert check_site(tmp_path, base_path='/demo/')['pages'] == 2
    with pytest.raises(ValueError, match='non-portable'):
        check_site(tmp_path, base_path='/other/')
    (tmp_path / '404.html').write_text('<a href="/demo/#missing">Home</a>')
    with pytest.raises(ValueError, match='missing anchor'):
        check_site(tmp_path, base_path='/demo/')


def test_removed_modules_cannot_survive_as_generated_source_downloads(tmp_path):
    stale_paths = [tmp_path / 'reference/code/lab/removed.py.md',
                   tmp_path / 'downloads/source/lab/removed.py']
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('obsolete generated source')
    build_book(ROOT, tmp_path)
    assert not any(path.exists() for path in stale_paths)
    assert (tmp_path / 'reference/code/lab/train.py.md').is_file()
    assert (tmp_path / 'downloads/source/lab/train.py').read_bytes() == (ROOT / 'lab/train.py').read_bytes()
