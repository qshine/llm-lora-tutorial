"""MkDocs hook: split the canonical tutorial without altering fenced code."""
from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

SLUGS = (
    '00-start', '01-concepts', '02-data', '03-gpu', '04-training',
    '05-deployment', '06-evaluation', '07-troubleshooting', '08-developer',
    'appendix-training-code',
)


def lines_with_context(text: str):
    """Yield lines and whether they belong to a fenced code block."""
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip('\n'))
        inside = fence is not None
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = run
                inside = True
            elif run[0] == fence[0] and len(run) >= len(fence) and not tail.strip():
                fence = None
        yield line, inside
    if fence is not None:
        raise ValueError('Unclosed Markdown code fence')


def split_chapters(text: str):
    intro, chapters = [], []
    for line, inside in lines_with_context(text):
        if not inside and line.startswith('## '):
            chapters.append((line[3:].strip(), []))
        elif chapters:
            chapters[-1][1].append(line)
        else:
            intro.append(line)
    return ''.join(intro), [(title, ''.join(lines)) for title, lines in chapters]


def rewrite(text: str, source: str, destination: str, routes: dict, *, promote=False):
    """Resolve source-relative links into book-relative links, outside code."""
    def replace(match):
        label, url = match.groups()
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source), parts.path))
        if target not in routes:
            raise ValueError(f'Unmapped local link in {source}: {url}')
        relative = posixpath.relpath(routes[target], posixpath.dirname(destination) or '.')
        if parts.fragment:
            relative += '#' + parts.fragment
        return f'[{label}]({relative})'

    output = []
    for line, inside in lines_with_context(text):
        if not inside:
            if promote and line.startswith('###'):
                line = line[1:]
            line = re.sub(r'\[([^\]\n]+)\]\(([^)\s]+)\)', replace, line)
        output.append(line)
    return ''.join(output)


def source_files(root: Path):
    """Explicit publication allowlist: never walk model/cache/secret folders."""
    paths = [root / 'README.md', root / 'pyproject.toml', root / '.gitignore', root / 'book_review.sh',
             root / 'GPU_UPLOAD_README.md']
    paths += list(root.glob('requirements-*.txt'))
    for folder in ('lab', 'tests', 'tools'):
        paths += list((root / folder).glob('*.py'))
    paths += list((root / 'tasks').glob('*.md'))
    paths += [root / 'data/README.md', root / 'data/dataset.json']
    paths += [p for p in (root / 'images').iterdir() if p.suffix in ('.png', '.jpg')]
    paths += [root / 'notebooks/01_gpu_train.ipynb']
    paths += [root / 'lab/download-lock.json']
    return sorted(path for path in paths if path.is_file())


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_book(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    original = (root / 'tasks/plan.md').read_text(encoding='utf-8')
    intro, chapters = split_chapters(original)
    if len(chapters) != len(SLUGS):
        raise ValueError('Tutorial chapter count changed: update book navigation and SLUGS together')
    # These two directories contain only generated copies. Rebuild them so removed
    # modules cannot survive as obsolete source pages or downloadable files.
    for generated in ('reference/code', 'downloads/source'):
        if (output / generated).exists():
            shutil.rmtree(output / generated)
    pages = [f'chapters/{slug}.md' for slug in SLUGS]
    routes = {
        'tasks/plan.md': 'index.md',
        'tasks/todo.md': 'reference/checklist.md',
        'tasks/verification.md': 'reference/verification.md',
        'data/README.md': 'reference/data.md',
        'README.md': 'index.md',
        'notebooks/01_gpu_train.ipynb': 'downloads/notebooks/01_gpu_train.ipynb',
    }
    files = source_files(root)
    # 教程截图统一放在 images/，发布到站点的 assets/。
    images = [p for p in (root / 'images').iterdir() if p.suffix in ('.png', '.jpg')]
    for image in images:
        routes[f'images/{image.name}'] = f'assets/{image.name}'
    viewable = [p for p in files if p.suffix in ('.py', '.json', '.jsonl', '.txt', '.sh') or p.name == 'pyproject.toml']
    for path in viewable:
        rel = path.relative_to(root).as_posix()
        routes[rel] = f'reference/code/{rel}.md'
    for page, (title, body) in zip(pages, chapters):
        body = rewrite(body, 'tasks/plan.md', page, routes, promote=True)
        write(output / page, f'# {title}\n\n{body.lstrip()}')
    intro = '\n'.join(intro.splitlines()[1:]).strip()
    intro = rewrite(intro, 'tasks/plan.md', 'index.md', routes)
    home = (root / 'book/home.md').read_text(encoding='utf-8')
    write(output / 'index.md', home.replace('{{ introduction }}', intro))
    for source in ('tasks/todo.md', 'tasks/verification.md', 'data/README.md'):
        write(output / routes[source], rewrite((root / source).read_text(), source, routes[source], routes))
    for path in viewable:
        rel = path.relative_to(root).as_posix()
        target = routes[rel]
        raw = 'downloads/source/' + rel
        (output / raw).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output / raw)
        link = posixpath.relpath(raw, posixpath.dirname(target))
        text = path.read_text(encoding='utf-8')
        fence = '`' * max(3, max((len(m[0]) + 1 for m in re.finditer(r'`{3,}', text)), default=3))
        lang = {'.py': 'python', '.json': 'json', '.jsonl': 'json', '.toml': 'toml', '.sh': 'bash'}.get(path.suffix, 'text')
        body = f'{fence}{lang}\n{text.rstrip()}\n{fence}\n'
        if rel == 'data/dataset.json':
            dataset = json.loads(text)
            guide = posixpath.relpath('reference/data.md', posixpath.dirname(target))
            first = dataset['train'][0]
            sample = {'facts': [card for card in dataset['facts'] if card['id'] in first['fact_ids']],
                      'train': [first], 'val': ['…'], 'test': ['…']}
            example = json.dumps(sample, ensure_ascii=False, indent=2)
            sizes = '、'.join(f'{split} {len(dataset[split])} 段' for split in ('train', 'val', 'test'))
            body = (f'唯一需要维护的数据文件：{len(dataset["facts"])} 张事实卡，'
                    f'对话按集合平级存放（{sizes}）。facts 提供每题可依据的资料，train/val/test 提供示范对话；[逐字段说明与使用位置]({guide})。\n\n'
                    f'## 第一段对话和它引用的事实\n\n下方只摘取一个完整样例，保留顶层结构；全部数据在折叠区与原文件中。\n\n'
                    f'```json\n{example}\n```\n\n'
                    f'<details markdown="1">\n<summary>展开全部原始记录</summary>\n\n'
                    f'{body}\n</details>\n')
        write(output / target, f'---\nsearch:\n  exclude: true\n---\n\n# {rel}\n\n'
              f'[查看原始文件]({link}) · [全部源码](../../../sources.md)\n\n'
              f'{body}')
        # Depth varies between lab/*.py and data/raw/*.jsonl.
        content = (output / target).read_text()
        index_link = posixpath.relpath('reference/sources.md', posixpath.dirname(target))
        write(output / target, content.replace('../../../sources.md', index_link))
    source_index = '# 源码与数据\n\n直接在浏览器阅读脚本，或下载完整练习包到本地执行。\n\n'
    source_index += '[下载练习包 ZIP](../downloads/lora-lab.zip){ .md-button .md-button--primary }\n\n'
    source_index += '练习包包含教程、Python 代码、测试、依赖清单和候选数据，不含模型权重、训练结果或本地环境。数据仍需审核。\n\n'
    source_index += ('[下载 GPU 训练 Notebook](../downloads/notebooks/01_gpu_train.ipynb) · '
                     '[使用方法](../chapters/03-gpu.md)\n\n'
                     'Notebook 直接读取包内 data/dataset.json，需要与项目代码一起使用。第 3 章的本地 GPU 上传包还包含分词器；上方通用源码练习包不包含模型缓存。\n\n')
    source_index += '| 文件 | 阅读 |\n| --- | --- |\n'
    for path in viewable:
        rel = path.relative_to(root).as_posix()
        source_index += f'| `{rel}` | [打开]({posixpath.relpath(routes[rel], "reference")}) |\n'
    write(output / 'reference/sources.md', source_index)
    (output / 'downloads').mkdir(exist_ok=True)
    for path in files:
        if path.suffix == '.ipynb':
            target = output / routes[path.relative_to(root).as_posix()]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    with ZipFile(output / 'downloads/lora-lab.zip', 'w', ZIP_DEFLATED) as archive:
        # Include book files too, so the root README's preview instructions work.
        extras = list((root / 'book').rglob('*')) + [root / 'mkdocs.yml', root / '.book/.gitkeep']
        extras += list((root / '.github/workflows').glob('*.yml'))
        for path in sorted(set(files + extras)):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path, 'llm-lora-tutorial/' + path.relative_to(root).as_posix())
    shutil.copytree(root / 'book/assets', output / 'assets', dirs_exist_ok=True)
    for image in images:
        shutil.copyfile(image, output / 'assets' / image.name)
    write(output / '.nojekyll', '')
    result = {'chapters': pages, 'source': 'tasks/plan.md', 'chapter_count': len(pages)}
    write(output / 'book-manifest.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def on_config(config):
    root = Path(__file__).resolve().parents[1]
    site_url = os.environ.get('BOOK_SITE_URL')
    if site_url:
        config['site_url'] = site_url.rstrip('/') + '/'
    result = build_book(root, Path(config['docs_dir']))
    result['base_path'] = urlsplit(config['site_url'] or '/').path
    write(Path(config['docs_dir']) / 'book-manifest.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return config


def on_serve(server, config, builder):
    # Only watch authoritative sources. Watching generated Markdown causes a
    # rebuild to trigger another rebuild indefinitely.
    server.unwatch(config['docs_dir'])
    return server
