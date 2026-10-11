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

# 两种语言共用同一套页面路径：中文站在根目录，英文站在 en/。
# 英文正文是同名的 *.en.md 文件；代码、数据和图片两边共用。
LANGUAGES = {
    'zh': {
        'suffix': '',
        'docs_dir': '.book',
        'config': 'mkdocs.yml',
        'dataset': ('唯一需要维护的数据文件：{facts} 张事实卡，对话按集合平级存放（{sizes}）。'
                    'facts 提供每题可依据的资料，train/val/test 提供示范对话；[逐字段说明与使用位置]({guide})。\n\n'
                    '## 第一段对话和它引用的事实\n\n下方只摘取一个完整样例，保留顶层结构；全部数据在折叠区与原文件中。\n\n'),
        'size': '{split} {count} 段',
        'size_sep': '、',
        'expand': '展开全部原始记录',
        'viewer': '[查看原始文件]({link}) · [全部源码]({index})',
        'sources': ('# 源码与数据\n\n直接在浏览器阅读脚本，或下载完整练习包到本地执行。\n\n'
                    '[下载练习包 ZIP](../downloads/lora-lab.zip){{ .md-button .md-button--primary }}\n\n'
                    '练习包包含教程、Python 代码、测试、依赖清单和候选数据，不含模型权重、训练结果或本地环境。数据仍需审核。\n\n'
                    '[下载 GPU 训练 Notebook](../downloads/{notebook}) · [使用方法](../chapters/03-gpu.md)\n\n'
                    'Notebook 直接读取包内 data/dataset.json，需要与项目代码一起使用。'
                    '第 3 章的本地 GPU 上传包还包含分词器；上方通用源码练习包不包含模型缓存。\n\n'
                    '| 文件 | 阅读 |\n| --- | --- |\n'),
        'open': '打开',
    },
    'en': {
        'suffix': '.en',
        'docs_dir': '.book-en',
        'config': 'mkdocs.en.yml',
        'dataset': ('The only data file to maintain: {facts} fact cards, with conversations stored side by side per split ({sizes}). '
                    'facts supply the material each question may rely on; train/val/test supply the example conversations; '
                    '[field-by-field guide and where each field is used]({guide}).\n\n'
                    '## The first conversation and the facts it cites\n\n'
                    'Only one complete sample is shown below, keeping the top-level structure; all data is in the collapsed section and the original file.\n\n'),
        'size': '{split} {count} conversations',
        'size_sep': ', ',
        'expand': 'Expand all raw records',
        'viewer': '[View raw file]({link}) · [All sources]({index})',
        'sources': ('# Source Code and Data\n\nRead the scripts in your browser, or download the full practice bundle to run locally.\n\n'
                    '[Download practice bundle ZIP](../downloads/lora-lab.zip){{ .md-button .md-button--primary }}\n\n'
                    'The bundle contains the tutorial, Python code, tests, requirement lists and candidate data; '
                    'no model weights, training outputs or local environments. The data still needs review.\n\n'
                    '[Download the GPU training notebook](../downloads/{notebook}) · [How to use it](../chapters/03-gpu.md)\n\n'
                    'The notebook reads data/dataset.json from the bundle directly and must be used together with the project code. '
                    'The local GPU upload bundle from chapter 3 also contains the tokenizer; the general source bundle above contains no model cache.\n\n'
                    '| File | Read |\n| --- | --- |\n'),
        'open': 'Open',
    },
}


def localized(path: str, lang: str):
    """tasks/plan.md -> tasks/plan.en.md for English; Chinese keeps the original name."""
    stem, dot, ext = path.rpartition('.')
    return f'{stem}{LANGUAGES[lang]["suffix"]}.{ext}'


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
    paths = [root / 'README.md', root / 'README.en.md', root / 'pyproject.toml', root / '.gitignore', root / 'book_review.sh',
             root / 'GPU_UPLOAD_README.md', root / 'GPU_UPLOAD_README.en.md']
    paths += list(root.glob('requirements-*.txt'))
    for folder in ('lab', 'tests', 'tools'):
        paths += list((root / folder).glob('*.py'))
    paths += list((root / 'tasks').glob('*.md'))
    paths += [root / 'data/README.md', root / 'data/README.en.md', root / 'data/dataset.json']
    paths += [p for p in (root / 'images').iterdir() if p.suffix in ('.png', '.jpg')]
    paths += [root / 'notebooks/01_gpu_train.ipynb', root / 'notebooks/01_gpu_train.en.ipynb']
    paths += [root / 'lab/download-lock.json']
    return sorted(path for path in paths if path.is_file())


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_book(root: Path, output: Path, lang: str = 'zh'):
    ui = LANGUAGES[lang]
    plan, todo, verification, data_readme, notebook = (
        localized(p, lang) for p in ('tasks/plan.md', 'tasks/todo.md', 'tasks/verification.md',
                                     'data/README.md', 'notebooks/01_gpu_train.ipynb'))
    output.mkdir(parents=True, exist_ok=True)
    original = (root / plan).read_text(encoding='utf-8')
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
        plan: 'index.md',
        todo: 'reference/checklist.md',
        verification: 'reference/verification.md',
        data_readme: 'reference/data.md',
        localized('README.md', lang): 'index.md',
        notebook: 'downloads/' + notebook,
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
        body = rewrite(body, plan, page, routes, promote=True)
        write(output / page, f'# {title}\n\n{body.lstrip()}')
    intro = '\n'.join(intro.splitlines()[1:]).strip()
    intro = rewrite(intro, plan, 'index.md', routes)
    home = (root / localized('book/home.md', lang)).read_text(encoding='utf-8')
    write(output / 'index.md', home.replace('{{ introduction }}', intro))
    for source in (todo, verification, data_readme):
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
            sizes = ui['size_sep'].join(ui['size'].format(split=split, count=len(dataset[split]))
                                          for split in ('train', 'val', 'test'))
            body = (ui['dataset'].format(facts=len(dataset['facts']), sizes=sizes, guide=guide)
                    + f'```json\n{example}\n```\n\n'
                    f'<details markdown="1">\n<summary>{ui["expand"]}</summary>\n\n'
                    f'{body}\n</details>\n')
        # Depth varies between lab/*.py and data/dataset.json.
        index_link = posixpath.relpath('reference/sources.md', posixpath.dirname(target))
        write(output / target, f'---\nsearch:\n  exclude: true\n---\n\n# {rel}\n\n'
              f'{ui["viewer"].format(link=link, index=index_link)}\n\n'
              f'{body}')
    source_index = ui['sources'].format(notebook=notebook)
    for path in viewable:
        rel = path.relative_to(root).as_posix()
        source_index += f'| `{rel}` | [{ui["open"]}]({posixpath.relpath(routes[rel], "reference")}) |\n'
    write(output / 'reference/sources.md', source_index)
    (output / 'downloads').mkdir(exist_ok=True)
    for path in files:
        if path.suffix == '.ipynb':
            target = output / 'downloads' / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    with ZipFile(output / 'downloads/lora-lab.zip', 'w', ZIP_DEFLATED) as archive:
        # Include book files too, so the root README's preview instructions work.
        extras = list((root / 'book').rglob('*')) + [root / 'mkdocs.yml', root / 'mkdocs.en.yml',
                                                   root / '.book/.gitkeep', root / '.book-en/.gitkeep']
        extras += list((root / '.github/workflows').glob('*.yml'))
        for path in sorted(set(files + extras)):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path, 'llm-lora-tutorial/' + path.relative_to(root).as_posix())
    shutil.copytree(root / 'book/assets', output / 'assets', dirs_exist_ok=True)
    for image in images:
        shutil.copyfile(image, output / 'assets' / image.name)
    write(output / '.nojekyll', '')
    result = {'chapters': pages, 'source': plan, 'chapter_count': len(pages), 'lang': lang}
    write(output / 'book-manifest.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def config_language(config):
    return 'en' if config['theme']['language'] == 'en' else 'zh'


def on_config(config):
    root = Path(__file__).resolve().parents[1]
    lang = config_language(config)
    site_url = os.environ.get('BOOK_SITE_URL')
    if site_url:
        config['site_url'] = site_url.rstrip('/') + '/' + ('en/' if lang == 'en' else '')
    result = build_book(root, Path(config['docs_dir']), lang)
    result['base_path'] = urlsplit(config['site_url'] or '/').path
    write(Path(config['docs_dir']) / 'book-manifest.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return config


def on_post_build(config):
    # 中文站构建完成后，把英文站构建到同一站点的 en/ 子目录：
    # mkdocs build / serve 一条命令就得到可互相切换的双语站点。
    if config_language(config) != 'zh':
        return
    from mkdocs.commands.build import build
    from mkdocs.config import load_config
    root = Path(__file__).resolve().parents[1]
    english = load_config(str(root / LANGUAGES['en']['config']), site_dir=str(Path(config['site_dir']) / 'en'),
                          strict=config['strict'])
    build(english)
    # 托管平台只用站点根目录的 404.html；英文站自己的 404 页链接前缀不对，删掉。
    (Path(english['site_dir']) / '404.html').unlink(missing_ok=True)


def on_serve(server, config, builder):
    # Only watch authoritative sources. Watching generated Markdown causes a
    # rebuild to trigger another rebuild indefinitely.
    server.unwatch(config['docs_dir'])
    return server
