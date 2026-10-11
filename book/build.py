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

# 各语言共用同一套页面路径：中文站在根目录，其他语言在 <lang>/ 子目录。
# 译文是同名的 *.<lang>.md 文件；代码、数据和图片各语言共用。
# 只有中英文有 Notebook，日文、韩文版链接英文 Notebook（代码完全相同）。
LANGUAGES = {
    'zh': {
        'suffix': '',
        'docs_dir': '.book',
        'config': 'mkdocs.yml',
        'notebook': 'notebooks/01_gpu_train.ipynb',
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
        'notebook': 'notebooks/01_gpu_train.en.ipynb',
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
    'ja': {
        'suffix': '.ja',
        'docs_dir': '.book-ja',
        'config': 'mkdocs.ja.yml',
        'notebook': 'notebooks/01_gpu_train.en.ipynb',
        'dataset': ('メンテナンスするデータファイルはこれ 1 つだけです：ファクトカード {facts} 枚と、分割ごとに並べて保存した会話（{sizes}）。'
                    'facts は各質問が根拠にできる資料、train/val/test はお手本となる会話です。'
                    '[フィールドごとの説明と使われる場所]({guide})。\n\n'
                    '## 最初の会話と、それが参照するファクト\n\n'
                    '以下ではトップレベルの構造を保ったまま完全なサンプルを 1 つだけ示します。全データは折りたたみ部分と元ファイルにあります。\n\n'),
        'size': '{split} {count} 件',
        'size_sep': '、',
        'expand': 'すべての元データを展開',
        'viewer': '[元ファイルを表示]({link}) · [すべてのソース]({index})',
        'sources': ('# ソースコードとデータ\n\nブラウザでスクリプトを読むか、練習パッケージ一式をダウンロードしてローカルで実行できます。\n\n'
                    '[練習パッケージ ZIP をダウンロード](../downloads/lora-lab.zip){{ .md-button .md-button--primary }}\n\n'
                    'パッケージにはチュートリアル、Python コード、テスト、依存関係リスト、候補データが含まれます。'
                    'モデルの重み、学習結果、ローカル環境は含みません。データはまだレビューが必要です。\n\n'
                    '[GPU 学習ノートブックをダウンロード（英語版）](../downloads/{notebook}) · [使い方](../chapters/03-gpu.md)\n\n'
                    'ノートブックはパッケージ内の data/dataset.json を直接読み込むため、プロジェクトのコードと一緒に使ってください。'
                    '第 3 章のローカル GPU アップロード用パッケージにはトークナイザーも含まれます。上の汎用ソースパッケージにはモデルキャッシュは含まれません。\n\n'
                    '| ファイル | 閲覧 |\n| --- | --- |\n'),
        'open': '開く',
    },
    'ko': {
        'suffix': '.ko',
        'docs_dir': '.book-ko',
        'config': 'mkdocs.ko.yml',
        'notebook': 'notebooks/01_gpu_train.en.ipynb',
        'dataset': ('관리할 데이터 파일은 이것 하나뿐입니다: 팩트 카드 {facts}장과 분할별로 나란히 저장한 대화({sizes}). '
                    'facts는 각 질문이 근거로 삼을 수 있는 자료이고, train/val/test는 예시 대화입니다. '
                    '[필드별 설명과 사용 위치]({guide}).\n\n'
                    '## 첫 번째 대화와 그 대화가 인용하는 팩트\n\n'
                    '아래에는 최상위 구조를 유지한 채 완전한 샘플 하나만 보여 줍니다. 전체 데이터는 접힌 영역과 원본 파일에 있습니다.\n\n'),
        'size': '{split} {count}개',
        'size_sep': ', ',
        'expand': '전체 원본 레코드 펼치기',
        'viewer': '[원본 파일 보기]({link}) · [전체 소스]({index})',
        'sources': ('# 소스 코드와 데이터\n\n브라우저에서 스크립트를 읽거나, 실습 패키지 전체를 내려받아 로컬에서 실행할 수 있습니다.\n\n'
                    '[실습 패키지 ZIP 다운로드](../downloads/lora-lab.zip){{ .md-button .md-button--primary }}\n\n'
                    '패키지에는 튜토리얼, Python 코드, 테스트, 의존성 목록, 후보 데이터가 들어 있습니다. '
                    '모델 가중치, 학습 결과, 로컬 환경은 포함하지 않습니다. 데이터는 아직 검토가 필요합니다.\n\n'
                    '[GPU 학습 노트북 다운로드(영어판)](../downloads/{notebook}) · [사용 방법](../chapters/03-gpu.md)\n\n'
                    '노트북은 패키지 안의 data/dataset.json을 직접 읽으므로 프로젝트 코드와 함께 사용해야 합니다. '
                    '3장의 로컬 GPU 업로드 패키지에는 토크나이저도 들어 있고, 위의 일반 소스 패키지에는 모델 캐시가 없습니다.\n\n'
                    '| 파일 | 보기 |\n| --- | --- |\n'),
        'open': '열기',
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
    paths = [root / 'pyproject.toml', root / '.gitignore', root / 'book_review.sh', root / 'LICENSE']
    paths += list(root.glob('README*.md')) + list(root.glob('GPU_UPLOAD_README*.md'))
    paths += list(root.glob('requirements-*.txt'))
    for folder in ('lab', 'tests', 'tools'):
        paths += list((root / folder).glob('*.py'))
    paths += list((root / 'tasks').glob('*.md'))
    paths += list((root / 'data').glob('README*.md')) + [root / 'data/dataset.json']
    paths += [p for p in (root / 'images').iterdir() if p.suffix in ('.png', '.jpg')]
    paths += list((root / 'notebooks').glob('01_gpu_train*.ipynb'))
    paths += [root / 'lab/download-lock.json']
    return sorted(path for path in paths if path.is_file())


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def build_book(root: Path, output: Path, lang: str = 'zh'):
    ui = LANGUAGES[lang]
    plan, todo, verification, data_readme = (
        localized(p, lang) for p in ('tasks/plan.md', 'tasks/todo.md', 'tasks/verification.md', 'data/README.md'))
    notebook = ui['notebook']
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
        extras = list((root / 'book').rglob('*')) + list(root.glob('mkdocs*.yml'))
        extras += [root / settings['docs_dir'] / '.gitkeep' for settings in LANGUAGES.values()]
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
    language = config['theme']['language']
    return language if language in LANGUAGES else 'zh'


def on_config(config):
    root = Path(__file__).resolve().parents[1]
    lang = config_language(config)
    site_url = os.environ.get('BOOK_SITE_URL')
    if site_url:
        config['site_url'] = site_url.rstrip('/') + '/' + ('' if lang == 'zh' else f'{lang}/')
    result = build_book(root, Path(config['docs_dir']), lang)
    result['base_path'] = urlsplit(config['site_url'] or '/').path
    write(Path(config['docs_dir']) / 'book-manifest.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return config


def on_post_build(config):
    # 中文站构建完成后，把其他语言构建到同一站点的 <lang>/ 子目录：
    # mkdocs build / serve 一条命令就得到可互相切换的多语言站点。
    if config_language(config) != 'zh':
        return
    from mkdocs.commands.build import build
    from mkdocs.config import load_config
    root = Path(__file__).resolve().parents[1]
    for lang, settings in LANGUAGES.items():
        if lang == 'zh':
            continue
        edition = load_config(str(root / settings['config']), site_dir=str(Path(config['site_dir']) / lang),
                              strict=config['strict'])
        build(edition)
        # 托管平台只用站点根目录的 404.html；子站点自己的 404 页链接前缀不对，删掉。
        (Path(edition['site_dir']) / '404.html').unlink(missing_ok=True)


def on_serve(server, config, builder):
    # Only watch authoritative sources. Watching generated Markdown causes a
    # rebuild to trigger another rebuild indefinitely.
    server.unwatch(config['docs_dir'])
    return server
