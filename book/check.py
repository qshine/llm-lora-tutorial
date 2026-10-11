"""Validate the built website's internal pages, anchors, and local assets."""
from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for name in ('href', 'src'):
            if attrs.get(name):
                self.links.append(attrs[name])


def check_site(directory: Path, base_path='/'):
    directory = directory.resolve()
    pages = {p.resolve(): Page(p) for p in directory.rglob('*.html')}
    if not pages:
        raise ValueError('No HTML pages found; build the book first')
    failures, checked = [], 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            # A 404 can be served at any depth, so MkDocs uses the site prefix
            # (the English edition's 404 under en/ uses the same prefix plus en/).
            # Normal pages must remain relative to work at a project subpath.
            if url.path.startswith('/'):
                if path.name != '404.html' or not url.path.startswith(base_path):
                    failures.append(f'{path.relative_to(directory)}: non-portable link {link}')
                    continue
                target = (directory / unquote(url.path[len(base_path):])).resolve()
            else:
                target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= 'index.html'
            checked += 1
            if not target.is_relative_to(directory) or not target.is_file():
                failures.append(f'{path.relative_to(directory)}: missing {link}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                failures.append(f'{path.relative_to(directory)}: missing anchor {link}')
    if failures:
        raise ValueError('\n'.join(failures))
    return {'pages': len(pages), 'internal_links_and_assets': checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path, nargs='?', default=Path('site'))
    directory = parser.parse_args().directory
    manifest = json.loads((directory / 'book-manifest.json').read_text())
    print(check_site(directory, base_path=manifest['base_path']))
