import re, urllib.request, urllib.parse, os
html = open('index.html', encoding='utf-8').read()
chars = set(html) | set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,:;/·→¥—–…、，。：？！「」（）')
chars = ''.join(sorted(c for c in chars if c.isprintable() and c not in '\n\t'))
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36'
fams = [('Noto Serif SC', '500;700;900'), ('Noto Sans SC', '300;400;500;700'), ('JetBrains Mono', '400;500')]
css_all = ''
for fam, w in fams:
    url = 'https://fonts.googleapis.com/css2?' + urllib.parse.urlencode({'family': f'{fam}:wght@{w}', 'text': chars, 'display': 'block'})
    css = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA})).read().decode()
    for i, u in enumerate(re.findall(r'url\((https://[^)]+)\)', css)):
        fn = f"{fam.replace(' ', '')}-{len(os.listdir('fonts'))}.woff2"
        open('fonts/' + fn, 'wb').write(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': UA})).read())
        css = css.replace(u, fn)
    css_all += css
open('fonts/fonts.css', 'w').write(css_all)
print(len(chars), 'chars'); print(css_all[:600])
