from copy import copy
from datetime import date
from pathlib import Path
from textwrap import indent
from urllib.parse import urljoin, urlparse

import requests
import requests_cache
from bs4 import BeautifulSoup

requests_cache.install_cache(
    cache_name='scraper_cache', backend='sqlite', expire_after=86400
)


url = 'https://docs2bm.readthedocs.io/en/latest/index.html'
baseurl = urljoin(url, '.')

savedir = 'APS-Docs'
Path(f'./{savedir}').mkdir(exist_ok=True)

today = date.today()  # .strftime("%Y-%b%d")


# -------- start scraping ----------


def download(link):
    print(f'scraping {link}')
    response = requests.get(link)
    page = BeautifulSoup(response.content, 'lxml')
    content = page.select_one('div.document')
    return response, page, content


response, index_page, content = download(url)
if response.from_cache:
    print('[ WARNING: page requested from cache ]\n')

title = index_page.title.string
out = open(f"{savedir}/{today}_{title.replace(' ','_')}.txt", 'w')


def write(text):
    # print(f"{text}\n")
    out.write(f"{text}\n\n")


write(f"url: {url}\ntitle: {title}\naccessed: {today}\n")


def get_text_with_link(tag):
    text = ''
    for s in tag.contents:
        text += s.text
        if s.name == 'a' and '//' in s['href']:
            text += f" ({s['href']})"
    return text.strip()


def read(content, base=baseurl, hlevel=1):

    for tag in content.descendants:

        name = str(tag.name)

        if name[0] == 'h':
            heading = tag.contents[0].string
            yield '#' * hlevel + f" {heading}"

        elif name == 'p':
            if tag.parent.name in ('td', 'th', 'li'):
                continue
            yield get_text_with_link(tag)

        elif name == 'div':
            if 'class' not in tag.attrs:
                continue
            elif 'line' in tag['class']:
                text = str(tag.text).strip()
                if text:
                    yield text

        elif name == 'pre':
            text = str(tag.text).strip()
            yield indent(f"{text}", '  ')

        elif name == 'img':
            imgfile = Path(tag['src']).name
            yield f"  <image {imgfile}: {tag['alt']}>"

        elif name == 'ul':

            if tag.parent.name in ('li',):
                continue

            list_str = ''
            levels = [0]
            items = tag.find_all(['li'])

            for item in items:
                current_level = len(item.find_parents('ul')) - 1

                # bullet point style
                if len(items) < 6:
                    bullet = '•'
                else:
                    for _ in range(current_level + 1 - len(levels)):
                        levels.append(0)
                    for i in range(current_level + 1, len(levels)):
                        levels[i] = 0
                    levels[current_level] += 1
                    bullet = (
                        '.'.join(map(str, levels[: current_level + 1])) + '.'
                    )

                # remove sublist
                textobj = copy(item)
                sublist = textobj.ul
                if sublist is not None:
                    sublist.decompose()

                list_str += (
                    ' ' * current_level
                    + f"{bullet} {get_text_with_link(textobj)}\n"
                )
            yield list_str.strip()

        elif name == 'table':

            # extract text from cell and max width of columns
            trs = [tr.find_all(['td', 'th']) for tr in tag.find_all('tr')]
            ws = [0] * len(trs[0])
            table = []
            for tr in trs:
                table.append([str(cell.text).strip() for cell in tr])
                ws = [max(c0, len(c)) for c0, c in zip(ws, table[-1])]

            # format table rows as strings
            table_str = ''
            for i, row in enumerate(table):
                row = ' '.join(
                    [text + ' ' * (c - len(text)) for text, c in zip(row, ws)]
                )
                table_str += f"{row.strip()}\n"
                if i == 0:
                    table_str += '-' * len(row) + '\n'

            yield table_str.strip()

        elif name == 'a':

            if tag.parent.name in ('p',):
                continue

            href = tag['href']
            link = urlparse(href)

            if not link.path or tag.find('img'):
                continue
            elif not link.scheme and not link.netloc:
                href = base + link.path

            href = href.split(';')[0]

            # non-html links, e.g., images
            if not Path(href).suffix[:4] == '.htm':
                yield f"<{href}>"
            # external links
            elif not href.startswith(baseurl):
                yield f"<{href}>"
            # visited
            elif href in visited:
                continue
            # internal links
            else:
                response, page, content = download(href)
                if response.status_code != 200:
                    yield f"<{href}> (cannot access)"
                    continue
                # title = page.title.string
                # yield f"url: {href}\ntitle: {title}"
                newbase = urljoin(href, '.')
                for line in read(content, newbase, hlevel=hlevel + 1):
                    yield line
                visited.add(href)


visited = set([url])
for line in read(content):
    write(line)

print(f"output: {out.name}")
out.close()
