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


url = 'https://yongtaoliu.github.io/aecroscopy.pyae/welcome_intro.html'
navigator_selector = 'div.bd-sidebar'
article_selector = 'article.bd-article'

baseurl = urljoin(url, '.')

savedir = 'DOC_STORE/AIT-Docs'
Path(f'./{savedir}').mkdir(exist_ok=True)

today = date.today()  # .strftime("%Y-%b%d")


# -------- start scraping ----------


def download(link, selection):
    print(f'scraping {link}')
    response = requests.get(link)
    page = BeautifulSoup(response.content, 'lxml')
    content = page.select_one(selection)
    return response, page, content


response, index_page, content = download(url, navigator_selector)

if response.from_cache:
    print('[ WARNING: page requested from cache ]\n')

title = index_page.title.string
out = open(f"{savedir}/{title.replace(' ','_')}.txt", 'w', encoding='utf-8')


def write(text):
    out.write(text)


write(f"url: {url}\ntitle: {title}\naccessed: {today}\n\n\n")


def format_link(tag, baseurl=baseurl):
    href = tag['href']
    link = urlparse(href)
    if not link.path:
        return href
    elif not link.scheme and not link.netloc:
        href = urljoin(baseurl, link.path)
    href = href.split(';')[0]
    return href


def show_link(href, text=''):
    if text == href:
        return False
    elif href[0] == '#':
        return False
    elif href.startswith('mailto:'):
        return False
    elif href.startswith(baseurl) and Path(href).suffix[:4] == '.htm':
        return False
    return True


def get_text_with_link(tag, newline=2, baseurl=baseurl):
    text = ''
    find_link = True
    # <p>some text <a></a> with link<p>
    for s in tag.contents:
        text += s.text
        if s.name == 'a':
            find_link = False
            href = format_link(s, baseurl=baseurl)
            if show_link(href, s.text):
                text += f" ({href})"
    # <p><a>link only</a><p>
    if find_link and tag.a is not None:
        href = format_link(tag.a, baseurl=baseurl)
        if show_link(href, tag.text):
            text += f" ({href})"
    return text.strip() + '\n' * newline


def format_table(tag):
    # extract text from cell and max width of columns
    trs = [tr.find_all(['td', 'th']) for tr in tag.find_all('tr')]
    ws = [0] * len(trs[0])
    table = []
    for tr in trs:
        table.append([])
        for cell in tr:
            imgs = cell.find_all('img')
            if len(imgs) > 1:
                table[-1].append(
                    ','.join([Path(img['src']).name for img in imgs])
                )
            else:
                table[-1].append(str(cell.text).strip())
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

    return table_str.strip()


def format_list(tag, baseurl):
    list_str = ''
    levels = [0]
    items = tag.find_all(['li'])

    for item in items:
        current_level = len(item.find_parents(['ul', 'ol'])) - 1

        # remove sublist
        textobj = copy(item)
        sub_ul = textobj.ul
        if sub_ul is not None:
            sub_ul.decompose()
        sub_ol = textobj.ol
        if sub_ol is not None:
            sub_ol.decompose()

        # skip empty list item
        item_text = get_text_with_link(textobj, baseurl=baseurl, newline=0)
        if not item_text.strip() and len(items) == 1:
            continue

        # bullet point style
        if tag.name == 'ul' and len(items) < 6:
            bullet = '•'
        else:
            for _ in range(current_level + 1 - len(levels)):
                levels.append(0)
            for i in range(current_level + 1, len(levels)):
                levels[i] = 0
            levels[current_level] += 1
            bullet = '.'.join(map(str, levels[: current_level + 1])) + '.'

        list_str += ' ' * current_level + f"{bullet} {item_text}\n"

    return list_str.strip()


def read(content, base=baseurl, hlevel=0):
    if content is None:
        # TODO: this happened when scarping ipynb, nbconvert can do ipynb to html
        return

    for tag in content.descendants:
        name = str(tag.name)

        if name != 'a' and base == baseurl:
            continue

        if name[0] == 'h' and tag.contents:
            heading = tag.contents[0].string
            pgindent = ('\n|' + '-' * hlevel + ' ') * (hlevel > 0)
            yield pgindent + '#' * int(name[1:]) + f" {heading}\n\n"

        elif name == 'p':
            if tag.parent.name in ('td', 'th', 'li'):
                continue
            yield get_text_with_link(tag, baseurl=base)

        elif name == 'div' and 'line' in tag.get('class', ()):
            yield get_text_with_link(tag, baseurl=base, newline=1)

        elif name == 'span' and 'brackets' in tag.get('class', ()):
            yield '[' + tag.text + '] '

        elif name == 'pre':
            text = str(tag.text).strip()
            yield indent(f"{text}", '  ') + '\n\n'

        elif name == 'img':
            imgfile = Path(tag['src']).name
            if tag.has_attr('alt'):
                yield f"  <image {imgfile}: {tag['alt']}>\n\n"
            else:
                yield f"  <image {imgfile}: {tag['class']}>\n\n"

        elif name in ('ul', 'ol'):
            if tag.parent.name in ('li',):
                continue
            list_str = format_list(tag, baseurl=base)
            yield list_str + '\n\n' * bool(list_str)

        elif name == 'table':
            table_str = format_table(tag)
            yield table_str + '\n\n' * bool(table_str)

        elif name == 'a':
            href = format_link(tag, baseurl=base)
            is_internal_link = href.startswith(baseurl)
            # is_html = Path(href).suffix[:4] == '.htm'

            # scrap internal links only
            if is_internal_link and href not in visited:
                response, page, content = download(href, article_selector)
                if response.status_code != 200:
                    yield f"<{href}> (cannot access)\n\n"
                    continue

                visited.add(href)
                # title = page.title.string
                # yield f"url: {href}\ntitle: {title}\n\n"
                newbase = urljoin(href, '.')
                for line in read(content, newbase, hlevel=hlevel + 1):
                    yield line


visited = set([url])
for line in read(content):
    write(line)

print(f"output: {out.name}")
out.close()
