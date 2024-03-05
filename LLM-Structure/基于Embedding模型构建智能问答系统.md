## 基于Embedding模型构建智能问答系统
本部分通过一个简单的示例介绍如何抓取网站（本例中为 OpenAI 网站），使用Embedding API 将抓取的页面转化为Embedding并存储，
然后创建一个基本的搜索功能，允许用户基于存储的信息提问。完整代码请查看：。。。

### 使用Scrapy爬取网站信息
首先，导入所需的软件包，设置基本的 URL，并定义 HTMLParser 类。
```python
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os

# 用于匹配URL的正则表达式模式
HTTP_URL_PATTERN = r'^http[s]*://.+'

domain = "openai.com" # <- 将要爬取的域名放在这里
full_url = "https://openai.com/" # <- 将要爬取的域名以https或http形式放在这里

# 创建一个类来解析HTML并获取超链接
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # 创建一个列表来存储超链接
        self.hyperlinks = []

    # 重写HTMLParser的handle_starttag方法来获取超链接
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # 如果标签是锚标签并且有href属性，则将href属性添加到超链接列表中
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])
```
将 URL 作为参数，打开 URL 并读取 HTML 内容。然后，返回在该页面上找到的所有超链接。
```python
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os

# 用于匹配URL的正则表达式模式
HTTP_URL_PATTERN = r'^http[s]*://.+'

domain = "openai.com" # <- 将要爬取的域名放在这里
full_url = "https://openai.com/" # <- 将要爬取的域名以https或http形式放在这里

# 创建一个类来解析HTML并获取超链接
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # 创建一个列表来存储超链接
        self.hyperlinks = []

    # 重写HTMLParser的handle_starttag方法来获取超链接
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # 如果标签是锚标签并且有href属性，则将href属性添加到超链接列表中
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

```
我们的目标是只抓取 OpenAI 域名下的内容并编制索引。为此，我们需要一个调用 `get_hyperlinks` 函数的函数，但要过滤掉不属于指定域的任何 URL。
```python
# 从URL获取同一域名内的超链接的函数
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # 如果链接是一个URL，检查它是否在同一域名内
        if re.search(HTTP_URL_PATTERN, link):
            # 解析URL并检查域名是否相同
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # 如果链接不是URL，检查它是否是一个相对链接
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # 返回同一域名内的超链接列表
    return list(set(clean_links))

```
`crawl`是网络搜索任务设置的最后一步。它跟踪访问过的 URL，以避免重复访问同一页面，因为同一页面可能在网站的多个页面中都有链接。它还会提取网页中没有 HTML 标记的原始文本，并将文本内容写入特定于该网页的本地 .txt 文件中。
```python
def crawl(url):
    # 解析URL并获取域名
    local_domain = urlparse(url).netloc

    # 创建一个队列来存储待爬取的URLs
    queue = deque([url])

    # 创建一个集合来存储已经看到的URLs（无重复）
    seen = set([url])

    # 创建一个目录来存储文本文件
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # 创建一个目录来存储csv文件
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # 当队列不为空时，继续爬取
    while queue:

        # 从队列中获取下一个URL
        url = queue.pop()
        print(url) # 用于调试和查看进度

        # 将url的文本保存到<url>.txt文件中
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            # 使用BeautifulSoup从URL获取文本
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # 获取文本但去除标签
            text = soup.get_text()

            # 如果爬虫到达需要JavaScript的页面，它将停止爬取
            if ("You need to enable JavaScript to run this app." in text):
                print("无法解析页面 " + url + " 因为需要启用JavaScript")

            # 否则，将文本写入text目录下的文件
            f.write(text)

        # 从URL获取超链接并将它们添加到队列中
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

crawl(full_url)
```
上述示例的最后一行运行爬网程序，该程序会浏览所有可访问链接，并将这些页面转化为文本文件。根据网站的大小和复杂程度，这需要几分钟的时间。

### 构建Embedding索引
要将文本转换为 CSV，需要循环浏览之前创建的文本目录中的文本文件。打开每个文件后，删除多余的行距，并将修改后的文本添加到列表中。然后，将删除了新行的文本添加到一个空的 Pandas 数据框中，并将数据框写入 CSV 文件。
```python
import pandas as pd

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# 创建一个列表来存储文本文件
texts=[]

# 获取text目录中的所有文本文件
for file in os.listdir("text/" + domain + "/"):

    # 打开文件并读取文本
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # 省略前11行和最后4行，然后将-、_和#update替换为空格。
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# 从文本列表创建一个dataframe
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# 将text列设置为删除了换行符的原始文本
df['text'] = df['fname'] + ". " + df['text'].replace('\n', ' ')
df.to_csv('processed/scraped.csv')
df.head()
```
