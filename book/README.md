# 电子书维护与本地阅读

正文源文件是 `tasks/plan.md`，清单、验证记录和数据说明也直接来自原文件。
使用 MkDocs 1.6.1 + Material 9.7.0 生成静态网页；阅读不需要 GPU、模型权重、后端 API 或外部字体服务。

本次运行证据见 [电子书验收记录](verification.md)。

## 本地打开

在仓库根目录执行：

```bash
./book_review.sh
```

服务就绪后，脚本会自动在系统默认浏览器打开 <http://127.0.0.1:8000>；指定其他端口时打开对应地址。脚本会创建或复用 `.venv-book`，按锁定版本检查依赖并在必要时安装。首次使用需有 Python 3.11；也可从其他目录用脚本的绝对路径启动。

按 `Ctrl+C` 停止，重新执行脚本即可继续阅读。若端口已占用，运行 `./book_review.sh 8001`；脚本不会结束已有服务。不要用双击 HTML 的方式打开；搜索和目录页面需要通过 HTTP 访问。

开发预览会监听原教程和代码变化。首次安装依赖需要联网；安装完成后，构建、阅读、中文搜索、主题切换均不依赖外部 CDN。首次构建的中文分词会在临时目录建立小型缓存。

如果修改的是生成器 `book/build.py` 本身，重启预览进程以重新加载 Python 钩子；修改教程正文不需要重启。

## 构建与验证

```bash
source .venv-book/bin/activate
python -m pytest -q tests/test_book.py
python -m mkdocs build --strict
python -m book.check site
```

`site/` 是完整静态站点，可用任意静态服务器托管。`book.check` 检查 HTML 内的本地链接、资源和锚点。普通页面要求相对链接；404 页按实际站点前缀检查，因为它可能在任意层级出现。完整训练测试仍按主 README 在 CPU 训练环境中运行。

## 内容组织与设计决定

- **多语言**：默认中文，另有 English（`en/`）、日本語（`ja/`）、한국어（`ko/`）。中文站在站点根目录，其他语言在对应子目录，各版页面路径一一对应；右上角语言菜单由 `extra.alternate` 生成，`book/assets/language.js` 让切换后停留在同一页。译文是同名的 `*.<lang>.md`（`tasks/plan`、`tasks/todo`、`tasks/verification`、`data/README`、`book/home`）；英文另有 `notebooks/01_gpu_train.en.ipynb`，日文、韩文版链接英文 Notebook，代码块与英文版逐字相同。`mkdocs.<lang>.yml` 继承 `mkdocs.yml`，只覆盖语言、导航标题、搜索语言和界面文字；`mkdocs build` / `./book_review.sh` 构建中文站后，钩子按 `book/build.py` 的 `LANGUAGES` 自动把其他语言构建到 `site/<lang>/`（生成目录 `.book-<lang>/`）。修改中文正文时同步修改各译本；`tests/test_tutorial.py` 和 `tests/test_book.py` 检查各版章节数、代码块（除注释外）和 Notebook 代码一致。新增或重排章节时，`SLUGS` 与所有 `mkdocs*.yml` 的导航一起改。
- `book/build.py` 作为构建钩子，把教程的二级标题拆成 15 页，提升章内标题层级，并转换文件链接。代码块逐字保留。
- `.book/` 为自动生成的 Markdown、源码浏览页、下载包与静态资源；只纳入 `.gitkeep`，不要手工编辑生成内容。每次构建会重建源码浏览和原始源码下载目录，避免删除的模块残留在电子书中。
- `book/home.md` 管理扉页，`book/assets/book.css` 管理纸色与深色排版；`mkdocs.yml` 管理章节顺序、搜索和目录。
- 主导航是一层平铺目录：第 0～8 章按操作顺序排列，附录 A～E 放在最后，不设折叠分组。章节编号与 `SLUGS` 文件名一一对应；Notebook 和 `tasks/todo.md` 中引用的章节号随正文一起维护。
- 第 3 章的机器配置截图以 `images/gpu-rtx4090-24gb-2026-09-29.png` 为原图，构建器将它复制到站点资源目录并纳入源码下载包。
- `notebooks/01_gpu_train.ipynb` 是第 4、5 章的训练与部署入口；第 3 章负责环境和国内源下载，构建器提供原始文件下载并纳入源码练习包。GPU 上传包 `llm-lora-tutorial-gpu.zip` 另含本地分词器；两种包都包含唯一数据源 `data/dataset.json`。更新 Notebook、下载器、下载校验清单或依赖后，应同步更新 GPU 上传包；交付 Notebook 保持输出为空，不预填 GPU 成功记录。
- `tasks/plan.md`、`tasks/todo.md`、`tasks/verification.md`、`data/README.md` 是正文真源。新增或重排章节时，同步调整 `SLUGS` 与 `mkdocs.yml`，然后运行检查。
- 源码和数据链接指向本站可阅读的副本。单文件 JSON 数据页先展示用途、数量和第一段对话及其引用的事实，全部原稿放在折叠区；字段解释来自 `data/README.md`。JSON 代码块自动换行，复制内容保持原样。
- 电子书的通用源码下载包只收录明确列出的教材、代码和数据，不包含 `models/`、`runs/`、缓存或环境变量；GPU 上传包额外带本地分词器。轻量数据依赖见 `requirements-data.txt`，训练依赖到第 6 章再安装。
- 源码页不进入全文搜索，以免大段代码挤占概念与正文结果。训练代码附录仍可搜索。
- 选择成熟静态主题是为了复用中文搜索、键盘导航、手机抽屉目录、代码复制和章节翻页。没有额外实现客户端路由或定制搜索后端。

## 后续发布到 GitHub Pages

当前只完成本地电子书，尚未通过此工作流发布公网地址。

以后将本项目放到目标 GitHub 仓库根目录，在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。提供的 `.github/workflows/book-pages.yml` 会在 `main` / `master` 的推送后构建、校验并发布；PR 只构建与校验。也可在 Actions 页面手动运行。

若默认分支使用其他名称，要同步修改工作流的分支条件。工作流需要仓库允许 Pages、Actions，以及 `github-pages` 环境允许该发布分支。具体公开地址以 Actions 部署结果为准。失败的构建不会进入部署；回退内容后重新推送即可重新发布。

普通页面的站内资源使用相对路径，支持 `https://用户名.github.io/仓库名/`。工作流自动取得 Pages 地址，通过 `BOOK_SITE_URL` 设置规范地址与 404 页前缀，无需把本机地址写进配置。手动部署到子路径时，构建前也应设置这个环境变量。

依据：[Material 导航配置](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/)、[搜索插件](https://squidfunk.github.io/mkdocs-material/plugins/search/)、[字体配置](https://squidfunk.github.io/mkdocs-material/setup/changing-the-fonts/)、[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
