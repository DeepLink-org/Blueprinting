# 文档维护指南

Blueprinting 文档是一项版本化工程产品。它使用唯一信息架构、同目录中英文 source pair、strict build 与显式 capability status，使设计意图不会悄悄偏离实现现实。

## 站点架构

文档按读者意图组织：

```text
docs/
├── index.{en,zh}.md                 orientation
├── exploration/                     产品判断、设计空间、工作流
├── modeling/                        hardware 与 workload model
├── design/                          形式化分析基础
│   ├── ir/                          representation contracts
│   ├── passes/                      transformation contracts
│   └── performance/                 evidence and simulation contracts
├── experiments/                     reproducible validation reports
├── project/                         status, roadmap, decisions
├── contributing/                    maintenance guides
├── overrides/home.html              双语产品首页
└── assets/
    ├── architecture/                共享技术图
    └── stylesheets/site.css         站点外壳与首页视觉系统
```

顶层 narrative 从 hardware question、candidate blueprint、model、simulation 与 decision output 开始；形式化分析 reference page 再解释使这些比较可复现的 model、derivation、verification obligation 与 automated analysis；experiment page 报告 evidence；project page 区分当前状态与已接受的未来设计。

## 首页与站点外壳

首页使用一份 locale-aware Material override，而不是分别维护大段中英文 HTML。`index.{en,zh}.md` 保留可搜索的 Markdown orientation，并通过 front matter 选择 `home.html`；override 根据 active locale 渲染产品导航。共享样式位于 `assets/stylesheets/site.css`，为 landing page 与普通 reference page 提供克制、统一的 Blueprinting 视觉系统。

Landing page 从硬件决策、探索闭环、证据阶梯、能力状态与读者路径开始，不得把产品重新叙述成通用 compiler、simulator 或 dashboard。首页上的产品 claim 必须服从[实现状态](../project/status.md)中的 capability matrix。

## 双语 Source Contract

站点使用 `mkdocs-static-i18n` 的 suffix structure：

- 英文是 default locale，发布在 `/`；
- 简体中文发布在 `/zh/`；
- 翻译在同一目录中成对保存为 `page.en.md` 与 `page.zh.md`；
- 只为复用 language-neutral static asset 启用 default-locale fallback；强制仓库检查要求每个 Markdown 页面同时存在两种语言，因此正文不会静默回退；
- Material language selector 会链接到另一语言的对应页面；
- search 按 locale 分别重建。

这遵循插件的 [suffix-structure guidance](https://ultrabug.github.io/mkdocs-static-i18n/setup/choosing-the-structure/) 与 [Material integration](https://ultrabug.github.io/mkdocs-static-i18n/setup/setting-up-material/)。不要在页面内手工增加中英文切换链接。

## Canonical Navigation 与链接

`mkdocs.yml` 和所有站内 Markdown link 都使用 language-neutral canonical path：

```markdown
[推导与验证模型](../design/compilation-model.md)
```

禁止链接到 `compilation-model.en.md`、`compilation-model.zh.md` 或 generated `/zh/` URL。i18n plugin 会为 active locale 解析 canonical path，并保持 language selector 对齐。

外部链接使用普通 absolute HTTPS URL。共享 SVG 使用相对路径。Language-specific asset 应使用相同 `.en`/`.zh` pairing convention 并说明必要性；在可行时，diagram 应偏好 language-neutral label。

## 添加或移动页面

新增页面时：

1. 按 reader intent 选择 owning section；
2. 在同一目录创建 `name.en.md` 与 `name.zh.md`；
3. 两个页面各有一个 H1，并保持相同 heading hierarchy；
4. 在 `mkdocs.yml` 增加唯一 canonical `name.md` entry；
5. 增加 navigation label 及对应中文 `nav_translations`；
6. 站内 link 只使用 canonical path；
7. 提交前运行文档检查。

移动页面时，同时更新两份 source、navigation、inbound link 与任何 public redirect policy。不要保留第二份非正式 translation copy。

## 写作 Contract

产品设计页从 hardware architecture decision 开始，解释 design variable、workload coverage、fidelity、output 与 claim boundary；工程页再解释 invariant、data ownership、algorithm、failure behavior、observability、status 与 implementation mapping。文档必须区分：

- semantic fact 与 performance evidence；
- accepted target architecture 与 current implementation；
- canonical IR 与 derived view/published artifact；
- verifier obligation 与 profiler comparison；
- measured result 与 inference/hypothesis。

Compiler 不得被描述为系统组件或顶层产品定义。产品方法是形式化建模、推导、验证与自动分析。只有在解释 workload mapping、semantic conservation、target binding 背后的借用技术或当前源码标识时，才使用 IR、lowering 与 pass 术语。

只有稳定规范要求才使用 **必须/不得/应该/不应该/可以**；英文页使用约定的对应词。Code identifier 保持 canonical spelling；两种语言的 prose 都应自然，不做机械逐词翻译。

## 翻译同步

中英文页面在结构和技术含义上对等。只有两边都更新后，change 才完整。Table、status label、equation、code、link、figure identity 与 claim boundary 必须一致；自然语言例子可以本地化。

`scripts/check_docs_i18n.py` 会强制完整 pair、唯一 H1、一致 heading shape 与 canonical link usage；正是这个 gate 使 shared-asset fallback 不会污染正文。它不能证明语义等价，因此 reviewer 仍需人工比较 claim 和 status。

## 状态与决策更新

描述未来架构的文档要在相应章节附近标记 **Planned** 或 **Contract Only**。只有仓库代码和相称 test 才能证明 **Implemented**。连接或移除 capability 的同一个 change 必须更新[实现状态](../project/status.md)。

改变 schema ownership、lowering gate、plugin protocol、evidence semantic、serialized identity 或 compatibility promise 时，需要按照[设计决策](../project/decisions.md)创建 ADR。Roadmap item 不能证明已经实现。

## 构建与评审

安装并验证：

```bash
uv sync --locked --no-dev --extra docs
uv run --no-dev --extra docs python scripts/check_docs_i18n.py
uv run --no-dev --extra docs mkdocs build --strict
```

同时运行形式化分析 test suite 的 contributor 可以去掉 `--no-dev`；default development dependency group 包含测试与 legacy workbench 依赖。

只预览单一语言时：

```bash
BUILD_ONLY_LOCALE=en uv run mkdocs serve
BUILD_ONLY_LOCALE=zh uv run mkdocs serve
```

评审前检查两个 locale route、逐页 language switching、navigation、search、table、code block 与 SVG rendering。`navigation.instant` 保持关闭，因为 i18n plugin 明确记录它与 language reconfiguration 不兼容。

## 持续交付

`.github/workflows/docs.yml` 会在文档 pull request 与 `main` 上运行双语 contract check 和 strict site build。Pull request 会把精确的 `site/` output 保留为带版本的 preview artifact；`main` 成功运行后，则会把同一份 output 打包为 GitHub Pages artifact，并通过受保护的 `github-pages` environment 部署到 [deeplink-org.github.io/Blueprinting](https://deeplink-org.github.io/Blueprinting/)。`docs/` 下的 Markdown 始终是唯一的文档 source tree；部署 artifact 是生成且不可变的 output，不是另一份可编辑 source。

## Review Checklist

- 两个 locale source 都存在且结构一致。
- Claim 明确区分 Implemented、Contract Only 与 Planned。
- Canonical link 在两个 generated locale tree 中都有效。
- 新术语已定义或沿用现有项目词汇。
- Figure 有有意义的 alt text，在 light/dark context 中均清晰。
- 可复现结果标明 command、input、revision 与 claim boundary。
- Design change 映射到 source/test，或明确声明尚无实现。
- `check_docs_i18n.py` 与 `mkdocs build --strict` 通过。
- 首页与普通文档页在两种 color scheme 和窄屏下均保持可用。

文档债务应像工程债务一样处理：明确 ownership boundary，增加能发现 regression 的 gate，并删除被取代 source，而不是维护有歧义的 duplicate。
