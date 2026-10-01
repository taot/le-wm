# World Model 讲解 slides

用 [Slidev](https://sli.dev) 做的一套 slides，讲 world model 的定义、核心难题、技术路线、评测、可解释性，
以及本仓库的 LeWorldModel。内容全部在 [slides.md](slides.md) 里。

## 运行

```bash
cd tutorials/slides
pnpm install
pnpm dev
```

然后打开 <http://localhost:3030>。常用按键：方向键翻页，`o` 总览，`d` 深色模式，`p` 演讲者模式（带备注）。左下角导航栏最右边的列表图标打开**目录**，点任意一项跳到那一页；每页右下角显示"当前页 / 总页数"。

在 RunPod 上跑、本机浏览器看：

```bash
ssh -L 3030:localhost:3030 runpod
cd /workspace/le-wm/tutorials/slides && pnpm install && pnpm dev
```

## 导出

- 静态网站：`pnpm build`，输出在 `dist/`
- PDF：把 `pnpm-workspace.yaml` 里的 `playwright-chromium` 改成 `true`，`pnpm install` 一次（会下载 Chromium），再 `pnpm export`

## 素材

- `public/lewm.gif` 复制自仓库根目录的 `assets/lewm.gif`
