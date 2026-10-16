# 本地绘图审计

本目录随B前台流程保存nature-figure审计脚本的副本，运行时只需Python与PyMuPDF，不调用Codex、MCP或生成模型。脚本行为保持原样；绘图导出复用`analyze_performance.export_figure`。

- `validate_figure.py`：Python源代码预检。布局尺寸、TIFF及统计表达等启发式WARN须与使用合同一起解释，不冒充渲染结果。
- `audit_panel_alignment.py`：最终Matplotlib plot-area几何，1.5pt容差，严格模式；导出前调用。
- `audit_pdf_text.py`：导出PDF实际字形，最小5pt。
- `audit_figure_collisions.py`：PyMuPDF文字/路径/边界碰撞；本流程使用`--strict`，WARN也阻塞自动审计通过。

项目交付合同是300dpi PNG与可编辑PDF/SVG，不是某期刊的正式送审图。所有seed单独展示，没有对训练seed做均值/区间。源预检的已解释WARN写入`automatic_audit.json`。人工视觉检查始终为“运行后待检查”。缺依赖、无法审计、非零退出或缺报告不会标记通过。
