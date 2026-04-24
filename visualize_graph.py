from main_graph import build_ooda_graph
from langchain_core.runnables.graph import MermaidDrawMethod

# 1. 构建图
app = build_ooda_graph()

# 2. 获取图对象
graph = app.get_graph()

# 3. 生成 Mermaid 图片数据 (PNG)
# 注意：这需要联网调用 Mermaid 渲染服务，或者本地有 graphviz
# 如果报错，我们可以换成打印文本格式
try:
    png_data = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)

    with open("ooda_system_architecture.png", "wb") as f:
        f.write(png_data)

    print("✅ 成功！架构图已保存为 'ooda_system_architecture.png'")
    print("快去打开图片看看吧！这就是你系统的‘真面目’。")

except Exception as e:
    print("❌ 生成图片失败（可能是网络原因），为您打印文本结构：")
    print(graph.draw_ascii())