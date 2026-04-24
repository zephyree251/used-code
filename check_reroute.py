import random
from core.context import SystemContext
from utils.tefnet_loader import TefnetLoader


def reroute_logic():
    print("🧪 [测试开始] 正在初始化环境...")

    # 1. 初始化环境 (和 main.py 一样)
    ctx = SystemContext()
    try:
        # 确保你的数据文件路径是对的
        loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
        ctx.graph = loader.load_topology()
        ctx.all_demands = loader.load_traffic_demands()
        ctx.active_services = ctx.all_demands[10:100]  # 只取前5个业务做测试
        print(f"✅ 环境加载成功，当前有 {len(ctx.active_services)} 条业务。")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 挑选一个幸运儿 (业务)
    target_service = ctx.active_services[0]
    print(f"🔍 调试信息：当前业务的所有字段名: {list(target_service.keys())}")
    s_id = target_service['id']

    # 记录“整容前”的样子
    print("\n------------------------------------------------")
    print(f"🧐 [整容前] 业务 {s_id}")
    print(f"   当前路径: {target_service.get('path', 'Unknown')}")
    print(f"   当前OSNR: {target_service.get('osnr', 0):.2f} dB")
    print("------------------------------------------------")

    # 3. 强制执行重路由动作！
    print(f"⚡ [动作执行] 正在对 {s_id} 执行重路由 (Reroute)...")
    success, log = ctx.execute_reroute(s_id)

    # 4. 验证结果
    if success:
        print("\n🎉 [整容成功]！")
        print(f"   日志反馈: {log}")

        # 再次检查数据对象，确认是否真的变了
        print(f"\n🧐 [整容后] 业务 {s_id}")
        print(f"   新路径: {target_service.get('path')}")
        print(f"   新OSNR: {target_service.get('osnr', 0):.2f} dB")

        if target_service.get('is_rerouted'):
            print("✅ 标记位 'is_rerouted' 已正确置为 True")
        else:
            print("❌ 警告：标记位未更新！")
    else:
        print(f"\n❌ [失败] 重路由失败: {log}")


if __name__ == "__main__":
    reroute_logic()