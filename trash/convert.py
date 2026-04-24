import pandas as pd

def convert_tefnet_data():
    print("正在提取 TEFNET24 数据...")

    # --- 1. 提取节点 (Nodes) ---
    # 源文件包含多个 Sheet，我们指定读取 'Optical+IP Nodes' 这一页
    print("正在转换节点表...")
    df_nodes = pd.read_excel('National_Network_20240615.xlsx', sheet_name='Optical+IP Nodes')
    # 保存为 CSV，index=False 表示不保存行号
    df_nodes.to_csv('tefnet_nodes.csv', index=False, encoding='utf-8')

    # --- 2. 提取链路 (Links) ---
    # 指定读取 'Fibre Links' 这一页
    print("正在转换链路表...")
    df_links = pd.read_excel('National_Network_20240615.xlsx', sheet_name='Fibre Links')
    df_links.to_csv('tefnet_links.csv', index=False, encoding='utf-8')

    # --- 3. 提取业务流量 (Traffic) ---
    # 指定读取 'HL1-HL2-HL3 traffic matrix' 这一页
    print("正在转换流量表...")
    df_traffic = pd.read_excel('Traffic_Matrix_20240822.xlsx', sheet_name='HL1-HL2-HL3 traffic matrix')
    df_traffic.to_csv('tefnet_traffic.csv', index=False, encoding='utf-8')

    print("✅ 转换完成！也就是你现在的目录下应该多了 tefnet_nodes.csv, tefnet_links.csv, tefnet_traffic.csv 这三个文件。")

if __name__ == "__main__":
    # 运行前请确保安装了 openpyxl: pip install openpyxl pandas
    try:
        convert_tefnet_data()
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件。请确保Excel文件在当前目录下且名字正确。\n详细信息: {e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")