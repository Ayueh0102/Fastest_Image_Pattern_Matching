#!/usr/bin/env python3
"""測試優化後的GUI界面"""

import sys
import os

# 確保可以導入模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fpm_py.gui import FpmGui
    print("✅ GUI模組導入成功")
    
    # 測試GUI初始化
    app = FpmGui()
    print("✅ GUI初始化成功")
    print(f"✅ 視窗標題: {app.title()}")
    print(f"✅ 視窗尺寸: {app.geometry()}")
    
    # 檢查是否有預設圖像載入
    src_path = app.ent_src.get()
    tmpl_path = app.ent_tmpl.get()
    
    if src_path:
        print(f"✅ 已載入來源圖像: {os.path.basename(src_path)}")
    else:
        print("⚠️  未載入預設來源圖像")
        
    if tmpl_path:
        print(f"✅ 已載入樣板圖像: {os.path.basename(tmpl_path)}")
    else:
        print("⚠️  未載入預設樣板圖像")
    
    print("\n🚀 GUI測試完成！可以運行 app.mainloop() 啟動界面")
    print("執行以下指令啟動GUI:")
    print("  python3 test_gui.py --run")
    
    if "--run" in sys.argv:
        print("\n啟動GUI界面...")
        app.mainloop()
    
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ GUI初始化失敗: {e}")
    sys.exit(1)