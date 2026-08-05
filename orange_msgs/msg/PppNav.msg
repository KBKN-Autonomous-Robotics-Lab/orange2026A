std_msgs/Header header

string  pos_type      # "NONE" / "PPP_CONVERGING" / "PPP"
float64 latitude      # 緯度（PPPNAVA #3 → 擬似単位 dd+mm/100 に変換済み）
float64 longitude     # 経度（PPPNAVA #4 → 擬似単位 dd+mm/100 に変換済み）
float64 lat_sd        # 緯度σ [m]（PPPNAVA #8）
float64 lon_sd        # 経度σ [m]（PPPNAVA #9）
float64 alt_sd        # 高度σ [m]（PPPNAVA #10）
float64 diff_age      # 補正データ齢 [s]（PPPNAVA #12）
float64 sol_age       # 解の齢 [s]（PPPNAVA #13）
uint8   num_tracked   # 追尾衛星数（#14）
uint8   num_used      # 測位使用衛星数（#15）
bool    valid         # pos_type が NONE のとき false（σ=0の罠回避）


