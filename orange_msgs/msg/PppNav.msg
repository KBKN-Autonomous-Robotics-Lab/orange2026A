std_msgs/Header header

string  pos_type      # "NONE" / "PPP_CONVERGING" / "PPP"
float64 lat_sd        # 緯度σ [m]（PPPNAVA #8）
float64 lon_sd        # 経度σ [m]（PPPNAVA #9）
float64 alt_sd        # 高度σ [m]（PPPNAVA #10）
float64 sol_age       # 収束時間 [s]（PPPNAVA #13）
uint8   num_tracked   # 追尾衛星数（#14）
uint8   num_used      # 測位使用衛星数（#15）
bool    valid         # pos_type が NONE のとき false（σ=0の罠回避）
