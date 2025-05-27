import sys
import os

# 添加 .so 路径（确保它指向 cpp_trading.so 所在的 build 目录）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../cpp_core/build")))

from cpp_trading import Order, OrderExecutor, OrderType

# 初始化执行器
executor = OrderExecutor()

# 创建一个订单
order = Order()
order.symbol = "AAPL"
order.type = OrderType.BUY
order.price = 180.5
order.quantity = 10
order.timestamp = 1680000000

# 提交并模拟执行
executor.submit_order(order)
executor.simulate_execution()

# 打印结果
print("✅ Filled orders:")
for filled in executor.get_filled_orders():
    side = "BUY" if filled.type == OrderType.BUY else "SELL"
    print(f"{side} {filled.quantity} {filled.symbol} @ ${filled.price}")
