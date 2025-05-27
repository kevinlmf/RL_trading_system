// File: include/order_executor.hpp
#pragma once

#include <string>
#include <vector>

enum class OrderType { BUY, SELL };

struct Order {
    std::string symbol;
    OrderType type;
    double price;
    int quantity;
    long timestamp;
};

class OrderExecutor {
public:
    OrderExecutor();

    void submit_order(const Order& order);

    void simulate_execution();  // For mock/backtest

    std::vector<Order> get_filled_orders() const;

private:
    std::vector<Order> order_queue;
    std::vector<Order> filled_orders;

    void fill_order(const Order& order);
};
