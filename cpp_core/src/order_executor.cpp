#include "order_executor.hpp"
#include <iostream>

OrderExecutor::OrderExecutor() {}

void OrderExecutor::submit_order(const Order& order) {
    order_queue.push_back(order);
}

void OrderExecutor::simulate_execution() {
    for (const auto& order : order_queue) {
        fill_order(order);
    }
    order_queue.clear();
}

void OrderExecutor::fill_order(const Order& order) {
    std::cout << "[Mock Fill] " 
              << (order.type == OrderType::BUY ? "BUY " : "SELL ")
              << order.quantity << " of " << order.symbol << " at $" << order.price << std::endl;
    filled_orders.push_back(order);
}

std::vector<Order> OrderExecutor::get_filled_orders() const {
    return filled_orders;
}
