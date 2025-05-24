#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "data_feed.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_trading, m) {
    py::class_<Row>(m, "Row")
        .def_readwrite("date", &Row::date)
        .def_readwrite("open", &Row::open)
        .def_readwrite("high", &Row::high)
        .def_readwrite("low", &Row::low)
        .def_readwrite("close", &Row::close)
        .def_readwrite("volume", &Row::volume);

    py::class_<DataFeed>(m, "DataFeed")
        .def(py::init<>())
        .def("load", &DataFeed::load)
        .def("next", &DataFeed::next)
        .def("current", &DataFeed::current)
        .def("moving_average", &DataFeed::moving_average);
}
