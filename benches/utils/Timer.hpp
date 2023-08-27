#pragma once

#include <chrono>

class Timer {
    std::chrono::system_clock::time_point start;

   public:
    Timer() : start(std::chrono::system_clock::now()) {}

    void reset() {
        start = std::chrono::system_clock::now();
    }

    long long elapsed_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now() - start)
                .count();
    }

    long long elapsed_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now() - start)
                .count();
    }
};