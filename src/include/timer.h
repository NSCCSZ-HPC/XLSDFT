#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>

class Timer
{
public:
    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::duration elapsed = std::chrono::steady_clock::duration::zero();
    inline void start() {this->t0 = std::chrono::steady_clock::now();};
    inline void stop() {this->elapsed += std::chrono::steady_clock::now() - this->t0;};
    inline void reset() {this->elapsed = std::chrono::steady_clock::duration::zero();};
    inline unsigned long int time_cost_hour() const { //h
        return std::chrono::duration_cast<std::chrono::hours>(this->elapsed).count();
    };
    inline unsigned long int time_cost_minute() const { //m
        return std::chrono::duration_cast<std::chrono::minutes>(this->elapsed).count();
    };
    inline unsigned long int time_cost_second() const { //s
        return std::chrono::duration_cast<std::chrono::seconds>(this->elapsed).count();
    };
    inline unsigned long int time_cost_millisecond() const { //ms
        return std::chrono::duration_cast<std::chrono::milliseconds>(this->elapsed).count();
    };
    inline unsigned long int time_cost_microsecond() const { //us
        return std::chrono::duration_cast<std::chrono::microseconds>(this->elapsed).count();
    };
    inline unsigned long int time_cost_nanosecond() const { //ns
        return std::chrono::duration_cast<std::chrono::nanoseconds>(this->elapsed).count();
    };
    inline double time_cost_millisecond_double() const {
        return std::chrono::duration<double, std::milli>(
            this->elapsed).count();
    }
    inline Timer& operator+=(const Timer& other) {
        this->elapsed += other.elapsed;
        return *this;
    }
    inline Timer operator+(const Timer& other) const {
        Timer result;
        result.elapsed = this->elapsed + other.elapsed;
        return result;
    }
};

#endif //_TIMER_H_
