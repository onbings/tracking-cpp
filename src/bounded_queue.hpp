/**
 * @file bounded_queue.hpp
 * @brief Blocking, bounded, closable FIFO used to hand frames between pipeline threads.
 *
 * The bound provides back-pressure: a fast producer (e.g. the decoder) blocks when the consumer
 * falls behind, so memory stays constant (one 4K BGR frame is ~25 MB).
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace ft {

/**
 * @brief Thread-safe bounded queue with "close" semantics for end-of-stream.
 * @tparam T element type (moved in and out).
 *
 * - push() blocks while the queue is full, returns false once the queue is closed.
 * - pop() blocks while the queue is empty, returns std::nullopt once closed *and* drained.
 */
template <class T>
class BoundedQueue {
public:
    /// @param capacity maximum number of queued elements (at least 1).
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity ? capacity : 1) {}

    /**
     * @brief Appends an element, waiting for room if necessary.
     * @return false if the queue was closed (the element is dropped).
     */
    bool push(T value) {
        std::unique_lock lock(mutex_);
        notFull_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
        if (closed_) return false;
        items_.push_back(std::move(value));
        notEmpty_.notify_one();
        return true;
    }

    /**
     * @brief Removes the oldest element, waiting for one if necessary.
     * @return the element, or std::nullopt when the queue is closed and empty (end of stream).
     */
    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        notEmpty_.wait(lock, [&] { return closed_ || !items_.empty(); });
        if (items_.empty()) return std::nullopt;
        T value = std::move(items_.front());
        items_.pop_front();
        notFull_.notify_one();
        return value;
    }

    /// @brief Marks the end of the stream and wakes every waiting thread.
    void close() {
        {
            std::scoped_lock lock(mutex_);
            closed_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    /// @return current number of queued elements (for the HUD).
    [[nodiscard]] std::size_t size() const {
        std::scoped_lock lock(mutex_);
        return items_.size();
    }
    /// @return the capacity given at construction.
    [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::deque<T> items_;
    bool closed_ = false;
};

}  // namespace ft
