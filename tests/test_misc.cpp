/**
 * @file test_misc.cpp
 * @brief Tests for small utilities: class parsing, NMS, line counting, bounded queue.
 */
#include "analytics.hpp"
#include "bounded_queue.hpp"
#include "detector.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace ft;

TEST(Misc, ParseClasses) {
    EXPECT_FALSE(parseClasses("all").has_value());
    const auto ids = parseClasses("person, 32");
    ASSERT_TRUE(ids.has_value());
    EXPECT_EQ(*ids, (std::vector<int>{0, 32}));
    EXPECT_THROW(parseClasses("dragon"), std::runtime_error);
}

TEST(Misc, NonMaxSuppression) {
    std::vector<Detection> dets = {
        {cv::Rect2f(0, 0, 100, 100), 0.9f, 0},
        {cv::Rect2f(10, 10, 100, 100), 0.8f, 0},   // overlaps the first -> removed
        {cv::Rect2f(500, 500, 50, 50), 0.7f, 0}};
    const auto kept = nonMaxSuppression(dets, 0.5f);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
}

TEST(Misc, LineCounterInOut) {
    LineCounter lc({0, 100}, {200, 100});
    EXPECT_EQ(lc.update(1, {50, 90}, 0), 0);
    EXPECT_EQ(lc.update(1, {50, 110}, 1), 1);   // crossed downward -> IN
    EXPECT_EQ(lc.update(1, {50, 95}, 2), -1);   // crossed back up -> OUT
    EXPECT_EQ(lc.countIn(), 1);
    EXPECT_EQ(lc.countOut(), 1);
}

TEST(Misc, BoundedQueueBlocksAndCloses) {
    // Blocking: a push into a full queue only returns once a consumer makes room.
    BoundedQueue<int> q(2);
    EXPECT_TRUE(q.push(1));
    EXPECT_TRUE(q.push(2));
    std::atomic<bool> pushReturned{false};
    std::thread producer([&] {
        q.push(3);  // blocks until the main thread pops one
        pushReturned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(pushReturned.load());  // still blocked (queue was full)
    EXPECT_EQ(q.pop().value(), 1);
    producer.join();
    EXPECT_TRUE(pushReturned.load());
    EXPECT_EQ(q.pop().value(), 2);
    EXPECT_EQ(q.pop().value(), 3);

    // Closing wakes a blocked consumer and drains the rest.
    q.close();
    EXPECT_FALSE(q.pop().has_value());
}
