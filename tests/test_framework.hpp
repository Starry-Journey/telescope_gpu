#pragma once

// 极简测试框架：不引入 GTest 依赖，保证 `cmake && make && ctest` 开箱即用。
// 以后如果要换成 GTest，测试用例的写法基本一一对应。

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int& failures() {
    static int count = 0;
    return count;
}

inline const char*& current() {
    static const char* name = "";
    return name;
}

inline void fail(const char* file, int line, const std::string& message) {
    ++failures();
    std::fprintf(stderr, "[FAIL] %s:%d [%s] %s\n", file, line, current(),
                 message.c_str());
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int run_all() {
    for (const auto& c : registry()) {
        current() = c.name;
        std::printf("[ RUN  ] %s\n", c.name);
        const int before = failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
        } catch (...) {
            fail(__FILE__, __LINE__, "uncaught unknown exception");
        }
        std::printf(failures() == before ? "[  OK  ] %s\n" : "[ FAIL ] %s\n", c.name);
    }
    std::printf("\n%d test(s), %d failure(s)\n",
                static_cast<int>(registry().size()), failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace test

#define TEST(name)                                                   \
    static void test_##name();                                       \
    static ::test::Registrar reg_##name(#name, test_##name);         \
    static void test_##name()

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) ::test::fail(__FILE__, __LINE__, #cond);         \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                        \
    do {                                                             \
        const double _a = (a);                                       \
        const double _b = (b);                                       \
        const double _t = (tol);                                     \
        if (!(std::fabs(_a - _b) <= _t)) {                           \
            std::ostringstream _os;                                  \
            _os << "|" << _a << " - " << _b << "| = "                \
                << std::fabs(_a - _b) << " > " << _t;                \
            ::test::fail(__FILE__, __LINE__, _os.str());             \
        }                                                            \
    } while (0)

#define CHECK_THROWS(expr)                                           \
    do {                                                             \
        bool _threw = false;                                         \
        try {                                                        \
            (void)(expr);                                            \
        } catch (...) {                                              \
            _threw = true;                                           \
        }                                                            \
        if (!_threw)                                                 \
            ::test::fail(__FILE__, __LINE__,                         \
                         "expected exception: " #expr);              \
    } while (0)

#define REQUIRE(cond)                                                \
    do {                                                             \
        if (!(cond)) {                                               \
            ::test::fail(__FILE__, __LINE__,                         \
                         std::string("REQUIRE failed: ") + #cond);   \
            throw std::runtime_error("REQUIRE failed");              \
        }                                                            \
    } while (0)
