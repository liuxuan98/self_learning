#ifndef RW_LOCK_H_
#define RW_LOCK_H_

#include <condition_variable>
#include <mutex>

#include "magic_xe_common.h"


namespace magic_xe {

class RWLock;
template <typename _RWLockable>
class LockWriteGuard {
public:
    explicit LockWriteGuard(_RWLockable &rw_lockable) : rw_lockable_(rw_lockable) {
        rw_lockable_.lock_write();
    }
    ~LockWriteGuard() {
        rw_lockable_.release_write();
    }

private:
    LockWriteGuard() = delete;
    LockWriteGuard(const LockWriteGuard &) = delete;
    LockWriteGuard &operator=(const LockWriteGuard &) = delete;

private:
    _RWLockable &rw_lockable_;
};

template <typename _RWLockable>
class LockReadGuard {
public:
    explicit LockReadGuard(_RWLockable &rw_lockable) : rw_lockable_(rw_lockable) {
        rw_lockable_.lock_read();
    }
    ~LockReadGuard() {
        rw_lockable_.release_read();
    }

private:
    LockReadGuard() = delete;
    LockReadGuard(const LockReadGuard &) = delete;
    LockReadGuard &operator=(const LockReadGuard &) = delete;

private:
    _RWLockable &rw_lockable_;
};

class RWLock {
    friend LockWriteGuard<RWLock>;  //通过友元类,LockWriteGuard<RWLock>可以访问另外一个类(RWLock)的私有成员变量和私有成员函数.缺点：破坏类的封装,增加代码复杂;优点：提高代码的可重用性.
    friend LockReadGuard<RWLock>;

public:
    RWLock() = default;
    ~RWLock() = default;

private:
    void lock_read() {
        std::unique_lock<std::mutex> ulk(counter_mutex);
        cond_r.wait(ulk, [=]() -> bool { return write_cnt == 0; });
        ++read_cnt;
    }

    void lock_write() {
        std::unique_lock<std::mutex> ulk(counter_mutex);
        ++write_cnt;
        cond_w.wait(ulk, [=]() -> bool { return read_cnt == 0 && !inwriteflag; });
        inwriteflag = true;
    }

    void release_read() {
        std::unique_lock<std::mutex> ulk(counter_mutex);
        if (read_cnt <= 0) {
            MAGIC_XE_LOGE("read_cnt:%d is error", read_cnt);
            if (read_cnt < 0) {
                read_cnt = 0;
            }
            if (write_cnt > 0) {
                cond_w.notify_one();
            }
            return;
        }
        if (--read_cnt == 0 && write_cnt > 0) {
            cond_w.notify_one();
        }
    }

    void release_write() {
        std::unique_lock<std::mutex> ulk(counter_mutex);
        if (write_cnt <= 0) {
            MAGIC_XE_LOGE("write_cnt:%d is error", write_cnt);
            if (write_cnt < 0) {
                write_cnt = 0;
            }
            cond_r.notify_all();
            return;
        }
        if (--write_cnt == 0) {
            cond_r.notify_all();
        } else {
            cond_w.notify_one();
        }
        inwriteflag = false;
    }

private:
    volatile int read_cnt{0};
    volatile int write_cnt{0};
    volatile bool inwriteflag{false};
    std::mutex counter_mutex;
    std::condition_variable cond_w;
    std::condition_variable cond_r;
};

} // namespace magic_xe

#endif //RW_LOCK_H_