#include "cblock_queue.h"

CBlockQueue::CBlockQueue() : capacity_(TASK_NUM), stopped_(false) {
}


CBlockQueue::~CBlockQueue() {
    BlockStop();
    cv_con_.notify_all();
    cv_prod_.notify_all();
}

void CBlockQueue::BlockPush(MagicXEFrameV2 *frame) {
    std::unique_lock<std::mutex> lck(mt_); //单例对象,异步线程加锁，std::unique_lock<std::mutex> 实现自动自动加解锁.
    while (BlockFull()) { //队列满
        cv_con_.notify_one(); //(cv_con_条件变量)通知消费者消费
        cv_prod_.wait(lck); //等待条件变量被通知,锁会释放掉，被通知时，锁会重新获得.
    }

    tasks_.push(frame);
    cv_con_.notify_one();
}

void CBlockQueue::BlockPop(MagicXEFrameV2 **frame) {
    std::unique_lock<std::mutex> lck(mt_); //通知线程持有锁，避免数据竞争
    while (BlockEmpty()) {
        if (this->BlockStopped()) {
            return;
        }
        cv_prod_.notify_one();
        cv_con_.wait(lck, [this]() { return this->BlockStopped() || !this->BlockEmpty(); }); //条件成立条件变量不用等待
    }

    *frame = tasks_.front();
    tasks_.pop();
    cv_prod_.notify_one();
}

bool CBlockQueue::BlockPopEx(MagicXEFrameV2 **frame, size_t *size) {
    std::unique_lock<std::mutex> lck(mt_);
    if (BlockEmpty()) {
        *frame = NULL;
        return false;
    } else {
        *frame = tasks_.front();
        *size = tasks_.size();
        tasks_.pop();
        cv_prod_.notify_one();
        return true;
    }
}