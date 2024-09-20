//创建

#ifndef BLOCK_QUEUE_H_
#define BLOCK_QUEUE_H_

#define TASK_NUM 10
class CBlockQueue {
public:
    CBlockQueue();
    ~CBlockQueue();

    void BlockStop() {
        stopped_.store(true);
        cv_con_.notify_all();
    }

    bool BlockFull() {
        return tasks_.size() == capacity_ ? true : false;
    }

    size_t Size() {
        return tasks_.size();
    }

    bool BlockEmpty() {
        return tasks_.size() == 0 ? true : false;
    }

    bool BlockAvailable() {
        return !BlockStopped() || !BlockEmpty();
    }

    void BlockPush(MagicXEFrameV2 *data);
    void BlockPop(MagicXEFrameV2 **data);
    bool BlockPopEx(MagicXEFrameV2 **frame, size_t *index);

private:
    std::mutex mt_;
    std::condition_variable cv_con_;
    std::condition_variable cv_prod_;
    std::queue<MagicXEFrameV2 *> tasks_;
    std::atomic<bool> stopped_;
    const int capacity_;

    bool BlockStopped() {
        return stopped_.load();
    }
};








#endif